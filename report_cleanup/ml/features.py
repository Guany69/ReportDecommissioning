"""Canonical pairwise duplicate signals — the single source of truth shared by
the deterministic weighted baseline and the PyTorch classifier.

This module is deliberately **torch-free**. `duplicate_similarity.py` imports the
primitives below, so the nine numbers the weighted formula blends and the nine
numbers the model sees are computed by exactly the same code. That is what makes
the baseline-vs-model comparison in `training/evaluate_duplicate_model.py`
meaningful rather than a comparison of two different feature pipelines.

Two representations exist:

*Raw signals* — nine values on a 0..100 percent scale, or ``None`` when the
underlying data is unavailable for either report. This is what the weighted
formula consumes (it renormalizes the missing components away) and what
`training/generate_pairs.py` writes into the reviewer CSV, because percentages
are what a human reviewer reads.

*Feature vector* — the model input. Each raw signal becomes **two** numbers: the
value scaled to 0.0..1.0 (0.0 when unavailable) and a missingness indicator
(1.0 when unavailable). Nine signals therefore produce an 18-dimensional vector.
Collapsing "no data" onto a plain 0.0 would teach the model that a report with no
field export is maximally dissimilar to everything, which is the opposite of what
the renormalizing baseline does.

``FEATURE_NAMES`` ordering is part of the model contract: it is persisted into the
artifact metadata and re-validated at load time (see `artifact.py`).
"""
from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from ..clean import text
from ..dedup import is_strong_name_match, normalize_name_for_similarity

# Bump when the meaning, ordering, or count of RAW_SIGNAL_NAMES changes. Artifacts
# trained under an older schema are then rejected at load instead of silently
# scoring a differently-shaped vector.
FEATURE_SCHEMA_VERSION = "1"

# The nine pairwise components, in canonical order. These mirror the keys of
# cfg.duplicate_weights exactly — deliberately, so neither side of the
# baseline-vs-model evaluation gets signals the other cannot see.
RAW_SIGNAL_NAMES: tuple[str, ...] = (
    "field_jaccard",
    "smaller_containment",
    "business_object",
    "name",
    "built_in_prompts",
    "related_business_object",
    "data_source",
    "authorized_usage",
    "report_type",
)

MISSING_SUFFIX = "_missing"

# 18-dim model input: value, then indicator, per signal. Order is contractual.
FEATURE_NAMES: tuple[str, ...] = tuple(
    part
    for name in RAW_SIGNAL_NAMES
    for part in (name, name + MISSING_SUFFIX)
)

FEATURE_COUNT = len(FEATURE_NAMES)


# ---- primitives (moved here from duplicate_similarity so both paths share them)
def set_jaccard(a: set, b: set) -> float | None:
    """Jaccard (0..100). Unavailable (None) if either set is empty."""
    if not a or not b:
        return None
    return 100.0 * len(a & b) / len(a | b)


def set_containment(a: set, b: set) -> float | None:
    """Containment over the smaller set (0..100). Unavailable if either empty."""
    if not a or not b:
        return None
    return 100.0 * len(a & b) / min(len(a), len(b))


def scalar_eq(a: Any, b: Any) -> float | None:
    """100 if both present and normalized-equal, else 0. Unavailable if either blank."""
    sa, sb = text(a).casefold(), text(b).casefold()
    if not sa or not sb:
        return None
    return 100.0 if sa == sb else 0.0


def report_field_set(r: dict) -> set:
    return r.get("report_fields_set") or set()


# ---- signal extraction ----------------------------------------------------
class CheapSignals:
    """The three signals needed before the weighted formula's ceiling
    short-circuit can be evaluated, plus the guarded name verdict.

    Computed separately from the remaining six because the baseline uses them to
    decide whether the expensive set comparisons are worth running at all. The ML
    path does not short-circuit (see `full_signals`), but reuses the same numbers.
    """

    __slots__ = ("field_jaccard", "smaller_containment", "name_sim",
                 "name_available", "name_match")

    def __init__(self, field_jaccard: float | None, smaller_containment: float | None,
                 name_sim: float | None, name_available: bool, name_match: bool) -> None:
        self.field_jaccard = field_jaccard
        self.smaller_containment = smaller_containment
        self.name_sim = name_sim
        self.name_available = name_available
        self.name_match = name_match


def cheap_signals(a: dict, b: dict, cfg) -> CheapSignals:
    """Field Jaccard / containment / name similarity for one candidate pair.

    ``name`` is UNAVAILABLE when either name de-noises to empty: RapidFuzz scores
    two empty strings 100, which would otherwise leak a false perfect match for
    nameless reports. ``name_match`` is the *guarded* verdict from
    `dedup.is_strong_name_match` (rejects empty names and year/ID variants) and is
    a different question from the raw score.
    """
    name_noise = cfg.clean.get("name_noise", [])
    th = cfg.duplicate_thresholds

    fa, fb = report_field_set(a), report_field_set(b)
    na = normalize_name_for_similarity(a.get("report_name"), name_noise)
    nb = normalize_name_for_similarity(b.get("report_name"), name_noise)
    name_available = bool(na and nb)

    return CheapSignals(
        field_jaccard=set_jaccard(fa, fb),
        smaller_containment=set_containment(fa, fb),
        name_sim=float(fuzz.token_sort_ratio(na, nb)) if name_available else None,
        name_available=name_available,
        name_match=is_strong_name_match(
            a.get("report_name"), b.get("report_name"), name_noise, th.get("name_match", 90)),
    )


def full_signals(a: dict, b: dict, cheap: CheapSignals) -> dict[str, float | None]:
    """All nine raw signals (0..100, or None when unavailable) for a pair.

    Takes the already-computed cheap signals so neither caller pays for the field
    and name comparisons twice.
    """
    return {
        "field_jaccard": cheap.field_jaccard,
        "smaller_containment": cheap.smaller_containment,
        "business_object": set_jaccard(a.get("business_objects_set") or set(),
                                       b.get("business_objects_set") or set()),
        "name": cheap.name_sim,
        "built_in_prompts": set_jaccard(a.get("built_in_prompts_set") or set(),
                                        b.get("built_in_prompts_set") or set()),
        "related_business_object": set_jaccard(a.get("related_bos_set") or set(),
                                               b.get("related_bos_set") or set()),
        "data_source": scalar_eq(a.get("data_source"), b.get("data_source")),
        "authorized_usage": set_jaccard(a.get("authorized_usage_set") or set(),
                                        b.get("authorized_usage_set") or set()),
        "report_type": scalar_eq(a.get("report_type"), b.get("report_type")),
    }


def extract_raw_signals(a: dict, b: dict, cfg) -> dict[str, float | None]:
    """Convenience wrapper: the nine raw signals for a pair, no short-circuit."""
    return full_signals(a, b, cheap_signals(a, b, cfg))


def build_feature_vector(raw: dict[str, float | None]) -> list[float]:
    """Nine raw 0..100-or-None signals -> the 18-dim model input.

    Value scaled to 0..1 (clamped, since a malformed input percentage must never
    hand the model an out-of-range activation) with 0.0 standing in for missing,
    immediately followed by the 0/1 missingness indicator.
    """
    out: list[float] = []
    for name in RAW_SIGNAL_NAMES:
        v = raw.get(name)
        if v is None:
            out.append(0.0)
            out.append(1.0)
        else:
            out.append(min(1.0, max(0.0, float(v) / 100.0)))
            out.append(0.0)
    return out


def named_feature_vector(raw: dict[str, float | None]) -> dict[str, float]:
    """The same 18 numbers as `build_feature_vector`, keyed by feature name.

    Used for the human-facing evidence column and for the reviewer CSV, where a
    bare list of 18 floats would be unreadable. Never used as model input — the
    model consumes the ordered list, so the two can never disagree about order.
    """
    return dict(zip(FEATURE_NAMES, build_feature_vector(raw)))


def feature_vector_for_pair(a: dict, b: dict, cfg) -> list[float]:
    """End-to-end: two report records -> the 18-dim model input."""
    return build_feature_vector(extract_raw_signals(a, b, cfg))
