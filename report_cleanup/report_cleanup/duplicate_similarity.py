"""Weighted duplicate similarity + relationship labels.

This is duplicate EVIDENCE only — it never feeds the Overall Decommissioning
Score (see overall.py). For a pair of reports we blend up to nine configurable
components into a single 0..100 similarity. Components whose data is missing for
either report are dropped and the remaining weights are RENORMALIZED, so a report
with no field export is not unfairly dragged toward zero on every comparison.

Set-valued components (fields, business objects, built-in prompts, related
business objects, authorized usage) use Jaccard. Scalar components (data source,
report type) use exact normalized equality. Name similarity reuses the project's
RapidFuzz-based fuzzy matcher (evidence only — never used for the join).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clean import text
from rapidfuzz import fuzz

from .dedup import (generate_candidate_pairs, is_strong_name_match,
                    normalize_name_for_similarity)
from .soft_scoring import Reason

# Component -> human label for the reason trail.
_COMPONENT_LABELS = {
    "field_jaccard": "Field Jaccard similarity",
    "smaller_containment": "Smaller-report containment",
    "business_object": "Business Object overlap",
    "name": "Name similarity",
    "built_in_prompts": "Built-in Prompt similarity",
    "related_business_object": "Related Business Object overlap",
    "data_source": "Data Source match",
    "authorized_usage": "Authorized Usage match",
    "report_type": "Report Type match",
}


@dataclass
class DuplicateSimilarity:
    overall: float                       # 0..100
    relationship: str
    potential_duplicate: bool
    field_jaccard: float | None          # 0..100 or None (unavailable)
    smaller_containment: float | None    # 0..100 or None
    components: dict[str, float] = field(default_factory=dict)   # available components, 0..100
    weights_used: dict[str, float] = field(default_factory=dict)  # renormalized weights
    reasons: list[Reason] = field(default_factory=list)


def _set_jaccard(a: set, b: set) -> float | None:
    """Jaccard (0..100). Unavailable (None) if either set is empty."""
    if not a or not b:
        return None
    return 100.0 * len(a & b) / len(a | b)


def _set_containment(a: set, b: set) -> float | None:
    """Containment over the smaller set (0..100). Unavailable if either empty."""
    if not a or not b:
        return None
    return 100.0 * len(a & b) / min(len(a), len(b))


def _scalar_eq(a, b) -> float | None:
    """100 if both present and normalized-equal, else 0. Unavailable if either blank."""
    sa, sb = text(a).casefold(), text(b).casefold()
    if not sa or not sb:
        return None
    return 100.0 if sa == sb else 0.0


def _fields(r: dict) -> set:
    return r.get("report_fields_set") or set()


def compute_duplicate_similarity(a: dict, b: dict, cfg) -> DuplicateSimilarity:
    w = cfg.duplicate_weights
    th = cfg.duplicate_thresholds
    name_noise = cfg.clean.get("name_noise", [])

    fa, fb = _fields(a), _fields(b)
    field_jaccard = _set_jaccard(fa, fb)
    smaller_containment = _set_containment(fa, fb)

    # Name similarity (de-noised) is computed up front: it both (a) lets a strong
    # name match flag a duplicate on its own and (b) tightens the short-circuit.
    # name_sim is the headline SCORE; name_match is the guarded VERDICT (empty and
    # year/ID-variant names score high but must not flag — see is_strong_name_match).
    # When either name de-noises to empty the component is UNAVAILABLE (None) — two
    # empty names score 100 in RapidFuzz, which would otherwise leak a false 100%
    # into the blend and label nameless reports "Nearly Identical".
    na = normalize_name_for_similarity(a.get("report_name"), name_noise)
    nb = normalize_name_for_similarity(b.get("report_name"), name_noise)
    name_available = bool(na and nb)
    name_sim = float(fuzz.token_sort_ratio(na, nb)) if name_available else None
    name_match = is_strong_name_match(
        a.get("report_name"), b.get("report_name"), name_noise, th.get("name_match", 90))

    # Ceiling short-circuit (skipped for name matches, which are always flagged below).
    # When both reports have fields, the field components carry most of the weight;
    # using the cheap field numbers + the real name score, the best the pair could
    # score (every remaining component perfect) is bounded. If that ceiling is below
    # the 'possible' threshold it can never be flagged, so skip the remaining
    # BO/prompt/related/auth comparisons. Correctness-preserving. An unavailable name
    # folds into other_w (counted at its 100 max) so the ceiling stays an upper bound.
    if not name_match and field_jaccard is not None:
        total = sum(w.values()) or 1.0
        name_w = w.get("name", 0) if name_available else 0
        other_w = total - w.get("field_jaccard", 0) - w.get("smaller_containment", 0) - name_w
        ceiling = (w.get("field_jaccard", 0) * field_jaccard
                   + w.get("smaller_containment", 0) * smaller_containment
                   + name_w * (name_sim or 0.0) + other_w * 100.0) / total
        if ceiling < th.get("possible", 70):
            return DuplicateSimilarity(
                overall=0.0, relationship="Not Flagged", potential_duplicate=False,
                field_jaccard=round(field_jaccard, 1),
                smaller_containment=round(smaller_containment, 1),
                components={}, weights_used={}, reasons=[],
            )

    # value per component (None = unavailable -> renormalized away).
    values: dict[str, float | None] = {
        "field_jaccard": field_jaccard,
        "smaller_containment": smaller_containment,
        "business_object": _set_jaccard(a.get("business_objects_set") or set(),
                                        b.get("business_objects_set") or set()),
        "name": name_sim,
        "built_in_prompts": _set_jaccard(a.get("built_in_prompts_set") or set(),
                                         b.get("built_in_prompts_set") or set()),
        "related_business_object": _set_jaccard(a.get("related_bos_set") or set(),
                                                b.get("related_bos_set") or set()),
        "data_source": _scalar_eq(a.get("data_source"), b.get("data_source")),
        "authorized_usage": _set_jaccard(a.get("authorized_usage_set") or set(),
                                         b.get("authorized_usage_set") or set()),
        "report_type": _scalar_eq(a.get("report_type"), b.get("report_type")),
    }

    available = {k: v for k, v in values.items() if v is not None and k in w}
    weight_sum = sum(w[k] for k in available) or 1.0
    weights_used = {k: w[k] / weight_sum for k in available}
    overall = sum(weights_used[k] * available[k] for k in available)
    overall = round(overall, 1)

    relationship, potential = _classify(overall, smaller_containment, th)

    # Name-based duplicate: a very high de-noised name match (e.g. "Copy of X" vs
    # "X", "(Old) X", or exact repeats) flags the pair EVEN when fields don't overlap
    # — copies often live in their own Fields-export rows. If field evidence didn't
    # already produce a stronger verdict, label it a name match and surface the name
    # score as the headline similarity.
    if name_match and not potential:
        potential = True
        relationship = "Likely Duplicate (Name Match)"
        overall = max(overall, round(name_sim, 1))

    reasons = [Reason("duplicate",
                      f"Weighted duplicate similarity {overall:.1f}% ({relationship})", None)]
    for k, v in available.items():
        reasons.append(Reason("duplicate_component",
                               f"{_COMPONENT_LABELS[k]}: {v:.0f}% (weight {weights_used[k]*100:.0f}%)", None))

    return DuplicateSimilarity(
        overall=overall,
        relationship=relationship,
        potential_duplicate=potential,
        field_jaccard=None if field_jaccard is None else round(field_jaccard, 1),
        smaller_containment=None if smaller_containment is None else round(smaller_containment, 1),
        components={k: round(v, 1) for k, v in available.items()},
        weights_used={k: round(v, 4) for k, v in weights_used.items()},
        reasons=reasons,
    )


def compute_duplicate_matches(records: list[dict], cfg) -> None:
    """Stamp weighted duplicate evidence onto every record.

    For each candidate pair (inverted-index generated) the weighted similarity is
    computed; pairs at/above the 'possible' threshold are retained as matches on
    BOTH reports. The highest-scoring match becomes each report's headline
    duplicate. This is evidence only — it never changes the Overall Score.
    """
    for r in records:
        r["potential_duplicate"] = False
        r["potential_duplicate_of"] = None
        r["duplicate_similarity"] = None
        r["duplicate_relationship"] = "Not Flagged"
        r["field_jaccard_similarity"] = None
        r["smaller_report_containment"] = None
        r["duplicate_matches"] = []          # all qualified matches (highest first)
        r["duplicate_reason_trail"] = []

    by_uid = {r["report_uid"]: r for r in records}
    max_matches = cfg.duplicate_thresholds.get("max_matches_per_report", 50)

    for i, j in generate_candidate_pairs(records, cfg):
        a, b = records[i], records[j]
        sim = compute_duplicate_similarity(a, b, cfg)
        if not sim.potential_duplicate:
            continue
        for src, other in ((a, b), (b, a)):
            src["duplicate_matches"].append({
                "other_uid": other["report_uid"],
                "other_report_name": text(other.get("report_name")),
                "similarity": sim.overall,
                "relationship": sim.relationship,
                "field_jaccard_similarity": sim.field_jaccard,
                "smaller_report_containment": sim.smaller_containment,
            })

    for r in records:
        matches = sorted(r["duplicate_matches"], key=lambda m: m["similarity"], reverse=True)
        r["duplicate_matches"] = matches[:max_matches]   # bound memory on huge clusters
        if matches:
            best = matches[0]
            r["potential_duplicate"] = True
            r["potential_duplicate_of"] = best["other_report_name"]
            r["duplicate_similarity"] = best["similarity"]
            r["duplicate_relationship"] = best["relationship"]
            r["field_jaccard_similarity"] = best["field_jaccard_similarity"]
            r["smaller_report_containment"] = best["smaller_report_containment"]
            other = by_uid.get(best["other_uid"])   # O(1) lookup, not O(n)
            sim = compute_duplicate_similarity(r, other, cfg) if other else None
            r["duplicate_reason_trail"] = sim.reasons if sim else []


def _classify(overall: float, containment: float | None, th: dict) -> tuple[str, bool]:
    strong = th.get("strong", 90)
    likely = th.get("likely", 80)
    possible = th.get("possible", 70)
    cont_rel = th.get("containment_relationship", 95)
    rel_min = th.get("relationship_min_overall", 80)

    if overall >= strong:
        return "Nearly Identical", True
    if containment is not None and containment >= cont_rel and overall >= rel_min:
        return "Smaller Report Contained in Larger Report", True
    if overall >= likely:
        return "High Field Overlap", True
    if overall >= possible:
        return "Possible Duplicate", True
    return "Not Flagged", False
