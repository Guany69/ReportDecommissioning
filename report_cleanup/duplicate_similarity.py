"""Weighted duplicate similarity + relationship labels, with optional ML scoring.

This is duplicate EVIDENCE only — it never feeds the Overall Decommissioning
Score (see overall.py). For a pair of reports we blend up to nine configurable
components into a single 0..100 similarity. Components whose data is missing for
either report are dropped and the remaining weights are RENORMALIZED, so a report
with no field export is not unfairly dragged toward zero on every comparison.

Set-valued components (fields, business objects, built-in prompts, related
business objects, authorized usage) use Jaccard. Scalar components (data source,
report type) use exact normalized equality. Name similarity reuses the project's
RapidFuzz-based fuzzy matcher (evidence only — never used for the join). All nine
are computed by `ml.features`, which is also what the PyTorch model consumes.

When a `DuplicatePredictor` is supplied to `compute_duplicate_matches`, the
learned probability — not the weighted score — decides whether a candidate pair
is flagged. The weighted components are still computed and still published as the
reason trail, and the descriptive relationship label still comes from the
deterministic containment/Jaccard rules. The model answers "is this a duplicate?";
the rules answer "what kind of duplicate is it?".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clean import text
from .dedup import generate_candidate_pairs
from .ml.features import (FEATURE_NAMES, build_feature_vector, cheap_signals,
                          full_signals)
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

# Relationship used when the model flags a pair the deterministic rules did not
# characterize. Reuses an existing label rather than inventing a sixth one.
_ML_ONLY_RELATIONSHIP = "Possible Duplicate"

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
    raw_signals: dict[str, float | None] = field(default_factory=dict)  # all 9, pre-renormalization


def compute_duplicate_similarity(a: dict, b: dict, cfg,
                                 skip_ceiling_short_circuit: bool = False) -> DuplicateSimilarity:
    """Weighted similarity, relationship label, and reason trail for one pair.

    ``skip_ceiling_short_circuit`` forces every component to be computed. The
    short-circuit below is an optimization tied to the weighted formula's own
    'possible' threshold; the ML path needs all nine raw signals regardless of
    whether the weighted score could ever clear that threshold, so it opts out.
    """
    w = cfg.duplicate_weights
    th = cfg.duplicate_thresholds

    cheap = cheap_signals(a, b, cfg)
    field_jaccard = cheap.field_jaccard
    smaller_containment = cheap.smaller_containment
    name_sim = cheap.name_sim
    name_match = cheap.name_match

    # Ceiling short-circuit (skipped for name matches, which are always flagged below).
    # When both reports have fields, the field components carry most of the weight;
    # using the cheap field numbers + the real name score, the best the pair could
    # score (every remaining component perfect) is bounded. If that ceiling is below
    # the 'possible' threshold it can never be flagged, so skip the remaining
    # BO/prompt/related/auth comparisons. Correctness-preserving. An unavailable name
    # folds into other_w (counted at its 100 max) so the ceiling stays an upper bound.
    if not skip_ceiling_short_circuit and not name_match and field_jaccard is not None:
        total = sum(w.values()) or 1.0
        name_w = w.get("name", 0) if cheap.name_available else 0
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
                raw_signals={},
            )

    # value per component (None = unavailable -> renormalized away).
    values = full_signals(a, b, cheap)

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
        raw_signals=values,
    )


def _reset_duplicate_fields(records: list[dict], scoring_mode: str,
                            model_version: str | None, threshold: float | None,
                            model_status: str) -> None:
    for r in records:
        r["potential_duplicate"] = False
        r["potential_duplicate_of"] = None
        r["duplicate_similarity"] = None
        r["duplicate_relationship"] = "Not Flagged"
        r["field_jaccard_similarity"] = None
        r["smaller_report_containment"] = None
        r["duplicate_matches"] = []          # all qualified matches (highest first)
        r["duplicate_reason_trail"] = []
        # ML evidence. duplicate_scoring_mode is stamped even when ML is off so the
        # export/DB always states which path produced the verdict.
        r["duplicate_scoring_mode"] = scoring_mode
        r["duplicate_model_status"] = model_status
        r["duplicate_model_version"] = model_version
        r["duplicate_ml_probability"] = None
        r["duplicate_ml_threshold"] = threshold
        r["duplicate_ml_prediction"] = None
        r["duplicate_feature_values"] = None


def compute_duplicate_matches(records: list[dict], cfg, predictor=None, *,
                              scoring_status: str | None = None) -> dict[str, object]:
    """Stamp weighted (and optionally ML) duplicate evidence onto every record.

    For each candidate pair from the deterministic inverted-index blocking, the
    weighted similarity is computed. Pairs that qualify are retained as matches on
    BOTH reports and the highest-scoring match becomes each report's headline
    duplicate. This is evidence only — it never changes the Overall Score.

    Qualification depends on the scoring mode:

    * baseline (``predictor is None``) — the weighted score clears
      ``duplicate_thresholds.possible``, or the guarded name-match rule fires.
    * ML (``predictor`` supplied) — the model's probability for the pair is at or
      above the decision threshold. Candidate blocking is unchanged, so the model
      never sees an all-pairs comparison; it re-ranks the same candidate set the
      deterministic stage already produced.
    """
    use_ml = predictor is not None
    threshold = float(predictor.threshold) if use_ml else None
    _reset_duplicate_fields(
        records,
        scoring_mode="ml" if use_ml else "weighted_baseline",
        model_version=predictor.model_version if use_ml else None,
        threshold=threshold,
        model_status=scoring_status or ("pytorch" if use_ml else "disabled"),
    )

    max_matches = cfg.duplicate_thresholds.get("max_matches_per_report", 50)

    pairs = sorted(generate_candidate_pairs(records, cfg))
    sims = [
        compute_duplicate_similarity(records[i], records[j], cfg,
                                     skip_ceiling_short_circuit=use_ml)
        for i, j in pairs
    ]

    if use_ml:
        # One batched pass over every candidate pair — not one forward call per pair.
        vectors = [build_feature_vector(s.raw_signals) for s in sims]
        probabilities = predictor.predict(vectors)
    else:
        probabilities = [None] * len(pairs)

    # Highest probability the model assigned to ANY candidate pair involving each
    # report — recorded even when it never clears the threshold, so "scored and
    # rejected at 12%" is distinguishable from "never a candidate" (None).
    best_prob: dict[int, tuple[float, dict[str, float | None]]] = {}

    for (i, j), sim, prob in zip(pairs, sims, probabilities):
        a, b = records[i], records[j]
        model_features = dict(zip(
            FEATURE_NAMES, build_feature_vector(sim.raw_signals), strict=True))
        if use_ml:
            for rec in (a, b):
                uid = rec["report_uid"]
                if uid not in best_prob or prob > best_prob[uid][0]:
                    best_prob[uid] = (prob, model_features)
        if use_ml:
            qualified = predictor.is_duplicate(prob)
            relationship = sim.relationship if sim.relationship != "Not Flagged" else _ML_ONLY_RELATIONSHIP
        else:
            qualified = sim.potential_duplicate
            relationship = sim.relationship
        if not qualified:
            continue
        for src, other in ((a, b), (b, a)):
            src["duplicate_matches"].append({
                "other_uid": other["report_uid"],
                "other_report_name": text(other.get("report_name")),
                "similarity": sim.overall,
                "relationship": relationship,
                "field_jaccard_similarity": sim.field_jaccard,
                "smaller_report_containment": sim.smaller_containment,
                "ml_probability": None if prob is None else float(prob),
                "feature_values": model_features,
                "_reason_trail": list(sim.reasons),
            })

    for r in records:
        matches = sorted(r["duplicate_matches"],
                         key=lambda m: (m.get("ml_probability") if use_ml else m["similarity"]) or 0.0,
                         reverse=True)
        best_reason_trail = list(matches[0]["_reason_trail"]) if matches else []
        for match in matches:
            match.pop("_reason_trail", None)
        r["duplicate_matches"] = matches[:max_matches]   # bound memory on huge clusters
        if not matches:
            continue
        best = matches[0]
        r["potential_duplicate"] = True
        r["potential_duplicate_of"] = best["other_report_name"]
        r["duplicate_similarity"] = best["similarity"]
        r["duplicate_relationship"] = best["relationship"]
        r["field_jaccard_similarity"] = best["field_jaccard_similarity"]
        r["smaller_report_containment"] = best["smaller_report_containment"]
        trail = best_reason_trail
        if use_ml and best["ml_probability"] is not None:
            # The model probability leads; the weighted components stay underneath as
            # supporting evidence. They are context for a reviewer, NOT a causal
            # explanation of what the network did.
            trail.insert(0, Reason(
                "duplicate",
                f"PyTorch duplicate model probability {best['ml_probability'] * 100:.1f}% "
                f"(decision threshold {threshold * 100:.1f}%).",
                None))
        r["duplicate_reason_trail"] = trail
        r["duplicate_feature_values"] = best["feature_values"]

    if use_ml:
        for r in records:
            best_candidate = best_prob.get(r["report_uid"])
            if best_candidate is None:
                continue        # never generated as a candidate — nothing was scored
            prob, raw = best_candidate
            r["duplicate_ml_probability"] = float(prob)
            r["duplicate_ml_prediction"] = predictor.is_duplicate(prob)
            r["duplicate_feature_values"] = dict(raw)

    return {
        "candidate_pair_count": len(pairs),
        "pairs_scored_by_ml": len(pairs) if use_ml else 0,
        "qualified_pair_count": sum(len(r["duplicate_matches"]) for r in records) // 2,
        "duplicate_scoring_mode": "ml" if use_ml else "weighted_baseline",
        "model_status": scoring_status or ("pytorch" if use_ml else "disabled"),
        "model_version": predictor.model_version if use_ml else None,
        "decision_threshold": threshold,
        **(predictor.stats.as_dict() if use_ml else {}),
    }


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
