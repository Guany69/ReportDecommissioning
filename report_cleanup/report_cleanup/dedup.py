"""Two-stage duplicate / consolidation detection.

Stage 1 (candidate matching): a cheap metadata + name-similarity gate decides
whether two reports are even worth a deeper look. This is what stops us from
comparing the field lists of every unrelated report.

Stage 2 (field comparison): only candidate pairs get their Report Fields
compared via Jaccard similarity and containment. Pairs that are similar enough
become edges; connected components form duplicate groups (DUP-0001, ...).

Blocking (name-prefix / data-source) keeps Stage 1 near-linear over thousands
of reports. RapidFuzz (already a dependency) provides name similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .clean import normalize_name, text
from .report_fields import calculate_field_containment, calculate_field_similarity


@dataclass
class DupGroup:
    group_id: str
    members: list[int]                 # report_uid list
    keeper_uid: int | None = None
    detection: str = "field_match"
    why: dict = field(default_factory=dict)


# ---- union-find -----------------------------------------------------------
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _eq(a, b) -> bool:
    """Exact (case-insensitive) match; blank on either side never matches."""
    return text(a) != "" and text(a).lower() == text(b).lower()


# ---- name similarity ------------------------------------------------------
def normalize_report_name(name, name_noise=None) -> str:
    """Normalized report name (lowercase, de-noised, punctuation stripped)."""
    return normalize_name(name, name_noise or [])


def calculate_name_similarity(name_a, name_b, name_noise=None) -> float:
    """Report-name similarity as a 0..100 percentage."""
    return float(fuzz.token_sort_ratio(
        normalize_report_name(name_a, name_noise),
        normalize_report_name(name_b, name_noise),
    ))


# ---- Stage 1: candidate matching ------------------------------------------
def candidate_similarity_score(a: dict, b: dict, cfg) -> float:
    """Weighted 0..100 metadata similarity (name is the biggest signal)."""
    d = cfg.dedup
    w = d["candidate_weights"]
    name_noise = cfg.clean["name_noise"]
    name_sim = calculate_name_similarity(a.get("report_name"), b.get("report_name"), name_noise)
    score = w["name"] * (name_sim / 100.0)
    if _eq(a.get("data_source"), b.get("data_source")):
        score += w["data_source"]
    if _eq(a.get("category"), b.get("category")):
        score += w["category"]
    if _eq(a.get("report_tag"), b.get("report_tag")):
        score += w["report_tag"]
    if _eq(a.get("report_type"), b.get("report_type")):
        score += w["report_type"]
    if _eq(a.get("worklet"), b.get("worklet")):
        score += w["worklet"]
    return round(score, 1)


def should_compare_report_fields(report_a: dict, report_b: dict, cfg) -> bool:
    """Stage-1 gate: only similar-enough reports earn a Report Fields comparison."""
    d = cfg.dedup
    name_noise = cfg.clean["name_noise"]
    name_sim = calculate_name_similarity(report_a.get("report_name"), report_b.get("report_name"), name_noise)

    if name_sim >= d["name_similarity_strong"]:
        return True
    if name_sim >= d["name_similarity_moderate"]:
        if _eq(report_a.get("data_source"), report_b.get("data_source")):
            return True
        if _eq(report_a.get("category"), report_b.get("category")) and \
           _eq(report_a.get("report_tag"), report_b.get("report_tag")):
            return True
    # Fall back to the blended candidate score.
    return candidate_similarity_score(report_a, report_b, cfg) >= d["candidate_score_threshold"]


# ---- Stage 2: classification ----------------------------------------------
def classify_field_match(field_similarity, field_containment, cfg) -> str:
    """Human label for a candidate pair based on Stage-2 metrics."""
    d = cfg.dedup
    sim = field_similarity or 0
    cont = field_containment or 0
    fc = d["field_containment"]
    fs = d["field_similarity"]
    if sim >= fs["nearly_identical"]:
        return "Nearly Identical Duplicate"
    if cont >= fc["contained"]:
        return "One Report Contained In Another"
    if sim >= fs["strong_duplicate"]:
        return "Strong Duplicate Candidate"
    if cont >= fc["strong_consolidation"]:
        return "Strong Consolidation Candidate"
    if sim >= fs["similar_review"]:
        return "Similar, Needs Review"
    return "Not A Match"


# Classification labels for pairs that passed Stage 1 but had no field data.
CLASSIFICATION_META_ONLY = "Metadata Similar - Fields Unavailable"
CLASSIFICATION_FIELD_EXPORT_MISSING = "Field Export Missing"
CLASSIFICATION_FIELD_MAPPING_UNAVAILABLE = "Field Mapping Unavailable"
CLASSIFICATION_CATALOG_ONLY = "Catalog Only - Cannot Compare Fields"


def _is_edge(sim, cont, d) -> bool:
    if sim is not None and sim >= d["group_min_field_similarity"]:
        return True
    if cont is not None and cont >= d["group_min_field_containment"]:
        return True
    return False


# ---- driver ---------------------------------------------------------------
def detect_duplicates(
    records: list[dict], cfg
) -> tuple[list[DupGroup], list[tuple[int, int]]]:
    """Return (confirmed_dup_groups, metadata_only_pairs).

    confirmed_dup_groups  — clusters where both reports had field sets and the
                            sets exceeded the similarity/containment thresholds.
    metadata_only_pairs   — pairs that passed Stage 1 (metadata gate) but could
                            not be field-compared because one or both reports have
                            an empty field set.  The caller should mark these as
                            CLASSIFICATION_META_ONLY and send to manual review.
    """
    d = cfg.dedup
    name_noise = cfg.clean["name_noise"]
    n = len(records)
    uf = _UF(n)

    # Field sets are stamped by attach_report_fields() before this is called.
    # Ensure every record has a defensive default so Stage 2 never KeyErrors.
    for r in records:
        if "report_fields_set" not in r:
            r["report_fields_set"] = set()

    # Blocking: name-prefix and data-source blocks keep Stage 1 near-linear.
    blocks: dict[str, list[int]] = {}
    plen = d["block_prefix_len"]
    for i, r in enumerate(records):
        nm = normalize_report_name(r.get("report_name"), name_noise)
        if nm:
            blocks.setdefault("pfx:" + nm[:plen], []).append(i)
        ds = text(r.get("data_source")).lower()
        if ds:
            blocks.setdefault("src:" + ds, []).append(i)

    seen: set[tuple[int, int]] = set()
    meta_only_pairs: list[tuple[int, int]] = []

    for idxs in blocks.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                pk = (i, j) if i < j else (j, i)
                if pk in seen:
                    continue
                seen.add(pk)
                ra, rb = records[i], records[j]
                if not should_compare_report_fields(ra, rb, cfg):
                    continue

                fa = ra.get("report_fields_set") or set()
                fb = rb.get("report_fields_set") or set()

                if not fa or not fb:
                    # Stage 1 passed but field evidence is absent — cannot confirm.
                    meta_only_pairs.append((ra["report_uid"], rb["report_uid"]))
                    continue

                sim = calculate_field_similarity(fa, fb)
                cont = calculate_field_containment(fa, fb)
                if _is_edge(sim, cont, d):
                    uf.union(i, j)

    # Collect clusters of size >= 2.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    groups: list[DupGroup] = []
    gnum = 0
    for members_idx in sorted(clusters.values(), key=lambda ids: min(ids)):
        if len(members_idx) < 2:
            continue
        gnum += 1
        gid = f"DUP-{gnum:04d}"
        member_uids = [records[i]["report_uid"] for i in members_idx]
        groups.append(DupGroup(group_id=gid, members=member_uids))

    return groups, meta_only_pairs
