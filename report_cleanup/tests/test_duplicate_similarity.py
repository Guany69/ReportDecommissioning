"""Weighted duplicate similarity: components, renormalization, thresholds, labels."""
from report_cleanup.duplicate_similarity import (compute_duplicate_matches,
                                                 compute_duplicate_similarity)


def rep(uid, name, fields=None, bos=None, prompts=None, related=None,
        auth=None, data_source="DS", report_type="Advanced"):
    return dict(
        report_uid=uid, report_name=name,
        report_fields_set=set(fields or []),
        business_objects_set=set(bos or []),
        built_in_prompts_set=set(prompts or []),
        related_bos_set=set(related or []),
        authorized_usage_set=set(auth or []),
        data_source=data_source, report_type=report_type,
    )


def test_exact_field_duplicate_is_nearly_identical(cfg):
    a = rep(0, "Active Worker Report", fields={"f1", "f2", "f3"}, bos={"worker"})
    b = rep(1, "Active Worker Report", fields={"f1", "f2", "f3"}, bos={"worker"})
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.field_jaccard == 100.0
    assert sim.overall >= 90
    assert sim.relationship == "Nearly Identical"
    assert sim.potential_duplicate is True


def test_high_jaccard_partial_overlap(cfg):
    a = rep(0, "Worker Report", fields={"f1", "f2", "f3", "f4"})
    b = rep(1, "Worker Report", fields={"f1", "f2", "f3", "z"})
    sim = compute_duplicate_similarity(a, b, cfg)
    # 3 shared / 5 union = 60% jaccard
    assert sim.field_jaccard == 60.0


def test_smaller_report_contained_relationship(cfg):
    # b's fields are a strict subset of a's -> containment 100%; tune other signals
    # so overall clears the relationship floor (80) but stays below Nearly Identical.
    a = rep(0, "Big Report", fields={"f1", "f2", "f3", "f4", "f5", "f6"},
            bos={"worker"}, data_source="DS", report_type="Advanced")
    b = rep(1, "Big Report", fields={"f1", "f2", "f3", "f4", "f5"},
            bos={"worker"}, data_source="DS", report_type="Advanced")
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.smaller_containment == 100.0
    assert sim.relationship in ("Smaller Report Contained in Larger Report", "Nearly Identical")


def test_both_field_sets_empty_field_component_unavailable(cfg):
    a = rep(0, "Report One", fields=set(), data_source="DS", report_type="T")
    b = rep(1, "Report One", fields=set(), data_source="DS", report_type="T")
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.field_jaccard is None
    assert sim.smaller_containment is None
    assert "field_jaccard" not in sim.components       # dropped from the blend
    # Remaining weights (name/data_source/report_type) renormalize to sum 1.0.
    assert abs(sum(sim.weights_used.values()) - 1.0) < 1e-6


def test_missing_components_renormalize_weights(cfg):
    # Only name + data_source + report_type available (no field/BO/prompt data).
    a = rep(0, "Quarterly Finance", fields=set(), bos=set(), data_source="Fin", report_type="Adv")
    b = rep(1, "Quarterly Finance", fields=set(), bos=set(), data_source="Fin", report_type="Adv")
    sim = compute_duplicate_similarity(a, b, cfg)
    assert set(sim.components) <= {"name", "data_source", "report_type"}
    assert abs(sum(sim.weights_used.values()) - 1.0) < 1e-6
    assert sim.overall == 100.0  # all available components agree perfectly


def test_low_field_overlap_short_circuits_to_not_flagged(cfg):
    # Shares 1 of ~19 union fields (jaccard ~5%, containment ~10%); even with a
    # perfect name/data-source/type match the ceiling stays below 'possible', so the
    # pair is Not Flagged via the short-circuit. Field numbers are still reported.
    # Dissimilar names so the field short-circuit is what's under test (a strong
    # name match would flag it regardless).
    a = rep(0, "Apples Quarterly Dashboard", fields={f"a{i}" for i in range(9)} | {"shared"},
            data_source="DS", report_type="T")
    b = rep(1, "Oranges Annual Listing", fields={f"b{i}" for i in range(9)} | {"shared"},
            data_source="DS", report_type="T")
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.relationship == "Not Flagged"
    assert sim.potential_duplicate is False
    assert sim.field_jaccard is not None       # field overlap still surfaced
    assert sim.components == {}                 # expensive components skipped


def test_below_possible_threshold_not_flagged(cfg):
    a = rep(0, "Apples Report", fields={"f1"}, data_source="A", report_type="X")
    b = rep(1, "Oranges Listing", fields={"z9"}, data_source="B", report_type="Y")
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.relationship == "Not Flagged"
    assert sim.potential_duplicate is False


def test_compute_matches_stamps_best_and_all(cfg):
    a = rep(0, "Worker Report", fields={"f1", "f2", "f3"}, bos={"worker"})
    b = rep(1, "Worker Report", fields={"f1", "f2", "f3"}, bos={"worker"})
    c = rep(2, "Totally Different", fields={"x1"}, bos={"finance"}, data_source="Z")
    records = [a, b, c]
    compute_duplicate_matches(records, cfg)
    assert a["potential_duplicate"] is True
    assert a["potential_duplicate_of"] == "Worker Report"
    assert a["duplicate_similarity"] >= 90
    assert c["potential_duplicate"] is False


# ---- name-match guard regressions -----------------------------------------
def test_empty_names_are_unavailable_not_a_match(cfg):
    """Two empty/noise-only names de-noise to "" (RapidFuzz scores them 100). The
    name component must be UNAVAILABLE, not a false 100% — otherwise nameless
    reports get labelled Nearly Identical."""
    # Differing data_source/report_type so only the (empty) name is in question —
    # otherwise those default-equal components would flag the pair on their own.
    for nm in ("", "Copy", "Report"):
        a = rep(0, nm, fields=set(), data_source="A", report_type="X")
        b = rep(1, nm, fields=set(), data_source="B", report_type="Y")
        sim = compute_duplicate_similarity(a, b, cfg)
        assert "name" not in sim.components, f"name should be unavailable for {nm!r}"
        assert sim.potential_duplicate is False
        assert sim.relationship == "Not Flagged"


def test_year_variant_names_not_flagged(cfg):
    """Year/ID variants score >90 on the char ratio but name different reports —
    the differing numeric tokens must prevent a name-match verdict."""
    a = rep(0, "Annual Review 2017", fields={"a1", "a2"})
    b = rep(1, "Annual Review 2018", fields={"b1", "b2"})
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.relationship != "Likely Duplicate (Name Match)"
    assert sim.potential_duplicate is False


def test_same_year_copy_is_name_match(cfg):
    """A copy keeps the original's year, so the guard still lets it flag on name
    alone even with no shared fields."""
    a = rep(0, "2017 Year End Summary", fields={"a1"})
    b = rep(1, "Copy of 2017 Year End Summary", fields={"z9"})
    sim = compute_duplicate_similarity(a, b, cfg)
    assert sim.relationship == "Likely Duplicate (Name Match)"
    assert sim.potential_duplicate is True
