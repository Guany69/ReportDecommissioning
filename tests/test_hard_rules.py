from report_cleanup.hard_rules import apply_hard_rules


def test_dnu_in_name():
    hit = apply_hard_rules({"report_name": "2018 Headcount DNU"})
    assert hit and hit.rule_id == "dnu_in_name"
    assert hit.reason == "Report name contains DNU."


def test_dnu_case_insensitive_substring():
    assert apply_hard_rules({"report_name": "old report dnu copy"}) is not None


def test_orphan_worklet():
    hit = apply_hard_rules({"report_name": "Tile", "worklet": "Yes",
                            "landing_page": "", "areas_used": ""})
    assert hit and hit.rule_id == "orphan_worklet"


def test_worklet_with_landing_is_not_orphan():
    assert apply_hard_rules({"report_name": "Tile", "worklet": "Yes",
                             "landing_page": "Home", "areas_used": ""}) is None


def test_no_hard_rule():
    assert apply_hard_rules({"report_name": "Clean Report", "worklet": "No"}) is None


def test_dnu_word_boundary_no_false_positive():
    # "Kidnumber" contains the substring "dnu" but not the standalone word DNU.
    assert apply_hard_rules({"report_name": "Kidnumber Tracker"}) is None


def test_dnu_word_boundary_matches_standalone_token():
    hit = apply_hard_rules({"report_name": "Headcount DNU 2019"})
    assert hit and hit.rule_id == "dnu_in_name"
    assert hit.name == "DNU in Name"


def test_orphan_worklet_requires_both_landing_columns_blank():
    # A populated standalone Landing Page means it is reachable -> not orphaned.
    assert apply_hard_rules({"report_name": "T", "worklet": "Yes", "landing_page": "",
                             "report_landing_page": "Home", "areas_used": ""}) is None
    # Both landing columns blank AND no area where used -> orphaned.
    hit = apply_hard_rules({"report_name": "T", "worklet": "Yes", "landing_page": "",
                            "report_landing_page": "", "areas_used": ""})
    assert hit and hit.rule_id == "orphan_worklet"
