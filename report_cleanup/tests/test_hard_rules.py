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
