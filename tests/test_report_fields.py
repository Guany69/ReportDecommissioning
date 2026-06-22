"""Report Fields parsing, normalization, Jaccard similarity and containment."""
from report_cleanup.report_fields import (calculate_field_containment,
                                         calculate_field_similarity,
                                         normalize_report_field,
                                         parse_report_fields)

ALIASES = {
    "worker id": "employee id",
    "emp id": "employee id",
    "cost centre": "cost center",
}


def test_parse_supports_separators():
    assert parse_report_fields("A, B; C | D\nE") == {"a", "b", "c", "d", "e"}


def test_parse_empty():
    assert parse_report_fields("") == set()
    assert parse_report_fields(None) == set()


def test_normalize_underscores_and_punct():
    assert normalize_report_field("Employee_ID!!") == "employee id"


def test_normalize_applies_alias():
    assert normalize_report_field("Worker ID", ALIASES) == "employee id"
    assert normalize_report_field("Cost Centre", ALIASES) == "cost center"


def test_parse_with_alias_collapses_synonyms():
    a = parse_report_fields("Worker ID, Name", ALIASES)
    b = parse_report_fields("Emp ID, Name", ALIASES)
    assert a == b == {"employee id", "name"}


def test_field_similarity_is_jaccard():
    a = {"x", "y", "z"}
    b = {"y", "z", "w"}
    # shared 2, union 4 -> 50%
    assert calculate_field_similarity(a, b) == 50.0


def test_field_similarity_identical():
    a = {"x", "y"}
    assert calculate_field_similarity(a, set(a)) == 100.0


def test_field_containment_uses_smaller_count():
    smaller = {"a", "b"}
    larger = {"a", "b", "c", "d", "e"}
    # shared 2 / min(2,5) = 100%
    assert calculate_field_containment(smaller, larger) == 100.0
    assert calculate_field_similarity(smaller, larger) == 40.0


def test_similarity_none_when_empty():
    assert calculate_field_similarity(set(), {"a"}) is None
    assert calculate_field_containment(set(), {"a"}) is None
