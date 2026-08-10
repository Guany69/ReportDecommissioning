"""Leakage-safe connected-component split tests."""
from __future__ import annotations

import pytest

from training.split import (
    SplitError,
    SplitResult,
    assert_no_id_overlap,
    connected_components,
    group_aware_split,
)


def _independent_pairs(count: int) -> list[tuple[str, str]]:
    return [(f"report-{2 * i}", f"report-{2 * i + 1}") for i in range(count)]


def test_connected_components_capture_transitive_report_links() -> None:
    components = connected_components(
        [("A", "B"), ("B", "C"), ("D", "E"), ("F", "G"), ("G", "H")]
    )

    assert {frozenset(component) for component in components} == {
        frozenset({"A", "B", "C"}),
        frozenset({"D", "E"}),
        frozenset({"F", "G", "H"}),
    }


def test_group_aware_split_is_deterministic_for_fixed_seed() -> None:
    pairs = _independent_pairs(24)

    first = group_aware_split(pairs, seed=73)
    second = group_aware_split(pairs, seed=73)

    assert first == second


def test_group_aware_split_has_zero_report_id_overlap() -> None:
    pairs = [
        ("A", "B"),
        ("B", "C"),
        ("D", "E"),
        ("F", "G"),
        ("H", "I"),
        ("J", "K"),
        ("L", "M"),
        ("N", "O"),
    ]

    split = group_aware_split(pairs, ratios=(0.5, 0.25, 0.25), seed=2)

    assert split.train_ids.isdisjoint(split.val_ids)
    assert split.train_ids.isdisjoint(split.test_ids)
    assert split.val_ids.isdisjoint(split.test_ids)
    assert sorted(split.train + split.val + split.test) == list(range(len(pairs)))
    for indices, owned_ids in (
        (split.train, split.train_ids),
        (split.val, split.val_ids),
        (split.test, split.test_ids),
    ):
        assert all(set(pairs[index]) <= owned_ids for index in indices)


def test_independent_components_produce_reasonable_requested_split_sizes() -> None:
    pairs = _independent_pairs(20)

    split = group_aware_split(pairs, ratios=(0.70, 0.15, 0.15), seed=42)

    assert split.sizes() == {"train": 14, "val": 3, "test": 3}
    assert split.id_counts() == {"train": 28, "val": 6, "test": 6}


def test_ratios_are_normalized() -> None:
    pairs = _independent_pairs(10)

    normalized = group_aware_split(pairs, ratios=(7, 2, 1), seed=9)
    fractions = group_aware_split(pairs, ratios=(0.7, 0.2, 0.1), seed=9)

    assert normalized == fractions


def test_empty_input_returns_empty_splits() -> None:
    split = group_aware_split([], seed=5)

    assert split.sizes() == {"train": 0, "val": 0, "test": 0}
    assert split.train_ids == split.val_ids == split.test_ids == set()


@pytest.mark.parametrize("ratios", [(1, 0), (-1, 1, 1), (0, 0, 0)])
def test_invalid_split_ratios_are_rejected(ratios) -> None:
    with pytest.raises(SplitError, match="ratios"):
        group_aware_split([("A", "B")], ratios=ratios)


def test_explicit_overlap_validation_fails_loudly() -> None:
    invalid = SplitResult(
        train=[0],
        val=[1],
        test=[],
        train_ids={"shared", "A"},
        val_ids={"shared", "B"},
        test_ids=set(),
    )

    with pytest.raises(SplitError, match="leakage"):
        assert_no_id_overlap(invalid)
