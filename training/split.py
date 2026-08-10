"""Group-aware train / validation / test splitting.

Random pair-level splitting leaks. If (A,B) trains and (A,C) tests, the model has
already seen A's field set, name, business objects and data source, so the test
score measures memorization of A rather than generalization to unseen reports.

The fix: treat the labeled pairs as an undirected graph over report IDs, take its
connected components, and assign whole components to splits. Every report — and
therefore every pair touching it — lands in exactly one split. Components are the
right unit rather than individual report IDs because a pair spanning two splits
would have to be discarded or would itself leak.

Assignment is deterministic for a fixed seed: components are shuffled with a
seeded RNG, then placed largest-first into whichever split is furthest below its
target share. That greedy step keeps sizes close to the requested ratios even when
one duplicate cluster is much larger than the rest.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")


class SplitError(ValueError):
    """Raised when a split cannot be produced as requested."""


SPLIT_MANIFEST_VERSION = 1


@dataclass
class SplitResult:
    """Row indices per split, plus the report IDs each split owns."""

    train: list[int]
    val: list[int]
    test: list[int]
    train_ids: set[Hashable]
    val_ids: set[Hashable]
    test_ids: set[Hashable]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def id_counts(self) -> dict[str, int]:
        return {"train": len(self.train_ids), "val": len(self.val_ids), "test": len(self.test_ids)}

    @property
    def validation(self) -> list[int]:
        """Readable alias; ``val`` is retained for compact manifest keys."""
        return self.val

    @property
    def validation_ids(self) -> set[Hashable]:
        return self.val_ids


def connected_components(pairs: Sequence[tuple[Hashable, Hashable]]) -> list[set[Hashable]]:
    """Connected components of the undirected report-pair graph (union-find)."""
    parent: dict[Hashable, Hashable] = {}

    def find(x: Hashable) -> Hashable:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: Hashable, b: Hashable) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    groups: dict[Hashable, set[Hashable]] = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    # Sorted by (size desc, then stringified min member) so the component list is
    # itself deterministic before the seeded shuffle.
    return sorted(groups.values(), key=lambda s: (-len(s), sorted(map(str, s))[0]))


def _normalize_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise SplitError("ratios must be (train, val, test).")
    if any(r < 0 for r in ratios):
        raise SplitError("ratios must be non-negative.")
    total = float(sum(ratios))
    if total <= 0:
        raise SplitError("ratios must sum to a positive number.")
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


def group_aware_split(
    pairs: Sequence[tuple[Hashable, Hashable]],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> SplitResult:
    """Split labeled pair rows so no report ID appears in more than one split.

    ``pairs[i]`` is the (report_id_a, report_id_b) of labeled row ``i``; the
    returned lists hold row indices into that same sequence.
    """
    if not pairs:
        return SplitResult([], [], [], set(), set(), set())

    train_r, val_r, test_r = _normalize_ratios(ratios)
    components = connected_components(pairs)

    rng = random.Random(seed)
    rng.shuffle(components)
    # Largest first: placing big clusters while every bucket is still empty is what
    # keeps a single huge component from blowing past its target share.
    components.sort(key=len, reverse=True)

    targets = (train_r, val_r, test_r)
    buckets: list[set[Hashable]] = [set(), set(), set()]
    counts = [0, 0, 0]
    # Optimize pair-row sizes (DataLoader/metric sizes), not report counts. A dense
    # component and a sparse component can own the same number of reports but a
    # very different number of labeled pairs.
    component_of = {
        node: component_index
        for component_index, component in enumerate(components)
        for node in component
    }
    component_row_counts = [0] * len(components)
    for a, _b in pairs:
        component_row_counts[component_of[a]] += 1
    total_rows = len(pairs)

    for component_index, comp in enumerate(components):
        row_weight = component_row_counts[component_index]
        # Place into the split with the largest shortfall against its target share.
        deficits = [
            (targets[k] * total_rows) - counts[k] if targets[k] > 0 else float("-inf")
            for k in range(3)
        ]
        k = max(range(3), key=lambda idx: (deficits[idx], -idx))
        buckets[k] |= comp
        counts[k] += row_weight

    id_to_split = {node: k for k, bucket in enumerate(buckets) for node in bucket}
    rows: list[list[int]] = [[], [], []]
    for i, (a, b) in enumerate(pairs):
        ka, kb = id_to_split[a], id_to_split[b]
        if ka != kb:  # impossible by construction; a loud failure beats silent leakage
            raise SplitError(
                f"Pair row {i} spans splits ({a} in {SPLIT_NAMES[ka]}, {b} in "
                f"{SPLIT_NAMES[kb]}) — connected-component assignment is broken."
            )
        rows[ka].append(i)

    result = SplitResult(
        train=rows[0], val=rows[1], test=rows[2],
        train_ids=buckets[0], val_ids=buckets[1], test_ids=buckets[2],
    )
    assert_no_id_overlap(result)
    return result


def assert_no_id_overlap(split: SplitResult) -> None:
    """Hard guarantee: zero report-ID overlap across train / val / test."""
    checks: Iterable[tuple[str, str, set, set]] = (
        ("train", "val", split.train_ids, split.val_ids),
        ("train", "test", split.train_ids, split.test_ids),
        ("val", "test", split.val_ids, split.test_ids),
    )
    for name_a, name_b, ids_a, ids_b in checks:
        overlap = ids_a & ids_b
        if overlap:
            raise SplitError(
                f"Report ID leakage between {name_a} and {name_b}: "
                f"{sorted(map(str, overlap))[:10]}"
            )


def default_manifest_path(model_path: str | Path) -> Path:
    """Companion path that keeps the artifact suffix visible."""
    model = Path(model_path)
    return model.with_suffix(model.suffix + ".split.json")


def build_split_manifest(
    split: SplitResult,
    pairs: Sequence[Any],
    *,
    dataset_sha256: str,
    feature_schema_version: str,
    feature_names: Sequence[str],
    seed: int,
    ratios: Sequence[float],
    unresolved_policy: str,
) -> dict[str, Any]:
    """Serialize exact held-out membership without embedding it in the model file."""

    def entries(indices: Sequence[int]) -> list[dict[str, Any]]:
        return [
            {
                "loaded_index": int(index),
                "csv_row_number": int(pairs[index].row_number),
                "report_uid_a": str(pairs[index].report_uid_a),
                "report_uid_b": str(pairs[index].report_uid_b),
            }
            for index in indices
        ]

    return {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "split_strategy": "connected_components_of_all_report_pair_ids",
        "dataset_sha256": dataset_sha256,
        "feature_schema_version": feature_schema_version,
        "feature_names": list(feature_names),
        "random_seed": int(seed),
        "split_ratios": [float(value) for value in ratios],
        "unresolved_label_policy": unresolved_policy,
        "splits": {
            "train": entries(split.train),
            "validation": entries(split.val),
            "test": entries(split.test),
        },
    }


def write_split_manifest(path: str | Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    """Write a deterministic JSON manifest and return its path and SHA-256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.write_bytes(encoded)
    if os.name == "posix":
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
    return destination, hashlib.sha256(encoded).hexdigest()


def load_split_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a manifest and return it with the SHA-256 of its exact bytes."""
    source = Path(path)
    if not source.is_file():
        raise SplitError(f"Split manifest not found: {source}")
    try:
        encoded = source.read_bytes()
        manifest = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitError(f"Could not read split manifest {source}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SplitError(f"Split manifest {source} must contain a JSON object.")
    if manifest.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        raise SplitError(
            f"Unsupported split manifest version {manifest.get('manifest_version')!r}; "
            f"expected {SPLIT_MANIFEST_VERSION}."
        )
    return manifest, hashlib.sha256(encoded).hexdigest()
