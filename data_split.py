"""Porosity-stratified train, validation, and test splitting."""

from functools import lru_cache
from itertools import combinations

import numpy as np


def porosity_stratified_split(
    porosities: np.ndarray,
    fractions: tuple[float, float, float],
    requested_bins: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split sample indices while approximately preserving porosity-bin counts."""
    values = np.asarray(porosities, dtype=np.float64)
    ratios = np.asarray(fractions, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("porosities must be a one-dimensional array of finite values.")
    if values.size < 3:
        raise ValueError("At least three matched samples are required for train/validation/test.")
    if (
        ratios.shape != (3,)
        or not np.isfinite(ratios).all()
        or np.any(ratios <= 0)
        or not np.isclose(ratios.sum(), 1.0)
    ):
        raise ValueError("Split fractions must be three positive values summing to 1.")
    if isinstance(requested_bins, bool) or not isinstance(requested_bins, int) or requested_bins < 1:
        raise ValueError("requested_bins must be a positive integer.")

    rng = np.random.default_rng(seed)
    sample_count = values.size
    raw_totals = sample_count * ratios
    target_totals = np.floor(raw_totals).astype(int)
    total_remainder = sample_count - int(target_totals.sum())
    tie_order = rng.permutation(3)
    total_order = sorted(
        tie_order,
        key=lambda index: raw_totals[index] - target_totals[index],
        reverse=True,
    )
    for split_index in total_order[:total_remainder]:
        target_totals[split_index] += 1
    for split_index in range(3):
        if target_totals[split_index] == 0:
            donor = int(np.argmax(target_totals))
            target_totals[donor] -= 1
            target_totals[split_index] += 1

    if sample_count < 5:
        shuffled = rng.permutation(sample_count)
        first_end = int(target_totals[0])
        second_end = first_end + int(target_totals[1])
        return (
            shuffled[:first_end],
            shuffled[first_end:second_end],
            shuffled[second_end:],
            1,
        )

    minimum_bin_size = int(np.ceil(1.0 / min(ratios)))
    bin_count = min(requested_bins, max(1, sample_count // minimum_bin_size))
    quantile_edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1)))
    bin_ids = np.digitize(values, quantile_edges[1:-1], right=True)
    unique_bins = np.unique(bin_ids)

    group_indices: list[np.ndarray] = []
    base_allocations: list[np.ndarray] = []
    remainders: list[np.ndarray] = []
    row_remainders: list[int] = []
    options: list[list[tuple[int, ...]]] = []
    for bin_id in unique_bins:
        indices = np.flatnonzero(bin_ids == bin_id)
        raw_allocation = indices.size * ratios
        base = np.floor(raw_allocation).astype(int)
        row_remainder = int(indices.size - base.sum())
        group_indices.append(indices)
        base_allocations.append(base)
        remainders.append(raw_allocation - base)
        row_remainders.append(row_remainder)
        bin_options = list(combinations(range(3), row_remainder))
        rng.shuffle(bin_options)
        options.append(bin_options)

    deficits = target_totals - np.sum(base_allocations, axis=0)
    if np.any(deficits < 0):
        raise RuntimeError("Could not reconcile per-bin split sizes with requested split fractions.")

    @lru_cache(maxsize=None)
    def allocate_extras(
        bin_index: int, remaining: tuple[int, int, int]
    ) -> tuple[tuple[int, ...], ...] | None:
        if bin_index == len(group_indices):
            return () if remaining == (0, 0, 0) else None

        best_allocation = None
        best_score = float("-inf")
        for option in options[bin_index]:
            if any(remaining[split] == 0 for split in option):
                continue
            next_remaining = list(remaining)
            for split in option:
                next_remaining[split] -= 1
            tail = allocate_extras(bin_index + 1, tuple(next_remaining))
            if tail is None:
                continue
            score = sum(remainders[bin_index][split] for split in option)
            if score > best_score:
                best_score = score
                best_allocation = (option,) + tail
        return best_allocation

    extra_allocations = allocate_extras(0, tuple(int(value) for value in deficits))
    if extra_allocations is None:
        raise RuntimeError("Could not distribute bin remainders across the requested splits.")

    split_indices: list[list[np.ndarray]] = [[], [], []]
    for indices, base, extra in zip(group_indices, base_allocations, extra_allocations):
        counts = base.copy()
        for split_index in extra:
            counts[split_index] += 1
        shuffled_indices = rng.permutation(indices)
        offset = 0
        for split_index, count in enumerate(counts):
            split_indices[split_index].append(shuffled_indices[offset : offset + count])
            offset += count

    return (
        rng.permutation(np.concatenate(split_indices[0])),
        rng.permutation(np.concatenate(split_indices[1])),
        rng.permutation(np.concatenate(split_indices[2])),
        len(unique_bins),
    )
