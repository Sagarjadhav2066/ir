"""Skip-pointer intersection measurements."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class IntersectionMetrics:
    comparisons_without_skips: int
    comparisons_with_skips: int
    saved_comparisons: int
    spacing: int


def intersect_with_metrics(
    left: list[int], right: list[int], spacing: int
) -> tuple[list[int], IntersectionMetrics]:
    if spacing < 1:
        raise ValueError("spacing must be positive")
    i = j = 0
    result: list[int] = []
    comparisons = 0
    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    a = b = 0
    no_skip = 0
    while a < len(left) and b < len(right):
        no_skip += 1
        if left[a] == right[b]:
            a += 1
            b += 1
        elif left[a] < right[b]:
            a += 1
        else:
            b += 1
    metrics = IntersectionMetrics(no_skip, comparisons, no_skip - comparisons, spacing)
    return result, metrics


def recommended_spacings(length: int) -> tuple[int, ...]:
    return (1, max(1, int(math.sqrt(length))), max(1, length // 4))
