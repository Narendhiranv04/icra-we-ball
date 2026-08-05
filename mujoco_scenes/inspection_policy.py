"""Interchangeable region-order policies for inspection ablations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


class InspectionPolicy(Protocol):
    def choose(
        self, available_regions: Sequence[str], inspected_regions: Sequence[str]
    ) -> str:
        """Choose one currently available region."""


@dataclass(frozen=True)
class FixedInspectionPolicy:
    order: tuple[str, ...]

    def choose(
        self, available_regions: Sequence[str], inspected_regions: Sequence[str]
    ) -> str:
        available = set(available_regions)
        for region in self.order:
            if region in available:
                return region
        raise ValueError("fixed inspection order contains no available region")


@dataclass
class RandomInspectionPolicy:
    seed: int = 0
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def choose(
        self, available_regions: Sequence[str], inspected_regions: Sequence[str]
    ) -> str:
        if not available_regions:
            raise ValueError("no inspection regions are available")
        return self._random.choice(sorted(available_regions))


@dataclass(frozen=True)
class RankedInspectionPolicy:
    """Adapter for a foundation-model region ranker."""

    rank: Callable[[tuple[str, ...], tuple[str, ...]], Sequence[str]]

    def choose(
        self, available_regions: Sequence[str], inspected_regions: Sequence[str]
    ) -> str:
        available = tuple(dict.fromkeys(available_regions))
        ranked = tuple(self.rank(available, tuple(inspected_regions)))
        if len(ranked) != len(set(ranked)):
            raise ValueError("region ranking contains duplicates")
        unknown = set(ranked) - set(available)
        if unknown:
            raise ValueError(f"region ranking contains unknown regions: {unknown}")
        if set(ranked) != set(available):
            raise ValueError("region ranking must include every available region")
        if not ranked:
            raise ValueError("no inspection regions are available")
        return ranked[0]
