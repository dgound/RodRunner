"""Small, dataset-independent primitives for chromosome topology tracing.

This module deliberately does no file I/O and makes no biological calls from
image intensities.  It accepts an already classified set of doubled/evaluable
loci and unordered pairs of already accepted 3-D component centres.

Coordinates are always ``(x, y, z)``.  Missing, non-finite, duplicated, or
out-of-universe inputs fail closed; this module never fills a missing locus or
projects a coordinate onto a mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite
from typing import Iterable, Sequence


Point3D = tuple[float, float, float]

__all__ = [
    "ChromosomeLayout",
    "LocusPair",
    "Point3D",
    "SisterAssignment",
    "ThetaPattern",
    "TopologyInputError",
    "analyze_theta_pattern",
    "assign_circular",
    "assign_open",
    "assign_sisters",
    "theta_pattern",
]


class TopologyInputError(ValueError):
    """The requested topology cannot be reconstructed from the given input."""


def _unique_labels(values: Iterable[str], name: str) -> tuple[str, ...]:
    labels = tuple(values)
    if not labels:
        raise TopologyInputError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise TopologyInputError(f"{name} must contain non-empty strings")
    if len(labels) != len(set(labels)):
        raise TopologyInputError(f"{name} contains duplicate loci")
    return labels


def _subset_labels(
    values: Iterable[str], name: str, universe: frozenset[str]
) -> frozenset[str]:
    labels = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise TopologyInputError(f"{name} must contain non-empty strings")
    if len(labels) != len(set(labels)):
        raise TopologyInputError(f"{name} contains duplicate loci")
    result = frozenset(labels)
    outside = result - universe
    if outside:
        raise TopologyInputError(
            f"{name} contains loci outside chromosome_order: {sorted(outside)!r}"
        )
    return result


@dataclass(frozen=True)
class ChromosomeLayout:
    """Circular chromosome order and its two Ori-to-Ter arms."""

    order: tuple[str, ...]
    ori: str
    ter: str

    def __post_init__(self) -> None:
        order = _unique_labels(self.order, "chromosome_order")
        if len(order) < 3:
            raise TopologyInputError("chromosome_order must contain at least three loci")
        if self.ori not in order:
            raise TopologyInputError("ori is outside chromosome_order")
        if self.ter not in order:
            raise TopologyInputError("ter is outside chromosome_order")
        if self.ori == self.ter:
            raise TopologyInputError("ori and ter must be different loci")
        object.__setattr__(self, "order", order)

    @property
    def universe(self) -> frozenset[str]:
        return frozenset(self.order)

    def arms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the forward and reverse circular Ori-to-Ter paths."""
        n_loci = len(self.order)
        ori_i = self.order.index(self.ori)
        ter_i = self.order.index(self.ter)

        def walk(step: int) -> tuple[str, ...]:
            result: list[str] = []
            index = ori_i
            while True:
                result.append(self.order[index])
                if index == ter_i:
                    return tuple(result)
                index = (index + step) % n_loci

        return walk(1), walk(-1)


def _anchored_run(arm: Sequence[str], doubled: frozenset[str]) -> tuple[str, ...]:
    result: list[str] = []
    for locus in arm:
        if locus not in doubled:
            break
        result.append(locus)
    return tuple(result)


def _forward_arc(
    order: tuple[str, ...], start_exclusive: str, end_exclusive: str
) -> tuple[str, ...]:
    """Circular forward arc after ``start`` and before ``end``."""
    result: list[str] = []
    index = (order.index(start_exclusive) + 1) % len(order)
    end = order.index(end_exclusive)
    while index != end:
        result.append(order[index])
        index = (index + 1) % len(order)
    return tuple(result)


@dataclass(frozen=True)
class ThetaPattern:
    """Deterministic result of the locus-state theta compatibility check.

    ``replicated_path`` is ordered from the forward-arm front, through Ori, to
    the reverse-arm front.  ``unreplicated_path`` is the complementary forward
    chromosome arc.  They are locus paths, not inferred fork coordinates.
    """

    pattern_status: str
    trace_status: str
    arm_forward: tuple[str, ...]
    arm_reverse: tuple[str, ...]
    run_forward: tuple[str, ...]
    run_reverse: tuple[str, ...]
    replicated_path: tuple[str, ...]
    unreplicated_path: tuple[str, ...]
    discordant_doubled: tuple[str, ...]
    unevaluable_unreplicated: tuple[str, ...]
    fork_neighbor_forward: str | None
    fork_neighbor_reverse: str | None

    @property
    def is_candidate(self) -> bool:
        return self.pattern_status in {
            "theta_pattern_supported",
            "theta_pattern_discordant",
        }

    @property
    def is_coherent(self) -> bool:
        return self.pattern_status == "theta_pattern_supported"

    @property
    def is_traceable(self) -> bool:
        return self.trace_status == "theta_traceable"

    # Stable orchestration names.  The longer field names above state the
    # biology explicitly; these aliases keep callers independent of it.
    @property
    def status(self) -> str:
        return self.pattern_status

    @property
    def doubled_path(self) -> tuple[str, ...]:
        return self.replicated_path

    @property
    def single_path(self) -> tuple[str, ...]:
        return self.unreplicated_path

    @property
    def fork_adjacent_forward(self) -> str | None:
        return self.fork_neighbor_forward

    @property
    def fork_adjacent_reverse(self) -> str | None:
        return self.fork_neighbor_reverse


def analyze_theta_pattern(
    layout: ChromosomeLayout,
    *,
    doubled_loci: Iterable[str],
    evaluable_loci: Iterable[str],
) -> ThetaPattern:
    """Classify whether locus states support a coherent theta graph.

    A candidate requires Ori and at least one further contiguous doubled locus
    on each chromosome arm.  Doubles beyond the first gap are reported as
    discordant.  A trace is ready only when the complementary arc is fully
    evaluable and confidently not doubled.
    """
    doubled = _subset_labels(doubled_loci, "doubled_loci", layout.universe)
    evaluable = _subset_labels(evaluable_loci, "evaluable_loci", layout.universe)
    if not doubled <= evaluable:
        raise TopologyInputError(
            "every doubled locus must also be present in evaluable_loci"
        )

    arm_forward, arm_reverse = layout.arms()
    run_forward = _anchored_run(arm_forward, doubled)
    run_reverse = _anchored_run(arm_reverse, doubled)
    coherent = frozenset(run_forward) | frozenset(run_reverse)
    discordant = doubled - coherent

    replicated_path: tuple[str, ...] = ()
    unreplicated_path: tuple[str, ...] = ()
    forward_neighbor: str | None = None
    reverse_neighbor: str | None = None

    if layout.ori not in doubled:
        pattern_status = "not_theta_ori_not_doubled"
    elif len(run_forward) < 2 or len(run_reverse) < 2:
        pattern_status = "not_theta_one_sided"
    elif coherent == layout.universe:
        pattern_status = "fully_doubled_not_theta"
    else:
        replicated_path = tuple(reversed(run_forward)) + run_reverse[1:]
        forward_front = run_forward[-1]
        reverse_front = run_reverse[-1]
        unreplicated_path = _forward_arc(
            layout.order, forward_front, reverse_front
        )
        if not unreplicated_path:
            # Defensive: the complete chromosome case is handled above.
            raise TopologyInputError(
                "theta candidate has no complementary single-copy arc"
            )
        forward_neighbor = unreplicated_path[0]
        reverse_neighbor = unreplicated_path[-1]
        pattern_status = (
            "theta_pattern_discordant" if discordant
            else "theta_pattern_supported"
        )

    unevaluable = tuple(
        locus for locus in unreplicated_path if locus not in evaluable
    )
    if pattern_status == "theta_pattern_supported" and not unevaluable:
        trace_status = "theta_traceable"
    elif pattern_status in {"theta_pattern_supported", "theta_pattern_discordant"}:
        trace_status = "theta_geometry_incomplete"
    else:
        trace_status = "theta_not_traceable"

    return ThetaPattern(
        pattern_status=pattern_status,
        trace_status=trace_status,
        arm_forward=arm_forward,
        arm_reverse=arm_reverse,
        run_forward=run_forward,
        run_reverse=run_reverse,
        replicated_path=replicated_path,
        unreplicated_path=unreplicated_path,
        discordant_doubled=tuple(
            locus for locus in layout.order if locus in discordant
        ),
        unevaluable_unreplicated=unevaluable,
        fork_neighbor_forward=forward_neighbor,
        fork_neighbor_reverse=reverse_neighbor,
    )


def theta_pattern(
    order: Sequence[str],
    ori: str,
    ter: str,
    doubled: Iterable[str],
    evaluable: Iterable[str],
) -> ThetaPattern:
    """Stable convenience API for :func:`analyze_theta_pattern`."""
    return analyze_theta_pattern(
        ChromosomeLayout(tuple(order), ori=ori, ter=ter),
        doubled_loci=doubled,
        evaluable_loci=evaluable,
    )


def _point(value: Sequence[float], name: str) -> Point3D:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise TopologyInputError(f"{name} must contain exactly three coordinates")
    try:
        point = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise TopologyInputError(f"{name} contains a non-numeric coordinate") from exc
    if not all(isfinite(coordinate) for coordinate in point):
        raise TopologyInputError(f"{name} contains a non-finite coordinate")
    return point  # type: ignore[return-value]


@dataclass(frozen=True)
class LocusPair:
    """Two unordered, accepted component centres for one locus."""

    locus: str
    component_1: Point3D
    component_2: Point3D

    def __post_init__(self) -> None:
        if not isinstance(self.locus, str) or not self.locus.strip():
            raise TopologyInputError("locus must be a non-empty string")
        first = _point(self.component_1, f"{self.locus}.component_1")
        second = _point(self.component_2, f"{self.locus}.component_2")
        # Fitted component numbers have no biological orientation.  Sorting
        # makes output invariant to their input order and anchors sister A.
        ordered = sorted((first, second))
        object.__setattr__(self, "component_1", ordered[0])
        object.__setattr__(self, "component_2", ordered[1])


@dataclass(frozen=True)
class SisterAssignment:
    """Globally minimum-cost ordering of unordered component pairs."""

    loci: tuple[str, ...]
    sister_a: tuple[Point3D, ...]
    sister_b: tuple[Point3D, ...]
    state_path: tuple[int, ...]
    total_cost: float
    runner_up_cost: float | None
    circular: bool
    dimensions: tuple[int, ...]

    @property
    def cost_margin(self) -> float | None:
        if self.runner_up_cost is None:
            return None
        return self.runner_up_cost - self.total_cost

    @property
    def a(self) -> tuple[Point3D, ...]:
        return self.sister_a

    @property
    def b(self) -> tuple[Point3D, ...]:
        return self.sister_b

    @property
    def states(self) -> tuple[int, ...]:
        return self.state_path

    @property
    def best_cost(self) -> float:
        return self.total_cost

    @property
    def second_cost(self) -> float | None:
        return self.runner_up_cost


def _oriented(pair: LocusPair, state: int) -> tuple[Point3D, Point3D]:
    if state == 0:
        return pair.component_1, pair.component_2
    return pair.component_2, pair.component_1


def _transition(left: LocusPair, left_state: int,
                right: LocusPair, right_state: int,
                dimensions: tuple[int, ...]) -> float:
    left_a, left_b = _oriented(left, left_state)
    right_a, right_b = _oriented(right, right_state)
    return (
        dist(tuple(left_a[i] for i in dimensions),
             tuple(right_a[i] for i in dimensions))
        + dist(tuple(left_b[i] for i in dimensions),
               tuple(right_b[i] for i in dimensions))
    )


def _best_two(
    candidates: Iterable[tuple[float, tuple[int, ...]]]
) -> list[tuple[float, tuple[int, ...]]]:
    unique: dict[tuple[int, ...], float] = {}
    for cost, path in candidates:
        previous = unique.get(path)
        if previous is None or cost < previous:
            unique[path] = cost
    return sorted(
        ((cost, path) for path, cost in unique.items()),
        key=lambda item: (item[0], item[1]),
    )[:2]


def assign_sisters(
    pairs: Sequence[LocusPair], *, circular: bool = False,
    dimensions: Sequence[int] = (0, 1, 2),
) -> SisterAssignment:
    """Assign unordered component pairs to two globally consistent strands.

    The objective is the sum of Euclidean A-to-A plus B-to-B links.  For a
    circular assignment it also includes the last-to-first closure.  Sister A
    is anchored to the lexicographically first coordinate at the first locus,
    eliminating the biologically meaningless global A/B label symmetry.
    """
    ordered_pairs = tuple(pairs)
    dims = tuple(dimensions)
    if not dims or len(dims) != len(set(dims)) or any(
            not isinstance(index, int) or isinstance(index, bool)
            or index not in (0, 1, 2) for index in dims):
        raise TopologyInputError(
            "dimensions must be a non-empty unique subset of (0, 1, 2)"
        )
    minimum = 3 if circular else 2
    if len(ordered_pairs) < minimum:
        kind = "circular" if circular else "open"
        raise TopologyInputError(
            f"{kind} sister assignment requires at least {minimum} loci"
        )
    if not all(isinstance(pair, LocusPair) for pair in ordered_pairs):
        raise TopologyInputError("pairs must contain only LocusPair objects")
    loci = tuple(pair.locus for pair in ordered_pairs)
    if len(loci) != len(set(loci)):
        raise TopologyInputError("sister assignment contains duplicate loci")

    # Keep the two globally best histories reaching each state.  Two are enough
    # to report the true runner-up as well as the optimum.
    histories: dict[int, list[tuple[float, tuple[int, ...]]]] = {
        0: [(0.0, (0,))],
        1: [],
    }
    for index in range(1, len(ordered_pairs)):
        next_histories: dict[int, list[tuple[float, tuple[int, ...]]]] = {}
        for state in (0, 1):
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for previous_state in (0, 1):
                for previous_cost, previous_path in histories[previous_state]:
                    candidates.append((
                        previous_cost + _transition(
                            ordered_pairs[index - 1], previous_state,
                            ordered_pairs[index], state, dims,
                        ),
                        previous_path + (state,),
                    ))
            next_histories[state] = _best_two(candidates)
        histories = next_histories

    finalists: list[tuple[float, tuple[int, ...]]] = []
    for final_state in (0, 1):
        for cost, path in histories[final_state]:
            if circular:
                cost += _transition(
                    ordered_pairs[-1], final_state, ordered_pairs[0], 0, dims
                )
            finalists.append((cost, path))
    ranked = _best_two(finalists)
    if not ranked:
        raise TopologyInputError("no valid sister assignment")
    total_cost, state_path = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None

    strands = tuple(
        _oriented(pair, state)
        for pair, state in zip(ordered_pairs, state_path)
    )
    return SisterAssignment(
        loci=loci,
        sister_a=tuple(pair[0] for pair in strands),
        sister_b=tuple(pair[1] for pair in strands),
        state_path=state_path,
        total_cost=total_cost,
        runner_up_cost=runner_up,
        circular=circular,
        dimensions=dims,
    )


def assign_open(
    pairs: Sequence[LocusPair], *, dims: Sequence[int] = (0, 1, 2)
) -> SisterAssignment:
    """Stable API for globally optimal open sister assignment."""
    return assign_sisters(pairs, circular=False, dimensions=dims)


def assign_circular(
    pairs: Sequence[LocusPair], *, dims: Sequence[int] = (0, 1, 2)
) -> SisterAssignment:
    """Stable API for globally optimal closed-loop sister assignment."""
    return assign_sisters(pairs, circular=True, dimensions=dims)
