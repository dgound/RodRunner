"""Synthetic tests for the C topology engine through its ctypes binding."""
import itertools
import math
from pathlib import Path
import sys
import unittest

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
from detect_topology import (  # noqa: E402
    ENGINE_SO, Engine, Layout, TopologyDetectionError, TopologyValidationError,
    CrossingsExceeded, _format_laurent,
)


ORDER = ("A", "B", "C", "D", "E", "F", "G", "H")
LAYOUT = Layout(ORDER, "A", "E", {tp: i for i, tp in enumerate(ORDER)})


def mask(loci):
    return sum(1 << LAYOUT.index[tp] for tp in loci)


def planar_theta(n=20, scale=1000.0):
    angles = [math.pi * i / (n // 2) for i in range(n // 2 + 1)]
    top = [(scale * math.cos(a), scale * math.sin(a), 0.0) for a in angles]
    bottom = [(scale * math.cos(a), -scale * math.sin(a), 0.0) for a in angles]
    middle = [(scale, 0.0, 0.0), (0.0, 0.0, 0.0), (-scale, 0.0, 0.0)]
    bottom[0], bottom[-1] = top[0], top[-1]
    middle[0], middle[-1] = top[0], top[-1]
    return [top, bottom, middle]


def hopf_loops(n=48, scale=1000.0):
    angle = [2.0 * math.pi * i / n for i in range(n)]
    first = [(scale * math.cos(a), scale * math.sin(a), 0.0) for a in angle]
    second = [(scale * (1.0 + math.cos(a)), 0.0, scale * math.sin(a)) for a in angle]
    return first, second


def rotate(points, axis_angles, shift=(0.0, 0.0, 0.0)):
    ax, ay, az = axis_angles
    out = []
    for x, y, z in points:
        y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
        x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
        x, y = x * math.cos(az) - y * math.sin(az), x * math.sin(az) + y * math.cos(az)
        out.append((x + shift[0], y + shift[1], z + shift[2]))
    return out


def brute_cost(pairs, states, circular):
    def oriented(item, state):
        first, second = sorted(item)
        return (first, second) if state == 0 else (second, first)

    cost = 0.0
    edges = [(i, i + 1) for i in range(len(pairs) - 1)]
    if circular:
        edges.append((len(pairs) - 1, 0))
    for left, right in edges:
        la, lb = oriented(pairs[left], states[left])
        ra, rb = oriented(pairs[right], states[right])
        cost += math.dist(la, ra) + math.dist(lb, rb)
    return cost


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENGINE_SO.is_file():
            raise unittest.SkipTest(f"{ENGINE_SO} not built")
        cls.engine = Engine()


class ThetaPatternTest(EngineTest):
    def test_coherent_two_sided_pattern_is_traceable(self):
        result = self.engine.theta_pattern(LAYOUT, mask("ABH"), mask(ORDER))
        self.assertEqual(result.status, "theta_pattern_supported")
        self.assertEqual(result.trace_status, "theta_traceable")
        self.assertEqual(result.replicated_path, ("B", "A", "H"))
        self.assertEqual(result.unreplicated_path, ("C", "D", "E", "F", "G"))

    def test_discordant_double_is_reported_not_ignored(self):
        result = self.engine.theta_pattern(LAYOUT, mask("ABDH"), mask(ORDER))
        self.assertEqual(result.status, "theta_pattern_discordant")
        self.assertEqual(result.trace_status, "theta_geometry_incomplete")
        self.assertEqual(result.replicated_path, ("B", "A", "H"))

    def test_missing_single_copy_locus_is_not_traceable(self):
        result = self.engine.theta_pattern(LAYOUT, mask("ABH"), mask("ABCDEFH"))
        self.assertEqual(result.status, "theta_pattern_supported")
        self.assertEqual(result.trace_status, "theta_geometry_incomplete")

    def test_not_theta_states_are_explicit(self):
        cases = [
            ("BH", "not_theta_ori_not_doubled"),
            ("AB", "not_theta_one_sided"),
            (ORDER, "fully_doubled_not_theta"),
            ("", "not_theta_ori_not_doubled"),
        ]
        for doubled, expected in cases:
            with self.subTest(doubled=doubled):
                result = self.engine.theta_pattern(LAYOUT, mask(doubled), mask(ORDER))
                self.assertEqual(result.status, expected)
                self.assertEqual(result.trace_status, "theta_not_traceable")
                self.assertEqual(result.replicated_path, ())
                self.assertEqual(result.unreplicated_path, ())

    def test_doubled_must_be_evaluable(self):
        with self.assertRaises(TopologyDetectionError):
            self.engine.theta_pattern(LAYOUT, mask("ABH"), mask("ABCDEFG"))


def pair(a, b):
    return ((a, 0.0, 0.0), (b, 0.0, 0.0))


class SisterAssignmentTest(EngineTest):
    def test_open_assignment_matches_exhaustive_global_optimum(self):
        pairs = (pair(0, 10), pair(2, 9), pair(8, 3), pair(4, 7))
        result = self.engine.assign("ABCD", pairs, circular=False)
        ranked = sorted(
            (brute_cost(pairs, (0,) + states, False), (0,) + states)
            for states in itertools.product((0, 1), repeat=len(pairs) - 1))
        self.assertEqual(result.state_path, ranked[0][1])
        self.assertAlmostEqual(result.total_cost, ranked[0][0])
        self.assertAlmostEqual(result.runner_up_cost, ranked[1][0])
        self.assertEqual(result.loci, ("A", "B", "C", "D"))

    def test_circular_assignment_includes_closure_and_is_global(self):
        pairs = (
            ((0, 0, 0), (10, 0, 0)), ((2, 1, 0), (8, -1, 0)),
            ((9, 2, 0), (1, -2, 0)), ((3, 3, 0), (7, -3, 0)),
        )
        result = self.engine.assign("ABCD", pairs, circular=True)
        ranked = sorted(
            (brute_cost(pairs, (0,) + states, True), (0,) + states)
            for states in itertools.product((0, 1), repeat=len(pairs) - 1))
        self.assertEqual(result.state_path, ranked[0][1])
        self.assertAlmostEqual(result.total_cost, ranked[0][0])
        self.assertAlmostEqual(result.runner_up_cost, ranked[1][0])

    def test_component_order_does_not_change_output(self):
        forward = (((0, 0, 0), (10, 0, 0)), ((1, 1, 0), (9, -1, 0)), ((2, 2, 0), (8, -2, 0)))
        swapped = tuple((b, a) for a, b in forward)
        self.assertEqual(self.engine.assign("ABC", forward, circular=True),
                         self.engine.assign("ABC", swapped, circular=True))

    def test_xy_cost_keeps_z_in_output(self):
        pairs = (((0, 0, 100), (10, 0, -100)), ((1, 0, -100), (9, 0, 100)))
        result = self.engine.assign("AB", pairs, circular=False, dims=(0, 1))
        self.assertEqual(result.sister_a[0], (0.0, 0.0, 100.0))
        self.assertEqual(result.sister_a[1], (1.0, 0.0, -100.0))

    def test_ties_are_deterministic(self):
        pairs = (((0, -1, 0), (0, 1, 0)), ((1, 0, -1), (1, 0, 1)))
        result = self.engine.assign("AB", pairs, circular=False)
        self.assertEqual(result.state_path, (0, 0))
        self.assertEqual(result.total_cost, result.runner_up_cost)
        self.assertEqual(result.cost_margin, 0)

    def test_too_few_loci_fail_closed(self):
        with self.assertRaisesRegex(TopologyDetectionError, "at least 2"):
            self.engine.assign("A", (pair(0, 1),), circular=False)
        with self.assertRaisesRegex(TopologyDetectionError, "at least 3"):
            self.engine.assign("AB", (pair(0, 1), pair(1, 2)), circular=True)


class ThetaYamadaTest(EngineTest):
    def test_planar_theta_polynomial_is_stable_across_seeds(self):
        strands = planar_theta()
        self.engine.validate_theta(strands)
        results = [self.engine.yamada(strands, tries=20, seed=seed, max_crossings=14)
                   for seed in (4, 9)]
        self.assertEqual(results[0].normalized_terms, results[1].normalized_terms)
        self.assertEqual(results[0].normalized_terms,
                         ((4, -1), (3, -1), (2, -2), (1, -1), (0, -1)))
        self.assertEqual(_format_laurent(results[0].normalized_terms),
                         "-A^4 - A^3 - 2*A^2 - A - 1")
        self.assertEqual(results[0].vertices, 2)
        self.assertEqual(results[0].crossings, 0)

    def test_yamada_is_invariant_under_rotation_and_translation(self):
        strands = planar_theta()
        moved = [rotate(s, (0.4, -1.1, 2.3), (300.0, -70.0, 110.0)) for s in strands]
        original = self.engine.yamada(strands, tries=20, seed=2, max_crossings=14)
        rotated = self.engine.yamada(moved, tries=20, seed=2, max_crossings=14)
        self.assertEqual(original.normalized_terms, rotated.normalized_terms)

    def test_crossing_cap_is_reported(self):
        n = 20
        t = [i / (n - 1) for i in range(n)]
        s1 = [(x, 0.0, 0.0) for x in t]
        s2 = [(x, 0.3 * math.sin(3 * math.pi * x), 0.2 * math.cos(2 * math.pi * x)) for x in t]
        s3 = [(x, -0.3 * math.sin(2 * math.pi * x), 0.2 * math.sin(3 * math.pi * x)) for x in t]
        for s in (s2, s3):
            s[0], s[-1] = s1[0], s1[-1]
        self.engine.validate_theta([s1, s2, s3])
        with self.assertRaises(CrossingsExceeded):
            self.engine.yamada([s1, s2, s3], tries=60, seed=42, max_crossings=1)

    def test_perturbation_agreement_for_planar_theta(self):
        # Observed loci (no forks): single-copy arc and two well-separated sisters.
        singles = [(1000.0 * math.cos(a), 1000.0 * math.sin(a), 0.0)
                   for a in [math.pi * i / 10 for i in range(1, 10)]]
        sister_a = [(1000.0 * math.cos(a), -1000.0 * math.sin(a), 0.0)
                    for a in [math.pi * i / 10 for i in range(1, 10)]]
        sister_b = [(600.0 * math.cos(a), -600.0 * math.sin(a), 0.0)
                    for a in [math.pi * i / 10 for i in range(1, 10)]]
        nominal = ((4, -1), (3, -1), (2, -2), (1, -1), (0, -1))
        agree, invalid, unknown = self.engine.theta_perturbation(
            singles, sister_a, sister_b, boundary_midpoint=True, tries=20, seed=3,
            max_crossings=16, draws=40, sigma_xy=20.0, sigma_z=80.0, nominal_terms=nominal)
        self.assertEqual((agree, invalid, unknown), (40, 0, 0))
        agree, invalid, unknown = self.engine.theta_perturbation(
            singles, sister_a, sister_b, boundary_midpoint=True, tries=20, seed=3,
            max_crossings=16, draws=40, sigma_xy=20.0, sigma_z=80.0,
            nominal_terms=((0, 1),))
        self.assertEqual(agree, 0)

    def test_theta_requires_exact_shared_endpoints(self):
        strands = planar_theta()
        x, y, z = strands[1][0]
        strands[1][0] = (x + 1e-9, y, z)
        with self.assertRaisesRegex(TopologyValidationError, "exact ordered"):
            self.engine.validate_theta(strands)

    def test_theta_rejects_zero_length_segments(self):
        strands = planar_theta()
        strands[0][3] = strands[0][2]
        with self.assertRaisesRegex(TopologyValidationError, "zero-length"):
            self.engine.validate_theta(strands)

    def test_theta_rejects_strands_overlapping_beyond_fork(self):
        strands = planar_theta()
        strands[1][1] = strands[0][1]
        with self.assertRaisesRegex(TopologyValidationError, "overlap beyond fork"):
            self.engine.validate_theta(strands)


class LinkTest(EngineTest):
    def test_hopf_link_and_unlink(self):
        first, second = hopf_loops()
        linked = self.engine.linking(first, second, n_projections=96)
        far = [(x + 6000.0, y, z) for x, y, z in second]
        unlinked = self.engine.linking(first, far, n_projections=96)
        self.assertEqual(abs(linked.linking_number), 1)
        self.assertTrue(linked.stable)
        self.assertGreater(linked.min_distance_nm, 0.0)
        self.assertEqual(unlinked.linking_number, 0)
        self.assertTrue(unlinked.stable)

    def test_linking_number_is_rotation_and_translation_invariant(self):
        first, second = hopf_loops()
        base = self.engine.linking(first, second, n_projections=96)
        moved = self.engine.linking(
            rotate(first, (1.0, 0.3, -0.7), (120.0, 800.0, -340.0)),
            rotate(second, (1.0, 0.3, -0.7), (120.0, 800.0, -340.0)),
            n_projections=96)
        self.assertEqual(base.linking_number, moved.linking_number)

    def test_reversing_one_loop_reverses_orientation(self):
        first, second = hopf_loops()
        forward = self.engine.linking(first, second, n_projections=96)
        backward = self.engine.linking(first[::-1], second, n_projections=96)
        self.assertEqual(backward.linking_number, -forward.linking_number)

    def test_explicit_and_implicit_closure_match(self):
        first, second = hopf_loops()
        implicit = self.engine.linking(first, second, n_projections=96)
        explicit = self.engine.linking(first + [first[0]], second + [second[0]],
                                       n_projections=96)
        self.assertEqual(implicit, explicit)

    def test_touching_loops_are_not_classically_eligible(self):
        first, _ = hopf_loops()
        with self.assertRaisesRegex(TopologyValidationError, "disjoint"):
            self.engine.linking(first, list(first), n_projections=96)

    def test_perturbation_separates_robust_and_fragile_calls(self):
        first, second = hopf_loops()
        agree, invalid = self.engine.link_perturbation(
            first, second, n_projections=96, draws=100, sigma_xy=20.0, sigma_z=80.0,
            seed=7, nominal=self.engine.linking(first, second, n_projections=96).linking_number)
        self.assertEqual((agree, invalid), (100, 0))
        # Second loop shifted so it pierces the first loop's plane 10 nm outside
        # the first loop's curve: nominally unlinked, but within localization noise.
        grazing = [(x + 2010.0, y, z) for x, y, z in
                   [(1000.0 * math.cos(a), 0.0, 1000.0 * math.sin(a))
                    for a in [2.0 * math.pi * i / 48 for i in range(48)]]]
        nominal = self.engine.linking(first, grazing, n_projections=96).linking_number
        agree, invalid = self.engine.link_perturbation(
            first, grazing, n_projections=96, draws=100, sigma_xy=20.0, sigma_z=80.0,
            seed=7, nominal=nominal)
        self.assertLess(agree, 95)

    def test_self_intersecting_loop_is_rejected(self):
        bow_tie = [(-1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0)]
        _, second = hopf_loops(n=16)
        with self.assertRaisesRegex(TopologyValidationError, "self-intersects"):
            self.engine.linking(bow_tie, second, n_projections=96)


if __name__ == "__main__":
    unittest.main()
