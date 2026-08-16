import itertools
import math
import unittest

from pipeline.topology_core import (
    ChromosomeLayout,
    LocusPair,
    TopologyInputError,
    analyze_theta_pattern,
    assign_circular,
    assign_open,
    assign_sisters,
    theta_pattern,
)


ORDER = ("A", "B", "C", "D", "E", "F", "G", "H")


class TestChromosomeLayout(unittest.TestCase):
    def test_builds_configurable_ori_to_ter_arms(self):
        layout = ChromosomeLayout(ORDER, ori="A", ter="E")
        self.assertEqual(layout.arms(), (
            ("A", "B", "C", "D", "E"),
            ("A", "H", "G", "F", "E"),
        ))

    def test_invalid_layouts_fail_closed(self):
        bad = [
            (("A", "A", "B"), "A", "B"),
            (("A", "B"), "A", "B"),
            (("A", "B", "C"), "X", "B"),
            (("A", "B", "C"), "A", "A"),
        ]
        for order, ori, ter in bad:
            with self.subTest(order=order, ori=ori, ter=ter):
                with self.assertRaises(TopologyInputError):
                    ChromosomeLayout(order, ori, ter)


class TestThetaPattern(unittest.TestCase):
    def setUp(self):
        self.layout = ChromosomeLayout(ORDER, ori="A", ter="E")

    def test_coherent_two_sided_pattern_is_traceable(self):
        result = analyze_theta_pattern(
            self.layout,
            doubled_loci=("A", "B", "H"),
            evaluable_loci=ORDER,
        )
        self.assertEqual(result.pattern_status, "theta_pattern_supported")
        self.assertEqual(result.trace_status, "theta_traceable")
        self.assertEqual(result.run_forward, ("A", "B"))
        self.assertEqual(result.run_reverse, ("A", "H"))
        self.assertEqual(result.replicated_path, ("B", "A", "H"))
        self.assertEqual(result.unreplicated_path, ("C", "D", "E", "F", "G"))
        self.assertEqual(result.fork_neighbor_forward, "C")
        self.assertEqual(result.fork_neighbor_reverse, "G")
        self.assertTrue(result.is_candidate)
        self.assertTrue(result.is_coherent)
        self.assertTrue(result.is_traceable)

    def test_discordant_double_is_reported_not_ignored(self):
        result = analyze_theta_pattern(
            self.layout,
            doubled_loci=("A", "B", "D", "H"),
            evaluable_loci=ORDER,
        )
        self.assertEqual(result.pattern_status, "theta_pattern_discordant")
        self.assertEqual(result.discordant_doubled, ("D",))
        self.assertEqual(result.trace_status, "theta_geometry_incomplete")

    def test_missing_single_copy_coordinate_is_not_traceable(self):
        result = analyze_theta_pattern(
            self.layout,
            doubled_loci=("A", "B", "H"),
            evaluable_loci=("A", "B", "C", "D", "E", "F", "H"),
        )
        self.assertEqual(result.pattern_status, "theta_pattern_supported")
        self.assertEqual(result.trace_status, "theta_geometry_incomplete")
        self.assertEqual(result.unevaluable_unreplicated, ("G",))

    def test_not_theta_states_are_explicit(self):
        cases = [
            (("B", "H"), "not_theta_ori_not_doubled"),
            (("A", "B"), "not_theta_one_sided"),
            (ORDER, "fully_doubled_not_theta"),
        ]
        for doubled, expected in cases:
            with self.subTest(doubled=doubled):
                result = analyze_theta_pattern(
                    self.layout, doubled_loci=doubled, evaluable_loci=ORDER
                )
                self.assertEqual(result.pattern_status, expected)
                self.assertEqual(result.trace_status, "theta_not_traceable")

    def test_invalid_locus_states_fail_closed(self):
        with self.assertRaisesRegex(TopologyInputError, "outside chromosome_order"):
            analyze_theta_pattern(
                self.layout, doubled_loci=("A", "X"), evaluable_loci=ORDER
            )
        with self.assertRaisesRegex(TopologyInputError, "also be present"):
            analyze_theta_pattern(
                self.layout,
                doubled_loci=("A", "B", "H"),
                evaluable_loci=("A", "B", "C", "D", "E", "F", "G"),
            )
        with self.assertRaisesRegex(TopologyInputError, "duplicate"):
            analyze_theta_pattern(
                self.layout,
                doubled_loci=("A", "A"),
                evaluable_loci=ORDER,
            )

    def test_stable_wrapper_and_empty_observations(self):
        result = theta_pattern(ORDER, "A", "E", (), ())
        self.assertEqual(result.status, "not_theta_ori_not_doubled")
        self.assertEqual(result.doubled_path, ())
        self.assertEqual(result.single_path, ())
        self.assertIsNone(result.fork_adjacent_forward)
        self.assertIsNone(result.fork_adjacent_reverse)


def pair(locus, a, b):
    return LocusPair(locus, (a, 0, 0), (b, 0, 0))


def brute_cost(pairs, states, circular):
    def oriented(item, state):
        return ((item.component_1, item.component_2)
                if state == 0 else (item.component_2, item.component_1))

    cost = 0.0
    edges = [(i, i + 1) for i in range(len(pairs) - 1)]
    if circular:
        edges.append((len(pairs) - 1, 0))
    for left, right in edges:
        la, lb = oriented(pairs[left], states[left])
        ra, rb = oriented(pairs[right], states[right])
        cost += math.dist(la, ra) + math.dist(lb, rb)
    return cost


class TestSisterAssignment(unittest.TestCase):
    def test_open_assignment_matches_exhaustive_global_optimum(self):
        pairs = (
            pair("A", 0, 10),
            pair("B", 2, 9),
            pair("C", 8, 3),
            pair("D", 4, 7),
        )
        result = assign_sisters(pairs)
        ranked = sorted(
            (brute_cost(pairs, (0,) + states, False), (0,) + states)
            for states in itertools.product((0, 1), repeat=len(pairs) - 1)
        )
        self.assertEqual(result.state_path, ranked[0][1])
        self.assertAlmostEqual(result.total_cost, ranked[0][0])
        self.assertAlmostEqual(result.runner_up_cost, ranked[1][0])
        self.assertEqual(result.states, result.state_path)
        self.assertEqual(result.a, result.sister_a)
        self.assertEqual(result.b, result.sister_b)
        self.assertEqual(result.best_cost, result.total_cost)
        self.assertEqual(result.second_cost, result.runner_up_cost)
        self.assertFalse(result.circular)

    def test_circular_assignment_includes_closure_and_is_global(self):
        pairs = (
            LocusPair("A", (0, 0, 0), (10, 0, 0)),
            LocusPair("B", (2, 1, 0), (8, -1, 0)),
            LocusPair("C", (9, 2, 0), (1, -2, 0)),
            LocusPair("D", (3, 3, 0), (7, -3, 0)),
        )
        result = assign_sisters(pairs, circular=True)
        ranked = sorted(
            (brute_cost(pairs, (0,) + states, True), (0,) + states)
            for states in itertools.product((0, 1), repeat=len(pairs) - 1)
        )
        self.assertEqual(result.state_path, ranked[0][1])
        self.assertAlmostEqual(result.total_cost, ranked[0][0])
        self.assertAlmostEqual(result.runner_up_cost, ranked[1][0])
        self.assertTrue(result.circular)

    def test_component_order_does_not_change_output(self):
        forward = (
            LocusPair("A", (0, 0, 0), (10, 0, 0)),
            LocusPair("B", (1, 1, 0), (9, -1, 0)),
            LocusPair("C", (2, 2, 0), (8, -2, 0)),
        )
        reversed_components = tuple(
            LocusPair(item.locus, item.component_2, item.component_1)
            for item in forward
        )
        self.assertEqual(
            assign_circular(forward),
            assign_circular(reversed_components),
        )

    def test_stable_open_api_can_use_xy_cost_without_dropping_z_output(self):
        pairs = (
            LocusPair("A", (0, 0, 100), (10, 0, -100)),
            LocusPair("B", (1, 0, -100), (9, 0, 100)),
        )
        result = assign_open(pairs, dims=(0, 1))
        self.assertEqual(result.dimensions, (0, 1))
        self.assertEqual(result.sister_a[0], (0.0, 0.0, 100.0))
        self.assertEqual(result.sister_a[1], (1.0, 0.0, -100.0))

    def test_invalid_dimensions_fail_closed(self):
        pairs = (pair("A", 0, 1), pair("B", 1, 2))
        for dims in ((), (0, 0), (3,), (True,)):
            with self.subTest(dims=dims):
                with self.assertRaisesRegex(TopologyInputError, "dimensions"):
                    assign_open(pairs, dims=dims)

    def test_ties_are_deterministic(self):
        pairs = (
            LocusPair("A", (0, -1, 0), (0, 1, 0)),
            LocusPair("B", (1, 0, -1), (1, 0, 1)),
        )
        result = assign_sisters(pairs)
        self.assertEqual(result.state_path, (0, 0))
        self.assertEqual(result.total_cost, result.runner_up_cost)
        self.assertEqual(result.cost_margin, 0)

    def test_invalid_coordinates_and_sequences_fail_closed(self):
        with self.assertRaisesRegex(TopologyInputError, "non-finite"):
            LocusPair("A", (0, 0, math.nan), (1, 1, 1))
        with self.assertRaisesRegex(TopologyInputError, "at least 2"):
            assign_sisters((pair("A", 0, 1),))
        with self.assertRaisesRegex(TopologyInputError, "at least 3"):
            assign_sisters((pair("A", 0, 1), pair("B", 1, 2)), circular=True)
        with self.assertRaisesRegex(TopologyInputError, "duplicate loci"):
            assign_sisters((pair("A", 0, 1), pair("A", 1, 2)))


if __name__ == "__main__":
    unittest.main()
