#!/usr/bin/env python3
"""Focused synthetic tests for the Yamada/link geometry adapter."""

from pathlib import Path
import sys
import unittest

import numpy as np


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
import topology_yamada as topology  # noqa: E402


YAMADA = Path("/mnt/altnas/work/dimos/Yamada")


def planar_theta(n=20, scale=1000.0):
    angles = np.linspace(0.0, np.pi, n // 2 + 1)
    top = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    bottom = np.column_stack((np.cos(angles), -np.sin(angles), np.zeros_like(angles)))
    middle = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    # The production contract requires shared vertices to be serialized
    # identically, not merely equal within floating-point trigonometric noise.
    bottom[0], bottom[-1] = top[0], top[-1]
    middle[0], middle[-1] = top[0], top[-1]
    return [scale * top, scale * bottom, scale * middle]


def hopf_loops(n=48, scale=1000.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    first = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    second = np.column_stack((1.0 + np.cos(angle), np.zeros_like(angle), np.sin(angle)))
    return scale * first, scale * second


def proper_rotation(seed=17):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q


class ThetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (YAMADA / "yamada.py").is_file():
            raise unittest.SkipTest(f"Yamada checkout unavailable: {YAMADA}")

    def test_planar_theta_has_stable_normalized_polynomial(self):
        strands = planar_theta()
        evaluation = topology.evaluate_theta(strands, YAMADA, seeds=(4, 9))
        self.assertTrue(evaluation.stable)
        self.assertEqual(evaluation.seed_results[0].vertices, 2)
        self.assertEqual(
            evaluation.normalized_terms,
            ((4, -1), (3, -1), (2, -2), (1, -1), (0, -1)))
        self.assertEqual(
            evaluation.normalized_polynomial, "-A^4 - A^3 - 2*A^2 - A - 1")
        self.assertTrue(evaluation.seed_results[0].raw_polynomial)
        self.assertEqual(
            evaluation.seed_results[0],
            topology.compute_yamada(strands, YAMADA / "yamada.py", seed=4))
        self.assertEqual(evaluation.to_json(), evaluation.to_json())

    def test_yamada_is_invariant_under_proper_rotation_and_translation(self):
        strands = planar_theta()
        rotation = proper_rotation()
        transformed = [strand @ rotation.T + np.array([300.0, -70.0, 110.0])
                       for strand in strands]
        original = topology.compute_yamada(strands, YAMADA, seed=2)
        moved = topology.compute_yamada(transformed, YAMADA, seed=2)
        self.assertEqual(original.normalized_terms, moved.normalized_terms)

    def test_theta_requires_exact_shared_endpoints(self):
        strands = planar_theta()
        strands[1] = strands[1].copy()
        strands[1][0, 0] += 1e-9
        with self.assertRaisesRegex(topology.TopologyValidationError, "exact ordered"):
            topology.validate_theta_strands(strands)

    def test_theta_rejects_nonfinite_and_zero_length_segments(self):
        strands = planar_theta()
        strands[0] = strands[0].copy()
        strands[0][3] = strands[0][2]
        with self.assertRaisesRegex(topology.TopologyValidationError, "zero-length"):
            topology.validate_theta_strands(strands)
        strands = planar_theta()
        strands[2] = strands[2].copy()
        strands[2][1, 1] = np.nan
        with self.assertRaisesRegex(topology.TopologyValidationError, "non-finite"):
            topology.validate_theta_strands(strands)

    def test_theta_rejects_strands_overlapping_beyond_fork(self):
        strands = planar_theta()
        strands[1] = strands[1].copy()
        strands[1][1] = strands[0][1]
        with self.assertRaisesRegex(topology.TopologyValidationError, "overlap beyond fork"):
            topology.validate_theta_strands(strands)


class LinkTests(unittest.TestCase):
    def test_hopf_link_and_unlink(self):
        first, second = hopf_loops()
        linked = topology.compute_linking_number(first, second, seed=11)
        unlinked = topology.compute_linking_number(
            first, second + np.array([6000.0, 0.0, 0.0]), seed=11)
        self.assertEqual(linked.absolute_linking_number, 1)
        self.assertTrue(linked.stable)
        self.assertGreater(linked.min_segment_distance_nm, 0.0)
        self.assertEqual(unlinked.linking_number, 0)
        self.assertTrue(unlinked.stable)

    def test_linking_number_is_rotation_and_translation_invariant(self):
        first, second = hopf_loops()
        rotation = proper_rotation(29)
        translation = np.array([120.0, 800.0, -340.0])
        base = topology.compute_linking_number(first, second, seed=5)
        moved = topology.compute_linking_number(
            first @ rotation.T + translation,
            second @ rotation.T + translation,
            seed=5)
        self.assertEqual(base.linking_number, moved.linking_number)
        self.assertEqual(base.projection_stability, moved.projection_stability)

    def test_reversing_one_loop_reverses_orientation(self):
        first, second = hopf_loops()
        forward = topology.compute_linking_number(first, second, seed=3)
        reversed_result = topology.compute_linking_number(first[::-1], second, seed=3)
        self.assertEqual(reversed_result.linking_number, -forward.linking_number)

    def test_explicit_and_implicit_closure_match(self):
        first, second = hopf_loops()
        implicit = topology.compute_linking_number(first, second, seed=13)
        explicit = topology.compute_linking_number(
            np.vstack((first, first[0])), np.vstack((second, second[0])), seed=13)
        self.assertEqual(implicit, explicit)

    def test_touching_loops_are_not_classically_eligible(self):
        first, _ = hopf_loops()
        with self.assertRaisesRegex(topology.TopologyValidationError, "disjoint"):
            topology.compute_linking_number(first, first.copy())

    def test_self_intersecting_loop_is_rejected(self):
        bow_tie = np.array([
            [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0],
        ])
        _, second = hopf_loops(n=16)
        with self.assertRaisesRegex(topology.TopologyValidationError, "self-intersects"):
            topology.compute_linking_number(bow_tie, second)

    def test_serialization_is_stable(self):
        first, second = hopf_loops()
        result = topology.compute_linking_number(first, second, n_projections=24, seed=7)
        self.assertEqual(result.to_json(), result.to_json())
        self.assertIn('"linking_number":', result.to_json())


if __name__ == "__main__":
    unittest.main()
