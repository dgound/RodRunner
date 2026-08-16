"""Movement QC is a policy, and `disabled` means NOT EVALUATED.

Repeatability QC used to be expressed only as a threshold, so "not applicable"
was written as ``--movement_max_nm 1e9``. Every walk then compared favourably
against a bound nothing could exceed, and the resulting ``_filtered.csv`` was
described as having passed movement QC. It had not: nothing was measured.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "pipeline"))

import drift_correct as dc  # noqa: E402

DRIFT = _REPO / "pipeline/drift_correct.py"


class ResolveMovementPolicyTest(unittest.TestCase):
    def test_disabled_has_no_threshold_at_all(self):
        policy, axes, max_nm = dc.resolve_movement_policy(
            dc.MOVEMENT_POLICY_DISABLED, None, None)
        self.assertEqual(policy, dc.MOVEMENT_POLICY_DISABLED)
        self.assertIsNone(max_nm, "a disabled test must carry NO threshold")
        self.assertEqual(axes, "", "and measure no axes")

    def test_a_threshold_with_disabled_is_refused(self):
        """This is the 1e9 trick; it must not be expressible."""
        for value in (1e9, 100.0, 0.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dc.resolve_movement_policy(
                        dc.MOVEMENT_POLICY_DISABLED, None, value)

    def test_each_measuring_policy_fixes_its_own_axes(self):
        self.assertEqual(dc.resolve_movement_policy(
            dc.MOVEMENT_POLICY_XY, None, None)[1], "xy")
        self.assertEqual(dc.resolve_movement_policy(
            dc.MOVEMENT_POLICY_XYZ_LEGACY, None, None)[1], "xyz")

    def test_axes_contradicting_the_policy_are_refused(self):
        with self.assertRaises(ValueError):
            dc.resolve_movement_policy(dc.MOVEMENT_POLICY_XY, "xyz", 100.0)
        with self.assertRaises(ValueError):
            dc.resolve_movement_policy(
                dc.MOVEMENT_POLICY_XYZ_LEGACY, "xy", 100.0)

    def test_agreeing_axes_are_accepted(self):
        self.assertEqual(dc.resolve_movement_policy(
            dc.MOVEMENT_POLICY_XY, "xy", 100.0),
            (dc.MOVEMENT_POLICY_XY, "xy", 100.0))

    def test_no_policy_preserves_the_historical_3d_default(self):
        """An old invocation that passes neither flag must not change meaning."""
        self.assertEqual(dc.resolve_movement_policy(None, None, None),
                         (dc.MOVEMENT_POLICY_XYZ_LEGACY, "xyz",
                          dc.MOVEMENT_MAX_NM_DEFAULT))
        self.assertEqual(dc.resolve_movement_policy(None, "xy", None)[0],
                         dc.MOVEMENT_POLICY_XY)

    def test_the_policies_are_exhaustive_and_mapped(self):
        self.assertEqual(set(dc.MOVEMENT_POLICY_AXES),
                         {dc.MOVEMENT_POLICY_XY, dc.MOVEMENT_POLICY_XYZ_LEGACY,
                          dc.MOVEMENT_POLICY_DISABLED})

    def test_not_evaluated_is_not_a_pass(self):
        self.assertNotEqual(dc.MOVEMENT_QC_NOT_EVALUATED,
                            dc.MOVEMENT_QC_EVALUATED)


class RefusedBeforeAnyWorkTest(unittest.TestCase):
    """The refusal must happen at parse time, not after drift correction."""

    def _run(self, *args):
        return subprocess.run([sys.executable, str(DRIFT), *args],
                              capture_output=True, text=True, cwd=str(_REPO))

    def test_disabled_with_a_threshold_exits_before_reading_any_input(self):
        proc = self._run("--movement_policy", "disabled",
                         "--movement_max_nm", "1e9")
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("disabled test has no threshold", proc.stderr)
        # nothing was read, so no stage output can exist
        self.assertNotIn("Movement filter", proc.stdout)

    def test_contradicting_axes_exit_before_reading_any_input(self):
        proc = self._run("--movement_policy", "xy_repeatability",
                         "--movement_axes", "xyz")
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("contradicts", proc.stderr)


class NoProductionPathDisablesByThresholdTest(unittest.TestCase):
    def test_no_driver_expresses_disabled_as_a_huge_threshold(self):
        offenders = []
        for path in sorted(_REPO.glob("drivers/*.sh")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue          # comments may NAME the retired trick
                if "1e9" in line:
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "a disabled test must be a policy, never a threshold "
                         "nothing can exceed")


if __name__ == "__main__":
    unittest.main()
