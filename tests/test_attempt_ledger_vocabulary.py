"""One authoritative reading of the attempt ledger, and blank is not success.

The same two columns used to be re-interpreted inline in five places. Three of
them read a BLANK ``fit_ret_code`` as zero, i.e. as a converged fit. That is the
dangerous direction: an attempt that may never have been fitted enters the
accepted set, where a locus counted as "tested, single spot" becomes negative
biological evidence for replication it was never tested for.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "pipeline"))

import center_ownership as own  # noqa: E402


def _row(ret="0", written="1", **extra):
    return {own.LEDGER_RET_CODE: ret, own.LEDGER_WRITTEN: written, **extra}


class AttemptOutcomeTest(unittest.TestCase):
    def test_the_three_real_outcomes(self):
        self.assertEqual(own.attempt_outcome(_row("0", "1")),
                         own.ATTEMPT_ACCEPTED)
        self.assertEqual(own.attempt_outcome(_row("0", "0")),
                         own.ATTEMPT_OWNERSHIP_REJECTED)
        self.assertEqual(own.attempt_outcome(_row("3", "0")),
                         own.ATTEMPT_FIT_FAILED)

    def test_a_failed_fit_is_failed_whatever_written_says(self):
        """A nonzero code means no centre exists, so ownership never ran."""
        for written in ("0", "1", ""):
            with self.subTest(written=written):
                self.assertEqual(own.attempt_outcome(_row("2", written)),
                                 own.ATTEMPT_FIT_FAILED)

    def test_blank_ret_code_is_refused_not_treated_as_zero(self):
        for blank in ("", "   ", "NA", "nan", "None", None):
            with self.subTest(blank=blank):
                with self.assertRaises(own.LedgerRowError):
                    own.attempt_outcome(_row(blank, "1"))

    def test_blank_written_on_a_converged_fit_is_refused(self):
        with self.assertRaises(own.LedgerRowError):
            own.attempt_outcome(_row("0", ""))

    def test_unparseable_ret_code_is_refused(self):
        with self.assertRaises(own.LedgerRowError):
            own.attempt_outcome(_row("failed", "1"))

    def test_a_missing_column_is_refused_rather_than_defaulted(self):
        with self.assertRaises(own.LedgerRowError):
            own.attempt_outcome({own.LEDGER_WRITTEN: "1"})

    def test_historical_inputs_get_indeterminate_never_accepted(self):
        """The lenient boundary must not manufacture an accepted attempt."""
        for row in (_row("", "1"), _row("0", ""), _row("bogus", "1")):
            outcome = own.attempt_outcome(row, strict=False)
            with self.subTest(row=row):
                self.assertEqual(outcome, own.ATTEMPT_INDETERMINATE)
                self.assertNotEqual(outcome, own.ATTEMPT_ACCEPTED)

    def test_outcomes_are_exhaustive_and_distinct(self):
        self.assertEqual(len(set(own.ATTEMPT_OUTCOMES)),
                         len(own.ATTEMPT_OUTCOMES))
        # The attempt vocabulary must not collide with the ownership CATEGORY
        # vocabulary: they answer different questions (did the attempt yield a
        # usable row / where did the centre land) and were conflated before.
        self.assertEqual(set(own.ATTEMPT_OUTCOMES) & set(own.CATEGORIES), set())

    def test_ret_code_accepts_the_float_form_upstream_writes(self):
        self.assertEqual(own.attempt_outcome(_row("0.0", "1")),
                         own.ATTEMPT_ACCEPTED)
        self.assertEqual(own.attempt_outcome(_row("3.0", "1")),
                         own.ATTEMPT_FIT_FAILED)


class ConsumersUseTheSharedReaderTest(unittest.TestCase):
    """No consumer may keep its own reading of these two columns."""

    FILES = ("pipeline/classify_replication.py",
             "pipeline/trace_salvage.py",
             ".campaigns/replication_publication_v1/campaignlib.py")

    def test_no_consumer_treats_a_blank_ret_code_as_zero(self):
        for name in self.FILES:
            text = (_REPO / name).read_text()
            with self.subTest(name):
                # the three exact idioms that meant "blank is success"
                self.assertNotIn('"fit_ret_code", "")).strip() in ("", "0")',
                                 text)
                self.assertNotIn(".strip() or 0)", text)
                self.assertNotIn("row.get('fit_ret_code', '0')", text)

    def test_every_consumer_calls_the_shared_reader(self):
        for name in self.FILES:
            with self.subTest(name):
                self.assertIn("attempt_outcome", (_REPO / name).read_text())



class OneRowEmissionPredicateTest(unittest.TestCase):
    """`localization_written` and actual row emission must never disagree."""

    def test_allow_emits_regardless_of_ownership(self):
        for passed in (True, False):
            self.assertTrue(own.emits_coordinate_row(
                own.POLICY_ALLOW, True, passed))

    def test_strict_and_constrained_agree_with_each_other(self):
        """The old code had these two diverge, which is the whole bug."""
        for passed in (True, False):
            with self.subTest(passed=passed):
                self.assertEqual(
                    own.emits_coordinate_row(own.POLICY_STRICT, True, passed),
                    own.emits_coordinate_row(
                        own.POLICY_CONSTRAINED_EXPERIMENTAL, True, passed))

    def test_constrained_experimental_does_not_claim_an_unwritten_row(self):
        """`policy != 'strict' or passed` used to make this True."""
        self.assertFalse(own.emits_coordinate_row(
            own.POLICY_CONSTRAINED_EXPERIMENTAL, True, False))

    def test_a_failed_fit_never_emits_under_any_policy(self):
        for policy in own.POLICIES:
            with self.subTest(policy):
                self.assertFalse(own.emits_coordinate_row(policy, False, True))

    def test_the_localizer_uses_the_shared_predicate_in_both_places(self):
        text = (_REPO / "pipeline/localize_probe0.py").read_text()
        self.assertEqual(text.count("emits_coordinate_row"), 2)
        # neither of the two retired spellings may remain
        self.assertNotIn("center_ownership != 'strict' or passed", text)
        self.assertNotIn("!= _own.POLICY_ALLOW and not passed", text)


class OneOwnershipCensorSpellingTest(unittest.TestCase):
    def test_trace_salvage_uses_the_canonical_status(self):
        sys.path.insert(0, str(_REPO / "pipeline"))
        import replication_status as rstatus
        import trace_salvage as ts
        self.assertEqual(ts.OWNERSHIP_CENSORED_STATUS,
                         rstatus.OWNERSHIP_UNEVALUABLE_STATUS)

    def test_the_retired_second_spelling_is_gone(self):
        text = (_REPO / "pipeline/trace_salvage.py").read_text()
        code = "\n".join(line for line in text.splitlines()
                         if not line.lstrip().startswith("#"))
        self.assertNotIn('"ownership_invalid"', code)

    def test_the_canonical_status_is_not_evaluable(self):
        sys.path.insert(0, str(_REPO / "pipeline"))
        import replication_status as rstatus
        self.assertNotIn(rstatus.OWNERSHIP_UNEVALUABLE_STATUS,
                         rstatus.EVALUABLE_STATUSES)

if __name__ == "__main__":
    unittest.main()
