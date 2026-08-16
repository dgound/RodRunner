#!/usr/bin/env python3
"""The replication status-code map, centralized. Deliberately dependency-free.

This vocabulary answers ONE question: what did the one-vs-two-spot test conclude
for a given (cell, locus) observation. It is distinct from, and must not be
merged with, either of the other two vocabularies in this pipeline:

* ``center_ownership`` CATEGORIES -- where the fitted centre landed
  (``own_mask`` / ``background`` / ...);
* ``center_ownership`` ATTEMPT_OUTCOMES -- whether a localization attempt yielded
  a usable coordinate row (``accepted`` / ``ownership_rejected`` / ...);
* ``z_support_2p5d.Z_GATE_EVALUABLE_STATUSES`` -- whether the DAPI z interval was
  usable for that cell and TP.

Conflating any two of them has already produced real misclassifications, which is
why each now lives in exactly one place and says in its own name what it is about.

This module holds no numpy/ctypes imports on purpose. It used to live inside
``replication_qc``, whose import compiles a C backend and pulls in ``dat_io``, so
a consumer needing only the three-element evaluable set had to drag all of that
in -- and did, until it broke ``classify_replication`` when run as a subprocess.
"""
from __future__ import annotations

#: Codes 0..9 come from the fitting backend. Ownership-unevaluable is a pipeline
#: decision layered on top, so it starts above the backend's range.
STATUS_OWNERSHIP_UNEVALUABLE = 10

#: The authoritative code -> label map. One definition; both the per-locus rows
#: and the run-level tallies render from it.
STATUS_MAP = {
    2: 'not_suspect',
    3: 'tested_not_confirmed',
    4: 'confirmed',
    5: '2spot_fail:non_converged',
    6: '2spot_fail:non_finite_params',
    7: '2spot_fail:invalid_input',
    8: '1spot_fail:non_converged',
    9: '1spot_fail:invalid_input',
    STATUS_OWNERSHIP_UNEVALUABLE: 'ownership_unevaluable',
}

#: Status strings for which the one-vs-two-spot question was actually ANSWERED.
#: `ownership_unevaluable` is deliberately absent, as are every fit failure:
#: absence of an answer must never become evidence of a single spot.
EVALUABLE_STATUSES = frozenset(
    {'not_suspect', 'tested_not_confirmed', 'confirmed'})

#: The label an ownership-unevaluable observation carries. Named so a consumer
#: matches the constant rather than retyping the string; a second spelling of it
#: (`ownership_invalid`) is what let one consumer miss these rows entirely.
OWNERSHIP_UNEVALUABLE_STATUS = STATUS_MAP[STATUS_OWNERSHIP_UNEVALUABLE]


def status_label(code) -> str:
    """The label for a status code, or an explicit unknown marker."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return f'unknown:{code}'
    return STATUS_MAP.get(code, f'unknown:{code}')


def fit2_converged(fit2_rc, status_code) -> int:
    """Whether the two-component fit converged, defined ONCE.

    Production and the simulator that calibrates its thresholds disagreed here:
    production accepted status 3, 4 and STATUS_OWNERSHIP_UNEVALUABLE, the
    simulator only 3 and 4. The calibration therefore counted convergence
    differently from the thing being calibrated, and every ownership-unevaluable
    locus was a converged fit in production but a non-convergence in simulation.

    STATUS_OWNERSHIP_UNEVALUABLE belongs here: the FIT converged. What is
    unevaluable is the biological one-vs-two-spot question, which is a separate
    concept (see :data:`EVALUABLE_STATUSES`) and must not be folded into this one.
    """
    return int(int(fit2_rc) == 0
               and int(status_code) in (3, 4, STATUS_OWNERSHIP_UNEVALUABLE))
