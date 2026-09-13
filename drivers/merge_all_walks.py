#!/usr/bin/env python
"""One unified CSV: all valid walks from every Location, globally numbered.

Valid = localized at every timepoint (the full-coverage set). A walk with a
missing timepoint is excluded, since the gap breaks any per-walk trace or
distance analysis.

Produces two output files with identical columns:

  1. **Full file** — every full-coverage walk, with trace-salvage columns.
  2. **Analysis set** — the subset passing movement QC with no unresolved
     doublets (``passes_movement == 1 & TraceSalvagedMultiSpot == 0``).

Trace salvage runs by default when ``--trace_salvage`` is set (always set
by ``pipeline.py``). Without it, ``TraceSalvagedMultiSpot`` mirrors
``MultiSpot`` so both files still carry the same columns.

  WalkID    continuous 1..N across all Locations (Location order, then
            first-appearance order within a Location)
  orig_ID   the per-Location WalkID it came from
  Location  source Location

  passes_movement  1 if the walk cleared the movement filter as configured

  TraceSalvagedMultiSpot  0 if no unresolved doublet loci remain after
                          trace salvage; mirrors MultiSpot when salvage
                          is skipped

  ReplicatingComplexID
            1..N for Replicating==1 walks in this output CSV, ordered by the
            globally renumbered WalkID; -1 for every other walk. Zero is never
            used. The filtered subset is renumbered independently so its
            positive IDs remain contiguous.

Row unit is (walk, timepoint), matching the source CSVs.
"""
import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE_DIR = os.path.join(_REPO_ROOT, "pipeline")
for _p in (_REPO_ROOT, _PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import center_ownership as _own
from replicating_complex_ids import annotate_dataframe_complex_ids

BASE_DEFAULT = "/mnt/altnas/work/dimos/Salmo_WF/third_replicate_outputs"
OUT_DEFAULT = "/mnt/altnas/work/dimos/Salmo_WF/third_replicate_merged"
REL = "probe_0/localizations"
VALID_CSV = "localizations_640_drift_corrected_full_coverage.csv"
MOV_CSV = "localizations_640_drift_corrected_filtered.csv"
EVIDENCE_CSV = "localizations_640_replication_qc.csv"
WALK_SUMMARY_CSV = "localizations_640_replication_classification.csv"
#: Per-attempt centre-ownership audit table. The only place a locus rejected by
#: `--center-ownership strict` is still itemized: strict drops the coordinate
#: row, so VALID_CSV records the rejection only as an absence and cannot tell
#: "rejected" from "never acquired".
OWNERSHIP_AUDIT_CSV = "localizations_640_z_gate_status.csv"
#: The UNFILTERED drift-corrected coordinates. VALID_CSV is its full-coverage
#: subset and stays the salvage decision input; this file is what lets the
#: comprehensive output still carry the accepted loci of a walk that lost full
#: coverage because one of its loci was rejected.
ALL_COORDINATES_CSV = "localizations_640_drift_corrected.csv"
#: Written by classify_replication; carries the CONFIGURED locus range under
#: hierarchy.replication_tp_{start,end}. Read rather than assumed, so the
#: expected-locus denominator is never a hardcoded 20 nor inferred from the rows
#: that happened to survive filtering.
REPLICATION_POLICY_JSON = "localizations_640_replication_policy.json"


def _configured_locus_order(locdir, explicit=None):
    """The configured genomic locus universe for one location, or None.

    Preference order: an explicit CLI order, then the range the pipeline itself
    persisted in the replication policy JSON. Never derived from surviving rows.
    """
    if explicit:
        return list(explicit)
    path = os.path.join(locdir, REPLICATION_POLICY_JSON)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            hierarchy = (json.load(handle) or {}).get('hierarchy') or {}
        start = int(hierarchy['replication_tp_start'])
        end = int(hierarchy['replication_tp_end'])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if end < start:
        return None
    return list(range(start, end + 1))


def _walk_ownership_frame(locdir, locus_order):
    """Walk-level ownership + usability columns for one location, or None.

    Reads the per-attempt audit table when present so a strict-rejected locus is
    counted as REJECTED rather than merely missing; falls back to the coordinate
    CSV's own per-locus ownership columns. Returns None when the dataset carries
    no ownership schema at all, so the fields are left visibly absent instead of
    being filled in as a clean pass.
    """
    if not locus_order:
        return None
    for name in (OWNERSHIP_AUDIT_CSV, VALID_CSV):
        path = os.path.join(locdir, name)
        if not os.path.isfile(path):
            continue
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not any(column in rows.columns
                   for column in _own.VALID_COLUMN_CANDIDATES):
            continue
        row_dicts = rows.to_dict('records')
        summary = _own.walk_ownership_summary(row_dicts, locus_order)
        if not summary:
            continue
        usab_summary = _own.walk_usability_summary(row_dicts, locus_order)
        records = []
        for walk, values in summary.items():
            rec = {'WalkID': walk, **values}
            if usab_summary and walk in usab_summary:
                rec.update(usab_summary[walk])
            else:
                for field in _own.USABILITY_WALK_FIELDS:
                    rec[field] = ''
            records.append(rec)
        frame = pd.DataFrame.from_records(records)
        if frame['WalkID'].duplicated().any():
            raise ValueError(
                f'{path}: walk-level ownership summary has duplicate WalkIDs')
        return frame, name
    return None

# Canonical v2 dataset identities.  Geometry is part of the scientific model,
# not a cosmetic label: in particular, confocal must never inherit the
# widefield defaults.  Expected locations also prevent a completely absent
# directory from disappearing silently before leave-one-location-out fitting.
TRACE_DATASET_PRESETS = {
    'rep1': {
        'location_prefix': 'Location_',
        'expected_locations': [
            'Location_1', 'Location_5', 'Location_6', 'Location_9',
            'Location_10', 'Location_11', 'Location_12', 'Location_13',
            'Location_14', 'Location_15',
        ],
        'chromosome_order': list(range(2, 22)),
        'ori_tp': 17, 'ter_tp': 7,
        'xy_pixel_nm': 103.28, 'z_step_nm': 100.0,
    },
    'rep3': {
        'location_prefix': 'Location-',
        'expected_locations': [f'Location-{index:02d}'
                               for index in range(1, 23)],
        'chromosome_order': list(range(2, 22)),
        'ori_tp': 17, 'ter_tp': 7,
        'xy_pixel_nm': 103.28, 'z_step_nm': 100.0,
    },
    # Novo shares rep1/rep3's acquisition layout -- 24 acquisitions, genomic loci
    # TP002..TP021, Ori TP017, Ter TP007 -- and differs only in treatment
    # (novobiocin) and camera geometry. It must NOT inherit rep3's 103.28 nm:
    # data.json reports WidefieldReferencePixelSizeX/Y 101.825/100.938 (mean
    # 101.38) on a 2304 px field, against rep3's 1866.
    'novo': {
        'location_prefix': 'Location-',
        'expected_locations': [f'Location-{index:02d}'
                               for index in range(1, 16)],
        'chromosome_order': list(range(2, 22)),
        'ori_tp': 17, 'ter_tp': 7,
        'xy_pixel_nm': 101.38, 'z_step_nm': 100.0,
    },
    'shx': {
        'location_prefix': 'Location-',
        'expected_locations': [f'Location-{index:02d}'
                               for index in range(1, 21)],
        'chromosome_order': list(range(2, 22)),
        'ori_tp': 17, 'ter_tp': 7,
        'xy_pixel_nm': 101.38, 'z_step_nm': 100.0,
    },
    'confocal': {
        'location_prefix': 'Location_',
        'expected_locations': [f'Location_{index}' for index in range(1, 7)],
        'chromosome_order': list(range(0, 20)),
        'ori_tp': 17, 'ter_tp': 7,
        'xy_pixel_nm': 81.25, 'z_step_nm': 180.0,
    },
}


def _tp_order(value):
    """Parse a comma-separated configured chromosome order.

    Values may be either integer indices (``2,3,4``) or TP labels
    (``TP002,TP003,TP004``).  The generated resolver config always stores
    integer indices so the representation is unambiguous.
    """
    order = []
    for token in str(value).split(','):
        token = token.strip()
        if not token:
            continue
        if token.upper().startswith('TP'):
            token = token[2:]
        try:
            order.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f'invalid TP in chromosome order: {token!r}') from exc
    if not order:
        raise argparse.ArgumentTypeError('chromosome order cannot be empty')
    if len(order) != len(set(order)):
        raise argparse.ArgumentTypeError(
            'chromosome order cannot contain duplicate TPs')
    return order


def _expected_locations(path):
    """Read an exact location universe from a JSON list or newline file."""
    source = Path(path)
    text = source.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [line.strip() for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
    if (not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip()
                   for item in value)):
        raise ValueError(
            '--trace_salvage_expected_locations must contain a non-empty '
            'JSON string list or one non-empty location per line')
    locations = [item.strip() for item in value]
    if len(locations) != len(set(locations)):
        raise ValueError('expected-location universe contains duplicates')
    return locations


def _manifest_trace_preset(manifest_path):
    """Trace-salvage settings derived from a DATASET MANIFEST.

    The presets above are a per-dataset table in code, so a new experiment could
    not be salvaged without editing this file. Everything they hold -- the locus
    order, Ori/Ter, the detector calibration, the Location universe -- is already
    an acquisition fact in the manifest, so read it from there instead.
    """
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pipeline'))
    import dataset_manifest as _dm
    m = _dm.load(manifest_path)
    prefix = m.location_glob.split('*')[0]
    if not prefix:
        raise ValueError(
            f'{manifest_path}: location_glob {m.location_glob!r} has no literal '
            'prefix, so the Location naming convention cannot be derived')
    return {
        'location_prefix': prefix,
        'expected_locations': list(m.locations),
        # The manifest names loci canonically (TP002); the salvage config uses
        # their integer indices.
        'chromosome_order': [int(str(name)[2:]) for name in m.chromosome_order],
        'ori_tp': m.ori_tp, 'ter_tp': m.ter_tp,
        'xy_pixel_nm': m.xy_pixel_nm, 'z_step_nm': m.z_step_nm,
    }


def _trace_settings(args):
    """Resolve canonical geometry/order without modality-dependent defaults."""
    dataset = str(args.trace_salvage_dataset or '').strip().lower()
    if not dataset:
        raise ValueError(
            '--trace_salvage_dataset is required when --trace_salvage is set; '
            'use a dataset manifest via --trace_salvage_manifest, or an '
            'explicit custom configuration')
    if getattr(args, 'trace_salvage_manifest', None):
        preset = _manifest_trace_preset(args.trace_salvage_manifest)
    else:
        preset = TRACE_DATASET_PRESETS.get(dataset)

    values = {'dataset': dataset}
    for attr in ('chromosome_order', 'ori_tp', 'ter_tp',
                 'xy_pixel_nm', 'z_step_nm'):
        cli_value = getattr(args, f'trace_salvage_{attr}')
        if cli_value is None and preset is not None:
            preset_value = preset[attr]
            cli_value = (list(preset_value)
                         if isinstance(preset_value, list) else preset_value)
        if cli_value is None:
            raise ValueError(
                f'custom trace-salvage dataset {dataset!r} requires '
                f'--trace_salvage_{attr}')
        values[attr] = cli_value

    if values['ori_tp'] not in values['chromosome_order']:
        raise ValueError('trace-salvage Ori TP is absent from chromosome order')
    if values['ter_tp'] not in values['chromosome_order']:
        raise ValueError('trace-salvage Ter TP is absent from chromosome order')
    if values['ori_tp'] == values['ter_tp']:
        raise ValueError('trace-salvage Ori and Ter must differ')
    if values['xy_pixel_nm'] <= 0 or values['z_step_nm'] <= 0:
        raise ValueError('trace-salvage pixel and z-step calibration must be > 0')

    values['location_prefix'] = (
        args.location_prefix
        or (preset['location_prefix'] if preset is not None else None))
    if values['location_prefix'] is None:
        raise ValueError(
            f'custom trace-salvage dataset {dataset!r} requires '
            '--location_prefix')

    if args.trace_salvage_expected_locations:
        values['expected_locations'] = _expected_locations(
            args.trace_salvage_expected_locations)
    elif preset is not None and not args.trace_salvage_policy:
        values['expected_locations'] = list(preset['expected_locations'])
    elif not args.trace_salvage_policy:
        raise ValueError(
            'fitting a custom dataset requires '
            '--trace_salvage_expected_locations so an absent location cannot '
            'be silently omitted')
    else:
        values['expected_locations'] = None
    return values


def _report_ownership(frame, sources):
    """State plainly whether ownership travelled, and from where."""
    present = [column for column in _own.OWNERSHIP_ROW_FIELDS
               if column in frame.columns]
    print(f"  per-locus ownership columns: {len(present)}/"
          f"{len(_own.OWNERSHIP_ROW_FIELDS)}"
          + ("" if len(present) == len(_own.OWNERSHIP_ROW_FIELDS)
             else " (upstream predates the ownership schema)"))
    usab_present = [column for column in _own.USABILITY_ROW_FIELDS
                    if column in frame.columns]
    if usab_present:
        print(f"  per-locus usability columns: {len(usab_present)}/"
              f"{len(_own.USABILITY_ROW_FIELDS)}")
    kinds = sorted(set(sources.values()))
    print(f"  walk-level ownership source: {', '.join(kinds) or 'none'}")
    if "CenterOwnershipComplete" not in frame.columns:
        return
    per_walk = frame.groupby("WalkID")["CenterOwnershipComplete"].first()
    complete = (per_walk.astype(str).str.strip() == "1").sum()
    blank = (per_walk.astype(str).str.strip() == "").sum()
    print(f"  CenterOwnershipComplete: {complete:,} of {len(per_walk):,} walks"
          + (f"; {blank:,} not determinable" if blank else ""))
    if "CenterOwnershipUsableComplete" in frame.columns:
        per_walk_u = frame.groupby("WalkID")["CenterOwnershipUsableComplete"].first()
        usab_complete = (per_walk_u.astype(str).str.strip() == "1").sum()
        print(f"  CenterOwnershipUsableComplete: {usab_complete:,} of "
              f"{len(per_walk_u):,} walks")


def _run_trace_salvage(args, location_dirs, settings):
    """Generate the explicit v2 resolver config and run it once per dataset."""
    locations = []
    missing = []
    for location, locdir in location_dirs:
        evidence = (locdir / EVIDENCE_CSV).resolve()
        coordinates = (locdir / VALID_CSV).resolve()
        walk_summary = (locdir / WALK_SUMMARY_CSV).resolve()
        if not evidence.is_file():
            missing.append(str(evidence))
        if not coordinates.is_file():
            missing.append(str(coordinates))
        if not walk_summary.is_file():
            missing.append(str(walk_summary))
        entry = {
            'location': location,
            'evidence_csv': str(evidence),
            'coordinates_csv': str(coordinates),
            'walk_summary_csv': str(walk_summary),
        }
        # Optional, and only when it actually exists: the resolver falls back to
        # the coordinate CSV's own ownership columns, and a dataset with neither
        # keeps its pre-ownership behaviour.
        ownership = (locdir / OWNERSHIP_AUDIT_CSV).resolve()
        if ownership.is_file():
            entry['ownership_audit_csv'] = str(ownership)
            # Only meaningful together with the ledger, and required by the
            # resolver as soon as the ledger lists an accepted attempt that the
            # full-coverage file does not contain.
            all_coordinates = (locdir / ALL_COORDINATES_CSV).resolve()
            if not all_coordinates.is_file():
                raise FileNotFoundError(
                    f'{location}: {ownership.name} defines the attempt '
                    f'universe but {ALL_COORDINATES_CSV} is missing, so '
                    'accepted loci of coverage-dropped walks could not be '
                    'represented')
            entry['all_coordinates_csv'] = str(all_coordinates)
        locations.append(entry)
    if missing:
        raise FileNotFoundError(
            'trace salvage requires evidence, drift-corrected full-coverage '
            'coordinates, and the classified walk summary for every finalized '
            'location; missing: '
            + ', '.join(missing))

    config_payload = {
        'schema_version': 'trace_salvage_input_v1',
        'dataset': settings['dataset'],
        'evidence_schema': 'v2',
        'chromosome_order': settings['chromosome_order'],
        'ori_tp': settings['ori_tp'],
        'ter_tp': settings['ter_tp'],
        'geometry': {
            'xy_pixel_nm': settings['xy_pixel_nm'],
            'z_step_nm': settings['z_step_nm'],
        },
        # Freeze the agreed conservative rule in the generated provenance,
        # rather than relying on mutable implementation defaults.
        'salvage_policy': settings.get('salvage_policy', 'calibrated_loo'),
        'min_calibration_links': 100,
        'clean_link_quantile': 0.95,
        'winner_ratio': 0.8,
        'locations': locations,
    }
    salvage_outdir = Path(
        args.trace_salvage_outdir or args.outdir).resolve()
    salvage_outdir.mkdir(parents=True, exist_ok=True)
    config_path = salvage_outdir / (
        f'{args.trace_salvage_output_prefix}_trace_salvage_config.json')
    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', dir=salvage_outdir, delete=False,
                prefix='.trace_salvage_config_', suffix='.json') as handle:
            tmp_path = Path(handle.name)
            json.dump(config_payload, handle, indent=2)
            handle.write('\n')
        os.replace(tmp_path, config_path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    try:
        from trace_salvage import (
            apply_policy, load_config, run_dataset,
        )
    except ImportError as exc:
        raise RuntimeError(
            'trace salvage was requested but pipeline.trace_salvage could not '
            'be imported') from exc

    config = load_config(config_path)
    if args.trace_salvage_policy:
        with Path(args.trace_salvage_policy).open() as handle:
            policy = json.load(handle)
        result = apply_policy(
            config, policy,
            outdir=salvage_outdir,
            output_prefix=args.trace_salvage_output_prefix,
            diagnostics=args.trace_salvage_diagnostics,
        )
    else:
        if len(locations) < 2:
            raise ValueError(
                'trace salvage cannot fit a leave-one-location-out policy '
                'from one location; pass --trace_salvage_policy with a '
                'precomputed compatible policy')
        result = run_dataset(
            config,
            outdir=salvage_outdir,
            output_prefix=args.trace_salvage_output_prefix,
            diagnostics=args.trace_salvage_diagnostics,
        )
    print(f'\nTrace salvage (explicit opt-in): {config_path}')
    for key, path in sorted(result.items()):
        print(f'  {key}: {path}')
    return result


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=BASE_DEFAULT)
    p.add_argument("--outdir", default=OUT_DEFAULT)
    p.add_argument("--out_name", default="all_locations_640_valid_walks.csv")
    p.add_argument("--clean_name",
                   default="all_locations_640_valid_walks_nosuspect_movement.csv")
    p.add_argument(
        '--ownership_locus_order', type=_tp_order, default=None,
        help='Comma-separated configured genomic locus universe used as the '
             'CenterOwnershipExpectedCount denominator. Omit to read the range '
             'the pipeline persisted in '
             f'{REPLICATION_POLICY_JSON} (hierarchy.replication_tp_start/end).')
    p.add_argument(
        '--location_prefix', default=None,
        help='Directory-name prefix used to discover locations. Legacy '
             'default: Location-. Trace-salvage presets select the canonical '
             'prefix; custom datasets must set it explicitly.')
    salvage = p.add_argument_group(
        'optional dataset-level trace salvage (OFF by default)')
    salvage.add_argument(
        '--trace_salvage', action='store_true',
        help='After writing the legacy merged outputs, run conservative v2 '
             'trace salvage and write separate audited outputs.')
    salvage.add_argument(
        '--trace_salvage_manifest', default=None,
        help='Dataset manifest supplying the locus order, Ori/Ter, detector '
             'calibration and Location universe. Preferred over the in-code '
             'presets, which cannot describe a new experiment.')
    salvage.add_argument(
        '--trace_salvage_dataset', default=None,
        help='Required canonical dataset identity. rep1, rep3, and confocal '
             'select modality-correct geometry/order presets; a custom name '
             'requires every geometry/order flag explicitly.')
    salvage.add_argument(
        '--trace_salvage_outdir', default=None,
        help='Directory for salvage outputs (default: --outdir).')
    salvage.add_argument(
        '--trace_salvage_diagnostics', action='store_true',
        help='Also write the per-locus salvage audit CSV, the reduced walk '
             'summary and the per-Location comprehensive files. Off by '
             'default: every field they hold is already on every row of the '
             'comprehensive CSV. Changes only which files appear, never a '
             'coordinate, a WalkID mapping, walk membership or a decision.')
    salvage.add_argument(
        '--trace_salvage_output_prefix', default='all_locations_640',
        help='Prefix for separate salvage outputs (default: %(default)s).')
    salvage.add_argument(
        '--trace_salvage_policy', default=None,
        help='Apply an existing compatible policy instead of fitting one. '
             'Required for a single-location finalization.')
    salvage.add_argument(
        '--trace_salvage_expected_locations', default=None,
        help='Optional exact expected-location universe as a JSON string list '
             'or newline file. Canonical presets enforce their full dataset '
             'when fitting; custom fits require this option.')
    salvage.add_argument(
        '--trace_salvage_chromosome_order', type=_tp_order,
        default=None,
        help='Comma-separated genomic TP order. Omit for a canonical preset.')
    salvage.add_argument('--trace_salvage_ori_tp', type=int, default=None)
    salvage.add_argument('--trace_salvage_ter_tp', type=int, default=None)
    salvage.add_argument(
        '--trace_salvage_xy_pixel_nm', type=float, default=None,
        help='Lateral pixel calibration. Omit for a canonical preset.')
    salvage.add_argument(
        '--trace_salvage_z_step_nm', type=float, default=None,
        help='Axial step calibration. Omit for a canonical preset.')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    trace_settings = _trace_settings(args) if args.trace_salvage else None
    location_prefix = (
        trace_settings['location_prefix'] if trace_settings is not None
        else (args.location_prefix or 'Location-'))
    locs = sorted(
        d for d in os.listdir(args.base)
        if d.startswith(location_prefix))
    if trace_settings is not None and trace_settings['expected_locations']:
        expected = set(trace_settings['expected_locations'])
        discovered = set(locs)
        if discovered != expected:
            raise FileNotFoundError(
                'trace-salvage location universe mismatch; '
                f'missing={sorted(expected - discovered)}, '
                f'unexpected={sorted(discovered - expected)}')
    counter, parts, summary = 0, [], []
    ownership_sources = {}
    # Keep the complete discovered set for the explicit resolver.  The legacy
    # merge may skip an incomplete location, but a requested calibrated run
    # must fail closed rather than silently calibrating on a smaller dataset.
    location_dirs = [
        (loc, Path(args.base) / loc / REL) for loc in locs
    ]
    for loc in locs:
        d = os.path.join(args.base, loc, REL)
        path = os.path.join(d, VALID_CSV)
        if not os.path.isfile(path):
            print(f"  [skip] {loc}: no {VALID_CSV}")
            continue
        df = pd.read_csv(path)
        mov_path = os.path.join(d, MOV_CSV)
        mov = (set(pd.read_csv(mov_path, usecols=["WalkID"]).WalkID.unique())
               if os.path.isfile(mov_path) else set())

        # Walk-level centre ownership, joined on the LOCATION-LOCAL WalkID
        # before global renumbering. Validated many-to-one so a duplicated key
        # raises instead of silently multiplying coordinate rows.
        locus_order = _configured_locus_order(d, args.ownership_locus_order)
        own_result = _walk_ownership_frame(d, locus_order)
        if own_result is None:
            ownership_sources[loc] = 'absent'
            for column in _own.OWNERSHIP_WALK_FIELDS:
                df[column] = ''
            for column in _own.USABILITY_WALK_FIELDS:
                df[column] = ''
        else:
            own_frame, own_source = own_result
            ownership_sources[loc] = own_source
            rows_before = len(df)
            df = df.merge(own_frame, on='WalkID', how='left',
                          validate='many_to_one')
            if len(df) != rows_before:
                raise AssertionError(
                    f'{loc}: ownership join changed row count '
                    f'{rows_before} -> {len(df)}')
            for column in _own.OWNERSHIP_WALK_FIELDS:
                df[column] = df[column].where(df[column].notna(), '')
            for column in _own.USABILITY_WALK_FIELDS:
                if column in df.columns:
                    df[column] = df[column].where(df[column].notna(), '')

        uniq = df["WalkID"].drop_duplicates().tolist()   # first-appearance order
        mapping = {o: counter + i + 1 for i, o in enumerate(uniq)}
        counter += len(uniq)

        df["orig_ID"] = df["WalkID"]
        df["passes_movement"] = df.WalkID.isin(mov).astype(int)
        df["WalkID"] = df["WalkID"].map(mapping)
        df["Location"] = loc
        parts.append(df)
        summary.append((loc, len(uniq), len(mov)))

    if not parts:
        raise SystemExit("no Locations found")
    allw = pd.concat(parts, ignore_index=True)
    tail = [*_own.OWNERSHIP_WALK_FIELDS, *_own.USABILITY_WALK_FIELDS,
            "passes_movement", "orig_ID", "Location"]
    allw = allw[[c for c in allw.columns if c not in tail] + tail]
    allw = annotate_dataframe_complex_ids(allw)

    # -- Trace salvage: join walk-level columns into the merge frame ----------
    _TRACE_WALK_COLUMNS = [
        'TraceSalvagedMultiSpot',
        'TraceSalvageStatus',
        'TraceSalvagedLociCount',
        'TraceRawDoubletLociCount',
        'TraceUnresolvedDoubletLociCount',
        'TraceSalvageEligibleLociCount',
        'TraceSalvagedSingleSpotComplete',
    ]

    if args.trace_salvage:
        salvage_result = _run_trace_salvage(
            args, location_dirs, trace_settings)
        # run_dataset()/apply_policy() return the salvage manifest; output
        # paths sit under manifest['outputs'][name]['path'] as basenames
        # relative to the salvage outdir.
        fs_entry = (salvage_result.get('outputs') or {}).get('full_schema')
        fs_name = fs_entry.get('path') if isinstance(fs_entry, dict) else fs_entry
        full_schema_path = (
            Path(args.trace_salvage_outdir or args.outdir) / fs_name
            if fs_name else None)
        if full_schema_path and os.path.isfile(str(full_schema_path)):
            # The salvage file numbers walks over the attempt-ledger universe,
            # so its global WalkID is NOT the merge's WalkID. Join on the
            # location-local key: (Location, OriginalWalkID) == (Location, orig_ID).
            salvage_df = pd.read_csv(
                str(full_schema_path),
                usecols=['Location', 'OriginalWalkID'] + _TRACE_WALK_COLUMNS,
                low_memory=False)
            salvage_walk = (salvage_df
                            .groupby(['Location', 'OriginalWalkID'])[_TRACE_WALK_COLUMNS]
                            .first()
                            .reset_index()
                            .rename(columns={'OriginalWalkID': 'orig_ID'}))
            salvage_walk['orig_ID'] = pd.to_numeric(salvage_walk['orig_ID'])
            allw['orig_ID'] = pd.to_numeric(allw['orig_ID'])
            rows_before = len(allw)
            allw = allw.merge(salvage_walk, on=['Location', 'orig_ID'],
                              how='left', validate='many_to_one')
            assert len(allw) == rows_before, (
                f'salvage join changed row count {rows_before} -> {len(allw)}')
            n_missing = int(allw['TraceSalvagedMultiSpot'].isna().sum())
            if n_missing:
                print(f'  [salvage] {n_missing} rows have no salvage data; '
                      f'falling back to MultiSpot')
                allw['TraceSalvagedMultiSpot'] = (
                    allw['TraceSalvagedMultiSpot']
                    .fillna(allw['MultiSpot']).astype(int))
                for col in _TRACE_WALK_COLUMNS:
                    if col != 'TraceSalvagedMultiSpot':
                        allw[col] = allw[col].fillna('')
            print(f'  [salvage] joined {len(_TRACE_WALK_COLUMNS)} '
                  f'walk-level columns into merge frame')
        else:
            print('  [salvage] WARNING: full_schema not found, '
                  'falling back to MultiSpot')
            allw['TraceSalvagedMultiSpot'] = allw['MultiSpot']
            for col in _TRACE_WALK_COLUMNS:
                if col not in allw.columns:
                    allw[col] = ''
    else:
        allw['TraceSalvagedMultiSpot'] = allw['MultiSpot']
        for col in _TRACE_WALK_COLUMNS:
            if col not in allw.columns:
                allw[col] = ''

    # -- Trace MAD outlier: flag walks with outlier TP positions ---------------
    _MAD_K = 3.0
    tp_centroids = (allw.groupby(['WalkID', 'tp'])[['x_nm', 'y_nm', 'z_nm']]
                    .mean())
    walk_medians = tp_centroids.groupby('WalkID').median()
    dev = tp_centroids.subtract(walk_medians, level='WalkID')
    dists = np.sqrt((dev ** 2).sum(axis=1))
    walk_max_dev = dists.groupby('WalkID').max()

    eligible = allw.groupby('WalkID')[['passes_movement',
                                       'TraceSalvagedMultiSpot']].first()
    eligible_mask = ((eligible['passes_movement'] == 1)
                     & (eligible['TraceSalvagedMultiSpot'] == 0))
    eligible_vals = walk_max_dev[eligible_mask].values
    med_dev = float(np.median(eligible_vals))
    mad_dev = float(np.median(np.abs(eligible_vals - med_dev)))
    mad_sigma = 1.4826 * mad_dev
    threshold = med_dev + _MAD_K * mad_sigma if mad_sigma > 0 else np.inf
    walk_outlier = (walk_max_dev > threshold).astype(int)

    mad_df = pd.DataFrame({
        'trace_max_tp_deviation_nm': walk_max_dev,
        'trace_mad_outlier': walk_outlier,
    })
    rows_before = len(allw)
    allw = allw.merge(mad_df, on='WalkID', how='left')
    assert len(allw) == rows_before

    n_outlier = int(walk_outlier[eligible_mask].sum())
    n_eligible = int(eligible_mask.sum())
    print(f'\n  [MAD outlier] metric=max TP centroid deviation from '
          f'walk median (nm)')
    print(f'  [MAD outlier] population={n_eligible} walks '
          f'(passes_movement & !MultiSpot)')
    print(f'  [MAD outlier] median={med_dev:.1f} nm, '
          f'MAD-sigma={mad_sigma:.1f} nm, '
          f'threshold (k={_MAD_K})={threshold:.1f} nm')
    print(f'  [MAD outlier] flagged={n_outlier}/{n_eligible} '
          f'({100*n_outlier/n_eligible:.1f}%)')

    # -- Write full file ------------------------------------------------------
    out = os.path.join(args.outdir, args.out_name)
    allw.to_csv(out, index=False)

    print(f"\n{'Location':<14}{'valid walks':>13}{'passes movement':>17}")
    for loc, a, m in summary:
        print(f"{loc:<14}{a:>13}{m:>17}")
    print("-" * 44)
    print(f"{'TOTAL':<14}{counter:>13}"
          f"{int(allw.groupby('WalkID').passes_movement.first().sum()):>17}")
    print(f"\n{args.out_name}")
    print(f"  rows={len(allw)}  cols={len(allw.columns)}  walks={allw.WalkID.nunique()}"
          f"  WalkID 1..{allw.WalkID.max()}"
          f"  continuous={allw.WalkID.nunique() == allw.WalkID.max()}")
    for c in ("movement_nm", "max_consecutive_confirmed_run"):
        print(f"  {c}: {'present' if c in allw.columns else 'ABSENT (re-run the step that writes it)'}")
    _report_ownership(allw, ownership_sources)
    print(f"  {os.path.getsize(out)/1e6:.0f} MB")

    # -- Write analysis set (filtered file) -----------------------------------
    sub = allw[(allw.passes_movement == 1)
               & (allw.TraceSalvagedMultiSpot == 0)]
    sub = annotate_dataframe_complex_ids(sub)
    sub_path = os.path.join(args.outdir, args.clean_name)
    sub.to_csv(sub_path, index=False)
    print(f"\n{args.clean_name}")
    print(f"  passes_movement=1, TraceSalvagedMultiSpot=0")
    print(f"  rows={len(sub)}  walks={sub.WalkID.nunique()}"
          f"  {os.path.getsize(sub_path)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
