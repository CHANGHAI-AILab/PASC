"""Derived fields of the patient-level analysis table (Methods; Supplementary Methods S3-S5).

    python scripts/03_analysis_table.py table.xlsx out.csv [--sheet SHEET] [--features patient_features.csv]
                                        [--id Slide_Number] [--check]

table.xlsx / .csv is the patient-level analysis table (one row per patient; inputs in
docs/data_dictionary.md). With --features, the mapped descriptors (SCC_in_tumor_pt, C_raw,
D1_raw..D5_raw, and B_raw, S1_raw..S3_raw, V_raw when present) are taken from
02_patient_features.py, linked on --id = patient_id; otherwise
the table's own SCC_in_tumor_pt and D1..D5 are used.

--check compares every derived column with the same-named column already present in the
table and recomputes the fixed training constants in the Training rows.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pasc_consol import analysis_table  # noqa: E402
from pasc_consol.constants import COMPOSITE_INPUTS, LOGIT_EPS, TRAINING_CONSTANTS  # noqa: E402

DESCRIPTORS = ('SCC_in_tumor_pt', *TRAINING_CONSTANTS)


def read_table(path, sheet):
    path = Path(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def attach_descriptors(tab, features, id_col):
    if features is None:
        d = tab.copy()
        for c in COMPOSITE_INPUTS:
            d[c] = d[c.replace('_raw', '')]
        p = d['SCC_in_tumor_pt'].clip(LOGIT_EPS, 1 - LOGIT_EPS)
        d['C_raw'] = np.log(p / (1 - p))
        return d
    feat = pd.read_csv(features, dtype={'patient_id': str})
    feat = feat[['patient_id', *[c for c in DESCRIPTORS if c in feat]]]
    d = tab.drop(columns=[c for c in DESCRIPTORS if c in tab]).assign(_id=tab[id_col].astype(str))
    d = d.merge(feat, left_on='_id', right_on='patient_id', how='left', validate='one_to_one')
    missing = d['D1_raw'].isna()
    if missing.any():
        print(f'warning: {int(missing.sum())} patients without mapped descriptors')
    return d.drop(columns=['_id', 'patient_id'])


def check(tab, derived):
    rows = []
    for c in derived.columns:
        if c not in tab:
            continue
        a, b = pd.to_numeric(tab[c], errors='coerce'), pd.to_numeric(derived[c], errors='coerce')
        diff = (a - b).abs()
        rows.append((c, float(diff.max()), int((diff > 1e-4).sum())))
    print('\nderived column vs table: max |diff|, rows > 1e-4')
    for c, m, n in rows:
        print(f'  {c:42s} {m:10.3g} {n:4d}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table')
    ap.add_argument('out_csv')
    ap.add_argument('--sheet')
    ap.add_argument('--features')
    ap.add_argument('--id', default='Slide_Number')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    tab = read_table(a.table, a.sheet)
    src = attach_descriptors(tab, a.features, a.id)
    derived = analysis_table.derive(src)
    if a.check:
        check(tab, derived)
        if 'Cohort_label' in src:
            k = analysis_table.training_constants_check(src, derived)
            print(f'\ntraining constants recomputed in {k.n_training.iloc[0]} training patients')
            print(k.drop(columns='n_training').to_string(index=False, float_format=lambda v: f'{v:.6f}'))
    keep = [c for c in tab.columns if c not in derived]
    out = pd.concat([tab[keep], derived], axis=1)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    print(f'\n{len(out)} patients -> {a.out_csv}')


if __name__ == '__main__':
    main()
