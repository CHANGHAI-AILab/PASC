"""Patient-level descriptors and mapped consolidation (Supplementary Method S3).

    python scripts/02_patient_features.py slide_features.csv patient_features.csv [--cohort cohort.csv]

The input is the output of 01_slide_features.py.

cohort.csv (patient_id, cohort = train or Training / validation / external) is optional; when given
it is merged and the training mean/SD of Consolidation_z are printed as a check. The fixed
training constants in constants.py are applied; nothing is refitted.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pasc_consol import analysis_table, composite, descriptors  # noqa: E402
from pasc_consol.constants import COMPOSITE_TRAINING_MEDIAN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slides')
    ap.add_argument('out_csv')
    ap.add_argument('--cohort')
    a = ap.parse_args()

    sl = pd.read_csv(a.slides, dtype={'patient_id': str, 'slide_id': str})
    unlinked = sl['patient_id'].isna()
    if unlinked.any():
        print(f'skipping {int(unlinked.sum())} slides without patient_id: {sl.slide_id[unlinked].tolist()}')
        sl = sl[~unlinked]
    print(f'{len(sl)} slides, {int(sl.eligible.sum())} eligible (tumor area >= 1.2e6 um2)')
    df = pd.DataFrame([{'patient_id': pid, **descriptors.patient_descriptors(g.to_dict('records'))}
                       for pid, g in sl.groupby('patient_id', sort=False) if g.eligible.any()])
    df['Consolidation_z'] = composite.mapped_consolidation(df)
    df['Consolidation_high'] = (df['Consolidation_z'] >= COMPOSITE_TRAINING_MEDIAN).astype(int)

    if a.cohort:
        cohort = pd.read_csv(a.cohort, dtype={'patient_id': str})
        df = cohort.merge(df, on='patient_id', how='left', validate='one_to_one')
        missing = df['D1_raw'].isna().sum()
        if missing:
            print(f'warning: {missing} cohort patients without an eligible slide')
        tr = df[analysis_table.is_training(df['cohort'])]
        print(f'training check: n={len(tr)}, Consolidation_z mean={tr.Consolidation_z.mean():.4f} '
              f'SD={tr.Consolidation_z.std(ddof=1):.4f} (expected ~0 and 1 when the training set matches)')
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f'{len(df)} patients -> {a.out_csv}')


if __name__ == '__main__':
    main()
