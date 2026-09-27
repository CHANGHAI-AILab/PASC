"""Derived fields of the patient-level analysis table (Methods; Supplementary Methods S3-S5).

Input columns follow the shared data dictionary (docs/data_dictionary.md): reader scores
Human_consol_P1..P3, Age, tumor_size_cm_archived, archived_pT_exact_for_audit,
pT_ge3_original, SCC_proportion_PATHOLOGIST, Eligible_slides, and the patient descriptors
D1_raw..D5_raw, C_raw, SCC_in_tumor_pt from 02_patient_features.py. Every constant is the
fixed training value; nothing is refitted.
"""
import numpy as np
import pandas as pd

from .composite import mapped_consolidation, visual_consolidation
from .constants import (COMPOSITE_INPUTS, COMPOSITE_TRAINING_MEDIAN,
                        COVARIATE_TRAINING_CONSTANTS, T_SIZE_CUTOFFS_CM, TRAINING_CONSTANTS,
                        TRAINING_COHORT_LABELS, VISUAL_TRAINING_MEDIAN)

READERS = ('Human_consol_P1', 'Human_consol_P2', 'Human_consol_P3')


def is_training(cohort):
    return cohort.astype(str).str.strip().str.lower().isin(TRAINING_COHORT_LABELS)


def t_category(size_cm, archived_pt):
    """T1-T3 from maximum gross diameter; archived pT4 (arterial involvement) retained."""
    lo, hi = T_SIZE_CUTOFFS_CM
    size_t = np.select([size_cm <= lo, size_cm <= hi], [1, 2], 3)
    final_t = np.where(archived_pt == 4, 4, size_t)
    return size_t, final_t


def _z(x, key, constants):
    mu, sd = constants[key]
    return (x - mu) / sd


def derive(df):
    """Return the derived columns; `df` needs the input columns listed in the module docstring."""
    out = visual_consolidation(df, READERS)
    out['Human_high'] = (out['Human_consol_mean'] > VISUAL_TRAINING_MEDIAN).astype(int)
    p1, p2, p3 = (df[c] for c in READERS)
    out['Reader12_mean'] = (p1 + p2) / 2
    out['Reader13_mean'] = (p1 + p3) / 2
    out['Reader23_mean'] = (p2 + p3) / 2

    # Mapped measures (Method S3)
    out['SCC_proportion_AI_RAW_fraction'] = df['SCC_in_tumor_pt']
    out['SCC_proportion_AI_RAW_pct'] = (100 * df['SCC_in_tumor_pt']).round(1)
    for c in COMPOSITE_INPUTS:
        d = c.replace('_raw', '')
        out[d] = df[c]
        out[d + '_z'] = _z(df[c], c, TRAINING_CONSTANTS)
    out['Consolidation_z'] = mapped_consolidation(df)
    out['Consol_high'] = (out['Consolidation_z'] >= COMPOSITE_TRAINING_MEDIAN).astype(int)
    out['Composition_z'] = _z(df['C_raw'], 'C_raw', TRAINING_CONSTANTS)

    # T category (Method S5)
    size_t, final_t = t_category(df['tumor_size_cm_archived'], df['archived_pT_exact_for_audit'])
    out['pT_size_harmonized_exact'] = size_t
    out['pT_ge3'] = (size_t >= 3).astype(int)
    out['pT_harmonized_final_exact'] = final_t
    out['pT_harmonized_T3T4'] = (final_t >= 3).astype(int)
    out['archived_pT4_flag_for_review'] = (df['archived_pT_exact_for_audit'] == 4).astype(int)
    out['pT_ge3_size_plus_archived_T4_sensitivity'] = (out['pT_ge3'] | out['archived_pT4_flag_for_review'])
    orig = df['pT_ge3_original']
    out['pT_changed_or_filled'] = (orig.isna() | (orig != out['pT_ge3'])).astype(int)

    # Survival-model covariates (Method S5)
    k = COVARIATE_TRAINING_CONSTANTS
    out['visual_z'] = _z(out['Human_consol_mean'], 'Human_consol_mean', k)
    out['size_z'] = _z(df['tumor_size_cm_archived'], 'tumor_size_cm_archived', k)
    out['Age_z'] = _z(df['Age'], 'Age', k)
    out['eligible_z'] = _z(df['Eligible_slides'], 'Eligible_slides', k)
    out['path10'] = df['SCC_proportion_PATHOLOGIST'] / 10
    return out


def training_constants_check(df, derived):
    """Recompute every fixed constant in the training rows; returns (name, fixed, recomputed)."""
    tr = is_training(df['Cohort_label'])
    rows = []
    for c, (mu, sd) in TRAINING_CONSTANTS.items():
        if c in df:
            rows += [(f'{c} mean', mu, df.loc[tr, c].mean()), (f'{c} SD', sd, df.loc[tr, c].std(ddof=1))]
    z = np.column_stack([derived.loc[tr, c.replace('_raw', '') + '_z'] for c in COMPOSITE_INPUTS])
    s = -z.mean(axis=1)
    rows += [('composite SD', 0.439442, s.std(ddof=1)),
             ('composite median', COMPOSITE_TRAINING_MEDIAN, np.median(s / s.std(ddof=1)))]
    src = pd.concat([df, derived[['Human_consol_mean']]], axis=1).loc[:, lambda d: ~d.columns.duplicated(keep='last')]
    for c, (mu, sd) in COVARIATE_TRAINING_CONSTANTS.items():
        rows += [(f'{c} mean', mu, src.loc[tr, c].mean()), (f'{c} SD', sd, src.loc[tr, c].std(ddof=1))]
    rows.append(('Human_consol_mean median', VISUAL_TRAINING_MEDIAN, src.loc[tr, 'Human_consol_mean'].median()))
    return pd.DataFrame(rows, columns=['constant', 'fixed', 'recomputed']).assign(
        diff=lambda d: d.recomputed - d.fixed, n_training=int(tr.sum()))
