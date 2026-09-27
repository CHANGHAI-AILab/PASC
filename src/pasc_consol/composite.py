"""Mapped consolidation (Supplementary Method S3) and visual consolidation (Method S4)."""
import numpy as np
import pandas as pd

from .constants import COMPOSITE_INPUTS, COMPOSITE_TRAINING_SD, TRAINING_CONSTANTS


def standardize(df, constants=TRAINING_CONSTANTS, suffix='_z'):
    out = pd.DataFrame(index=df.index)
    for col, (mu, sd) in constants.items():
        if col in df:
            out[col.replace('_raw', '') + suffix] = (df[col] - mu) / sd
    return out


def mapped_consolidation(df, inputs=COMPOSITE_INPUTS, constants=TRAINING_CONSTANTS,
                         composite_sd=COMPOSITE_TRAINING_SD):
    """Consolidation_z: mean of training-standardized D1-D5, sign reversed, / training SD.

    Higher values = greater consolidation.
    """
    z = np.column_stack([(df[c] - constants[c][0]) / constants[c][1] for c in inputs])
    return pd.Series(-z.mean(axis=1) / composite_sd, index=df.index, name='Consolidation_z')


def refit_composite_constants(train_df, inputs=COMPOSITE_INPUTS):
    """Recompute training constants (for the repeat-calculation check only)."""
    consts = {c: (train_df[c].mean(), train_df[c].std(ddof=1)) for c in inputs}
    z = np.column_stack([(train_df[c] - consts[c][0]) / consts[c][1] for c in inputs])
    return consts, float((-z.mean(axis=1)).std(ddof=1))


def visual_consolidation(df, readers=('Human_consol_P1', 'Human_consol_P2', 'Human_consol_P3')):
    """Unrounded three-reader mean and its descriptive rounded level (no half-integer ties)."""
    mean = df[list(readers)].mean(axis=1)
    return pd.DataFrame({'Human_consol_mean': mean,
                         'Visual_rounded': np.floor(mean + 0.5).astype(int)}, index=df.index)
