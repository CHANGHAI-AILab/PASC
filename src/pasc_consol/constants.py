"""Constants of the feature pipeline (Supplementary Methods S2-S5, Table S3).

Definitions are listed in docs/implementation.md.
All standardization constants were estimated once in the training patients and are applied
unchanged to the validation and external test cohorts. Do not refit them.
"""

# Label coding of tissue maps
LABEL_BACKGROUND = 0
LABEL_SCC = 1
LABEL_ADC = 2
LABEL_IGNORE = 255

# Descriptors are computed on the mapping-scale label map (level 0 reduced 40x per axis:
# 10 um/px for 0.25 um/px scans). Lengths and areas are in its pixels unless stated.
# Region descriptors (Method S3)
CONNECTIVITY = 8
MIN_REGION_PIXELS = 10              # applies to CCCount and DCR
SIGNIFICANT_REGION_FRACTION = 0.05  # counted over all regions, no size floor
MIN_TUMOR_AREA_UM2 = 1.2e6          # slide eligibility: (SCC + ADC pixels) x mpp_x x mpp_y, um^2
# Inclusive cutoff: exactly 12,000 tumor pixels when mpp_x = mpp_y = 10 um/px.
SHAPE_CLOSING_KERNEL = (3, 3)       # rectangular, used only before contour tracing (B)
INTERFACE_DILATION_RADIUS = 1       # elliptical (2r+1) kernel (S1)
CANNY_THRESHOLDS = (50, 150)
FRONT_THICKNESS_PX = 500            # S2; S3 is the mean distance, in pixels
LOGIT_EPS = 1e-6

# Training mean / sample SD of the raw patient-level descriptors (Table S3)
TRAINING_CONSTANTS = {
    'C_raw': (0.340959, 1.023500),
    'D1_raw': (0.673560, 0.196579),
    'D2_raw': (0.969697, 0.172733),
    'D3_raw': (6.179709, 0.929012),
    'D4_raw': (1.575454, 0.297984),
    'D5_raw': (0.176059, 0.076425),
    'B_raw': (5.803356, 1.015159),
    'S1_raw': (0.447550, 0.139817),
    'S2_raw': (0.998605, 0.011334),
    'S3_raw': (-1.287651, 0.269379),
    'V_raw': (-2.148562, 0.552438),
}

# Inputs of mapped consolidation (Method S3)
COMPOSITE_INPUTS = ('D1_raw', 'D2_raw', 'D3_raw', 'D4_raw', 'D5_raw')
COMPOSITE_TRAINING_SD = 0.439442
COMPOSITE_TRAINING_MEDIAN = -0.162862

# Training mean / sample SD of the survival-model covariates (Method S5)
COVARIATE_TRAINING_CONSTANTS = {
    'Human_consol_mean': (2.090909, 0.832144),
    'tumor_size_cm_archived': (3.851515, 1.750450),
    'Age': (64.242424, 9.750835),
    'Eligible_slides': (6.075758, 2.957705),
}
VISUAL_TRAINING_MEDIAN = 2.0        # Human_high: reader mean above the training median
T_SIZE_CUTOFFS_CM = (2.0, 4.0)      # AJCC 8th: T1 <= 2 cm, T2 > 2-4 cm, T3 > 4 cm
TRAINING_COHORT_LABELS = ('train', 'training')
