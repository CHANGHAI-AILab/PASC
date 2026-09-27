# Squamous-component spatial consolidation in pancreatic adenosquamous carcinoma

Feature-extraction code for the Modern Pathology manuscript on visual and mapped spatial
consolidation of the squamous component in pancreatic adenosquamous carcinoma (PASC).
It turns whole-slide tissue masks into the slide- and patient-level descriptors of Table S3
and the mapped consolidation score (Supplementary Methods S2-S3).

The repository contains code and fixed training constants only: the tissue segmentation pipeline
that produces the masks (`segmentation/`) and the feature code. It contains no patient data,
whole-slide images, masks, annotations or model weights. Statistical analyses (agreement, Cox and
discrimination models) are not included.

## Pipeline

| Step | Code | Output |
|---|---|---|
| WSI tiles, DeepLab-v3+ prediction, whole-slide label masks | `segmentation/` (see its README) | `{slide}_labels.png` (0 background, 1 SCC, 2 ADC, 255 no sampled tissue) and `{slide}.json` (pixel size) |
| Slide descriptors | `scripts/01_slide_features.py` | `slide_features.csv` |
| Patient descriptors, mapped consolidation | `scripts/02_patient_features.py` | `patient_features.csv` (C_raw, D1-D5, B, S1-S3, V, Consolidation_z) |
| Derived fields of the analysis table | `scripts/03_analysis_table.py` | visual consolidation, D1-D5 z, Consolidation_z, T category, model covariates (Methods S3-S5) |
| Display maps (Fig. 1b, Fig. 3, Fig. S1a) | `scripts/00_display_maps.py` | red SCC / blue ADC PNG |

Definitions are listed in `docs/implementation.md`; constants are in `src/pasc_consol/constants.py`.
The main points:

- Masks:
  - One three-class model (background, SCC, ADC) predicts 256 x 256 level-0 tiles of 40x scans.
  - Pixels whose maximum softmax probability is < 0.5 are background.
  - One 3 x 3 elliptical closing is applied before connected-component analysis.
  - Descriptors are computed on the mapping-scale label map (level 0 reduced 40x per axis, 10 um/px at 0.25 um/px).
  - Regions are 8-connected.
- Slide descriptors:
  - `CCCount` and DCR use regions >= 10 px.
  - A significant region has >= 5% of all SCC pixels.
  - After image-quality review, a slide meets the tumor-coverage criterion when its mapped SCC + ADC area is >= 1.2 x 10^6 um^2 (tumor pixel count x pixel area from the mask JSON). At 10 um/px in both axes, this is exactly >= 12,000 tumor pixels, including equality.
- Patient descriptors, computed over eligible slides:
  - Proportions are weighted by tumor area; counts use the slide maximum.
  - `C_raw` = logit(SCC / tumor).
  - `D1_raw` = 1 - weighted DCR.
  - `D2_raw` = any multifocal slide.
  - `D3_raw` = log(1 + max `CCCount`).
  - `D4_raw` = log(1 + max significant-region count).
  - `D5_raw` = max second-largest-region fraction.
- `Consolidation_z` = -mean(z(D1..D5)) / 0.439442.
  - z uses the training constants in Table S3; the training median is -0.162862.
  - Constants are applied unchanged to the validation and external cohorts.
- Visual consolidation `Human_consol_mean` is the unrounded mean of three reader scores (1-4). `Visual_rounded` is for display only.
- Analysis-table fields (`src/pasc_consol/analysis_table.py`):
  - T1-T3 come from maximum gross diameter (<= 2, > 2-4, > 4 cm). An archived pT4 is kept.
  - `visual_z`, `size_z`, `Age_z` and `eligible_z` use the training mean and SD from Method S5.
  - `path10` = pathologist squamous % / 10.

## Usage

The final study table records 903 retrievable tumor-bearing slides from 148 patients,
of which 880 passed image-quality and tumor-coverage review; 23 were excluded for quality
or coverage. These are study counts, not hard-coded outputs. See
`docs/implementation.md` for the cohort totals and the distinction between image-quality
review and the area check performed by the code. The feature manifest must contain only
slides that have passed image-quality review.

```bash
pip install -r requirements.txt

# manifest columns: patient_id, slide_id; MASK_DIR is the output of segmentation/assemble_masks.py
python scripts/01_slide_features.py manifest.csv results/slide_features.csv --masks MASK_DIR
# optional cohort columns: patient_id, cohort (train / validation / external)
python scripts/02_patient_features.py results/slide_features.csv results/patient_features.csv --cohort cohort.csv

# derived fields of the patient-level analysis table; --check compares them with the table
# and recomputes the training constants in the Training rows
python scripts/03_analysis_table.py analysis_table.xlsx results/analysis_table.csv \n    --features results/patient_features.csv --check

pytest tests/
```

Unit tests check the descriptor definitions on synthetic masks, the composite constants, and the
tiling, thresholding, closing and mask assembly of the segmentation pipeline (torch is not needed).

## Environment

Feature code: Python >= 3.10 with numpy, scipy, pandas, opencv-python, Pillow and openpyxl.
Segmentation (Ubuntu 20.04, Python 3.6.13, PyTorch 1.10.2, OpenSlide): see
`segmentation/requirements-seg.txt`.
