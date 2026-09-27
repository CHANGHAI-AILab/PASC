# Data dictionary

## Slide manifest (`01_slide_features.py`)

Only image-quality-passed slides enter this feature-extraction manifest. The complete
study inventory needed for a slide-level audit would also include excluded slides
and their quality or coverage reasons; it is not the same as this processing input.

| Field | Definition |
|---|---|
| `patient_id` | Study ID (deidentified) |
| `slide_id` | Slide ID; the label map is `MASK_DIR/{slide_id}_labels.png` (0 background tissue, 1 SCC, 2 ADC, 255 no sampled tissue) with its pixel size in `MASK_DIR/{slide_id}.json` |

## Slide features (`slide_features.csv`)

| Field | Definition |
|---|---|
| `mpp_x`, `mpp_y` | Pixel size of the label map (um) |
| `N_px` | Sampled tissue pixels (label-map pixels other than 255) |
| `A_SCC_px`, `A_ADC_px`, `T_px` | Squamous, adenocarcinoma and total tumor pixels |
| `T_um2` | Tumor area, `T_px` x mpp_x x mpp_y |
| `eligible` | Tumor-coverage flag after image-quality review: `T_um2` >= 1.2 x 10^6 um^2; exactly >= 12,000 SCC + ADC pixels at 10 um/px in both axes |
| `CCCount`, `LargestRegion_px`, `DCR` | SCC regions >= 10 px: count, largest size, largest / all SCC pixels |
| `LesionCount` | SCC regions >= 5% of all SCC pixels |
| `SecondClusterFrac` | Second-largest SCC region / all SCC pixels |
| `Multifocal` | `LesionCount` >= 2 or `SecondClusterFrac` >= 0.05 |
| `ShapeFactor`, `PerimeterAreaRatio` | P^2 / (4 pi A) and P / sqrt(A) of SCC |
| `InterfaceFrac`, `InterfaceFracSym` | SCC-ADC contact fraction of SCC edges / of both edges |
| `FrontFrac`, `FrontDistance_px` | SCC fraction within 500 px of the tumor boundary; mean SCC distance to it |

## Cohort file (optional, `02_patient_features.py --cohort`)

| Field | Definition |
|---|---|
| `patient_id` | Study ID |
| `cohort` | `train` / `validation` / `external` |

## Patient features (`patient_features.csv`)

| Field | Definition |
|---|---|
| `n_eligible_slides` | Eligible slides contributing to the descriptors |
| `SCC_in_tumor_pt` | Full-precision mapped squamous fraction |
| `C_raw`, `D1_raw`..`D5_raw`, `B_raw`, `S1_raw`..`S3_raw`, `V_raw` | Table S3 descriptors (docs/implementation.md) |
| `DCR_wavg` | Tumor-area-weighted DCR |
| `Consolidation_z` | Mapped consolidation |
| `Consolidation_high` | `Consolidation_z` >= training median (-0.162862) |

Visual consolidation (`composite.visual_consolidation`) takes `Human_consol_P1`..`P3` (integer 1-4)
and returns `Human_consol_mean` (unrounded) and `Visual_rounded` (display only).

## Patient-level analysis table (`03_analysis_table.py`)

One row per patient. Input columns:

| Field | Definition |
|---|---|
| `Slide_Number` | Case identifier linking to `patient_id` of `patient_features.csv` (`--id`) |
| `Cohort_label` | `Training` / `Validation` / `External test` |
| `Human_consol_P1`..`P3` | Original reader scores, integer 1-4 |
| `SCC_proportion_PATHOLOGIST` | Pathologist-estimated squamous %, resection level |
| `SCC_in_tumor_pt`, `D1`..`D5` | Mapped descriptors (used when `--features` is not given) |
| `Age`, `tumor_size_cm_archived` | Years; maximum gross diameter in cm |
| `archived_pT_exact_for_audit` | Archived T category; only T4 is carried into the final category |
| `pT_ge3_original` | Historical binary pT field (may be missing) |
| `Total_slides` | Retrievable tumor-bearing slides before quality/coverage exclusion; summed across the final 148 patients: 903 |
| `Eligible_slides` | Slides passing image-quality review and the tumor-area criterion of >= 1.2 x 10^6 um^2 (>= 12,000 SCC + ADC pixels at 10 um/px in both axes); summed across the final 148 patients: 880 |

Derived columns (all constants fixed in the training cohort):

| Field | Definition |
|---|---|
| `Human_consol_mean`, `Visual_rounded` | Unrounded three-reader mean; nearest integer, display only |
| `Human_high` | Reader mean > training median (2.0) |
| `Reader12_mean`, `Reader13_mean`, `Reader23_mean` | Two-reader means |
| `SCC_proportion_AI_RAW_fraction`, `_pct` | Mapped squamous fraction; percent rounded to 0.1 |
| `D1`..`D5`, `D1_z`..`D5_z` | Composite inputs and their training z-scores |
| `Consolidation_z`, `Consol_high` | Mapped consolidation; >= training median (-0.162862) |
| `Composition_z` | z-score of `C_raw` |
| `pT_size_harmonized_exact` | T1-T3 from diameter: <= 2, > 2-4, > 4 cm |
| `pT_harmonized_final_exact`, `pT_harmonized_T3T4` | Size T with archived T4 retained; T3-T4 indicator |
| `pT_ge3`, `archived_pT4_flag_for_review`, `pT_ge3_size_plus_archived_T4_sensitivity` | Size T >= 3; archived T4; either |
| `pT_changed_or_filled` | `pT_ge3_original` missing or different from `pT_ge3` |
| `visual_z`, `size_z`, `Age_z`, `eligible_z` | (x - training mean) / training SD, Method S5 |
| `path10` | `SCC_proportion_PATHOLOGIST` / 10 |
