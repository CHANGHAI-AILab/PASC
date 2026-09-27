# Implementation notes

Definitions used by `scripts/01_slide_features.py` and `scripts/02_patient_features.py`
(Supplementary Methods S2-S3, Table S3). Constants are in `src/pasc_consol/constants.py`.

## Masks

- Each slide has one label map, `{slide}_labels.png` (lossless uint8 PNG: 0 background, 1 SCC,
  2 ADC, 255 ignore), and `{slide}.json` with its pixel size. Both are written by
  `segmentation/assemble_masks.py` (see `segmentation/README.md`).
  - One three-class DeepLab-v3+ model (xception backbone, 256 x 256 input) predicts 256 x 256
    level-0 tiles of 40x scans (0.25 um/px). Pixels whose maximum softmax probability is < 0.5
    are background; otherwise the argmax class is used.
  - Tiles are stitched at level 0 with every coordinate written once. One 3 x 3 elliptical
    closing is applied before connected-component analysis.
  - The level-0 map is reduced 40x per axis to the mapping scale (10 um/px); a block is SCC if
    it contains SCC, otherwise ADC if it contains ADC, otherwise background if it contains
    tissue (gray value < 220 in a kept tile), otherwise 255. Non-ignore pixels (`N_px`) are
    therefore the sampled tissue.
- `masks.load_slide` reads the PNG and the pixel size (mpp_x, mpp_y) from the JSON.

## Slide descriptors (`descriptors.slide_descriptors`)

Lengths and areas are in mapping-scale pixels, except `T_um2`. Regions are 8-connected.

| Field | Definition |
|---|---|
| `CCCount` | Number of SCC regions >= 10 px |
| `DCR` | Largest region >= 10 px / all SCC pixels |
| `LesionCount` | Number of SCC regions (any size) >= 5% of all SCC pixels |
| `SecondClusterFrac` | Second-largest SCC region (any size) / all SCC pixels; 0 with one region |
| `Multifocal` | `LesionCount` >= 2 or `SecondClusterFrac` >= 0.05 |
| `ShapeFactor` | P^2 / (4 pi A_SCC). P is the external-contour perimeter after one 3 x 3 rectangular closing (closing used only here) |
| `InterfaceFracSym` | (SCC Canny edge pixels touching dilated ADC + ADC edge pixels touching dilated SCC) / all edge pixels. The dilation kernel is a 3 x 3 ellipse |
| `FrontFrac` | Fraction of SCC pixels within 500 px of the tumor (SCC + ADC) boundary (Euclidean distance transform) |
| `FrontDistance_px` | Mean distance of SCC pixels to the tumor boundary |
| `T_um2` | `T_px` (SCC + ADC pixels) x mpp_x x mpp_y |
| `eligible` | `T_um2` >= 1.2 x 10^6 um^2, exactly >= 12,000 SCC + ADC pixels at 10 um/px in both axes; equality is included |

A slide without SCC has 0 for all SCC descriptors. It still counts, with DCR = 0, in the weighted D1.

### Slide eligibility and reported counts

Image-quality review precedes feature extraction. The input manifest must contain only
slides that passed that review; `eligible` tests tumor coverage, not image quality.
Tumor area is `(A_SCC_px + A_ADC_px) * mpp_x * mpp_y` on the mapping-scale map,
before the 10-pixel component filter. Background and label 255 do not count as tumor.
With `mpp_x = mpp_y = 10`, 11,999 tumor pixels fail, while 12,000 and 12,001 pass.
For other calibrated pixel sizes, the physical-area threshold remains 1.2 x 10^6 um^2;
12,000 is the equivalent pixel threshold only at 10 um/px in both axes.

The final patient-level study table records the following counts (`Total_slides` and
`Eligible_slides`). The overall 903/880 totals are also reported in the manuscript
and Supplementary Method S3:

| Cohort | Patients | Retrievable tumor-bearing slides | Eligible slides | Excluded for quality or coverage |
|---|---:|---:|---:|---:|
| Training | 66 | 410 | 401 | 9 |
| Validation | 36 | 211 | 205 | 6 |
| External test | 46 | 282 | 274 | 8 |
| Total | 148 | 903 | 880 | 23 |

These totals are sums of the final patient table, not a substitute for a slide-level
eligibility audit. The 23 exclusions combine image-quality and tumor-coverage review;
they must not all be described as area failures. A full audit also requires the complete
slide manifest, patient linkage, quality-review decisions and calibrated tumor areas.
Neither the complete manifest nor patient data are included in this repository.
The scripts compute counts from their inputs and never force them to 903 or 880.

## Patient descriptors (`descriptors.patient_descriptors`)

These are computed over eligible slides, with slide weight `T_px`.

| Field | Definition |
|---|---|
| `C_raw` | logit(sum A_SCC / sum T) |
| `D1_raw` | 1 - weighted mean DCR |
| `D2_raw` | any slide multifocal |
| `D3_raw` | log(1 + max CCCount) |
| `D4_raw` | log(1 + max LesionCount) |
| `D5_raw` | max SecondClusterFrac |
| `B_raw` | log(weighted mean ShapeFactor) |
| `S1_raw` | weighted mean InterfaceFracSym |
| `S2_raw` | weighted mean FrontFrac |
| `S3_raw` | -log(1 + weighted mean FrontDistance_px) |
| `V_raw` | logit(weighted mean of T_px / N_px): mapped tumor / sampled tissue |

`Consolidation_z` = -mean(z(D1..D5)) / 0.439442, where z uses the Table S3 training constants.

## Analysis-table fields (`analysis_table.derive`)

- Visual consolidation:
  - `Human_consol_mean` is the unrounded mean of the three reader scores.
  - `Human_high` is a mean above the training median of 2.0.
- T category:
  - T1 <= 2 cm, T2 > 2-4 cm, T3 > 4 cm, from maximum gross diameter.
  - An archived pT4 is retained.
- Model covariates (Method S5):
  - Each is (x - training mean) / training SD: reader mean 2.090909 / 0.832144, size 3.851515 / 1.750450 cm, age 64.242424 / 9.750835, eligible slides 6.075758 / 2.957705.
  - `path10` = pathologist % / 10.
