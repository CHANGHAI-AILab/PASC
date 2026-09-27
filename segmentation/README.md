# Tissue segmentation (whole-slide SCC / ADC label masks)

This folder holds the code that produces the whole-slide label masks read by
`scripts/01_slide_features.py`:

- `{slide}_labels.png` is a lossless uint8 PNG with 0 background, 1 SCC, 2 ADC and 255 for blocks without sampled tissue, at the mapping scale.
- `{slide}.json` records the level-0 size, the level-0 pixel size and the pixel size of the label map.

One three-class DeepLab-v3+ model (background, SCC, ADC) is used. The model code
(`nets/`, `utils/`, `deeplab.py`) is from
[bubbliiiing/deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch).
`nets/` and `utils/` are unchanged except for one NumPy alias in `utils_metrics.py` and the
exclusion of ignore pixels from the focal-loss mean and the Dice loss in `nets/deeplabv3_training.py`.
The shared label rules (threshold, closing, reduction) are in `labels.py`.

**Not included:** model weights, WSIs, annotations and tiles.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Tiles | `slice_wsi.py WSI_DIR TILES` | `TILES/{slide}/{slide} [x=X,y=Y,w=256,h=256].tif` and `slide.json` |
| 2. Prediction | `predict_tiles.py --weights W --tiles TILES --out PRED` | `PRED/{slide}/{tile}.npz`: float16 softmax, 3 x 256 x 256, and the tile tissue mask |
| 3. Whole-slide mask | `assemble_masks.py PRED MASKS` | `MASKS/{slide}_labels.png`, `MASKS/{slide}.json` |

```bash
pip install -r segmentation/requirements-seg.txt
cd segmentation
python slice_wsi.py /path/wsi tiles --backend openslide          # or --backend ndpisplit --ndpisplit /path/ndpisplit
python predict_tiles.py --weights logs/best_epoch_weights.pth --tiles tiles --out pred
python assemble_masks.py pred masks
```

### Details

- **Tiles:**
  - Slides are 40x scans. Tiles are read at level 0 (0.25 um/px), 256 x 256 pixels, with no resizing.
  - Tiles lie on a 256-pixel grid (stride 256). The last tile in each row and column is moved inward so that it ends at the image border, so it overlaps its neighbour.
  - Tiles with < 20% tissue (gray value < 220) are skipped.
  - A slide smaller than one tile, and any area outside the scanned region, is padded with white.
  - `slice_wsi.py` warns when a slide's objective power is not 40x.
- **Prediction:**
  - DeepLab-v3+ with an xception backbone, output stride 16, 256 x 256 input and 3 classes (background, SCC, ADC).
  - Inputs are scaled to [0, 1]. Tiles other than 256 x 256 are rejected rather than resized.
  - The softmax probabilities are saved as float16 NPZ files, together with the tile tissue mask (gray value < 220, the rule used for skipping tiles).
  - Label rule: pixels whose maximum softmax probability is < 0.5 are background; otherwise the argmax class is used.
- **Assembly:**
  - Labels are placed at their level-0 coordinates. Each coordinate is written once, with no averaging or voting:
    - Where tiles overlap, the grid tile keeps its label.
    - The shifted border tile writes only the pixels beyond the grid.
  - White padding is cropped. Skipped tiles are background and contain no tissue.
  - One 3 x 3 elliptical morphological closing is applied at level 0, before connected-component analysis. It fills only background pixels, and SCC takes precedence over ADC.
  - The level-0 map is then reduced 40x per axis to the mapping scale (0.25 x 40 = 10 um/px). Each 40 x 40 block takes, in order of priority:
    1. SCC, if any pixel in the block is SCC.
    2. ADC, if any pixel is ADC.
    3. Background, if any pixel is tissue.
    4. 255 (no sampled tissue) otherwise.
  - The non-255 pixels of the label map are the sampled tissue, the denominator of V (mapped tumor / sampled tissue).
  - Assembly runs on bands of 2,560 level-0 rows, with a 2-row halo for the closing. The result is identical to processing the whole slide at once.
  - The label PNG is the file used for quantification. The NPZ probabilities are kept for audit.

## Training

```bash
python json_to_dataset.py annotations/ dataset/ --classes _background_ SCC ADC --ignore ignore   # labelme 5.2.1 polygons
python voc_annotation.py dataset/                  # ImageSets/Segmentation/{train,val}.txt, 80/20 by WSI, seed 0
python train.py dataset/ logs/ --model-path model_data/deeplab_xception.pth
python get_miou.py dataset/ --weights logs/best_epoch_weights.pth --out miou/
```

Dataset layout:

- `Images/{name}.tif` holds the 256 x 256 level-0 patches.
- `Labels/{name}.png` holds the class indices: 0 background, 1 SCC, 2 ADC and 255 ignore (folds, severe artifact, uncertain necrosis).
- Pixels labelled 255 are excluded from the loss (focal / cross-entropy and Dice terms) and from the evaluation.

| Setting | Value (`train.py` defaults) |
|---|---|
| Backbone / input / output stride | xception / 256 x 256 / 16 |
| Initial weights | VOC-pretrained `deeplab_xception.pth` from the upstream releases |
| Epochs | backbone frozen for 50, 1000 in total |
| Optimizer | SGD, Nesterov, momentum 0.9, weight decay 1e-4 |
| Learning rate | 7e-3 x batch/16 (limits 5e-4 to 0.1), 3-epoch warm-up, cosine decay to 1% (held for the last 15 epochs) |
| Loss | focal loss with class weights (`--cls-weights`; alpha 0.5, gamma 2) + dice; `--no-focal` uses weighted cross-entropy instead of focal |
| Batch | 4 (`--batch-size`, `--freeze-batch-size`) |
| Preprocessing | pixel values / 255 |
| Augmentation | random scaling with aspect jitter, horizontal flip, placement on a gray canvas, Gaussian blur, rotation, HSV jitter (`utils/dataloader.py`) |
| Split | 80% of WSIs for development, 20% held out (`voc_annotation.py`, random.seed(0)) |

`get_miou.py` scores the held-out WSIs using the same label rule as whole-slide inference.
If the mean IoU is below 0.80, it prints a warning, which prompts annotation review and retraining.

Environment: see `requirements-seg.txt`.
