"""Evaluate the trained model on the held-out split (per-class IoU, recall, precision).

    python get_miou.py dataset/ --weights logs/best_epoch_weights.pth --out miou_out/

Predictions for ImageSets/Segmentation/val.txt (the held-out WSIs) are written to
{out}/detection-results/ with the same rule as whole-slide inference (softmax < 0.5 ->
background) and compared with Labels/; pixels labelled 255 are ignored. Results (tables and
bar plots) are saved in {out}. A mean IoU below 0.80 prompts annotation review and retraining.
"""
import argparse
import os

import numpy as np
from tqdm import tqdm

from utils.utils_metrics import compute_mIoU, show_results

MIN_MIOU = 0.80


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', default='miou_out')
    ap.add_argument('--split', default='val')
    ap.add_argument('--num-classes', type=int, default=3)
    ap.add_argument('--names', nargs='+', default=['background', 'SCC', 'ADC'])
    ap.add_argument('--backbone', default='xception')
    ap.add_argument('--mode', type=int, default=0, choices=[0, 1, 2],
                    help='0: predict and evaluate, 1: predict only, 2: evaluate existing predictions')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    image_ids = open(os.path.join(args.dataset, 'ImageSets/Segmentation/%s.txt' % args.split)).read().splitlines()
    gt_dir = os.path.join(args.dataset, 'Labels')
    pred_dir = os.path.join(args.out, 'detection-results')

    if args.mode in (0, 1):
        from PIL import Image
        from deeplab import DeeplabV3
        os.makedirs(pred_dir, exist_ok=True)
        deeplab = DeeplabV3(args.weights, num_classes=args.num_classes, backbone=args.backbone, cuda=not args.cpu)
        for image_id in tqdm(image_ids):
            image = Image.open(os.path.join(args.dataset, 'Images', image_id + '.tif'))
            deeplab.get_miou_png(image).save(os.path.join(pred_dir, image_id + '.png'))

    if args.mode in (0, 2):
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, args.num_classes, args.names)
        show_results(args.out, hist, IoUs, PA_Recall, Precision, args.names)
        if np.nanmean(IoUs) < MIN_MIOU:
            print('mean IoU %.3f < %.2f: review the annotations and retrain.' % (np.nanmean(IoUs), MIN_MIOU))


if __name__ == '__main__':
    main()
