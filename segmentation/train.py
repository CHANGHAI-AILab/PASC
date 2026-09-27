"""Train the DeepLab-v3+ tissue segmentation model (3 classes: background, SCC, ADC).

    python voc_annotation.py dataset/
    python train.py dataset/ logs/ --model-path model_data/deeplab_xception.pth

Dataset layout (see README): dataset/Images/*.tif (256 x 256 level-0 patches),
dataset/Labels/*.png (0 background, 1 SCC, 2 ADC, 255 ignore),
dataset/ImageSets/Segmentation/{train,val}.txt (WSI-level split). Pixels labelled 255 are
excluded from the loss (focal / cross-entropy and Dice terms).

Defaults: xception, 256 x 256 input, output stride 16, backbone frozen for 50 epochs then
1000 epochs in total, SGD (Nesterov, momentum 0.9, weight decay 1e-4), initial lr 7e-3
scaled by batch / 16 inside the limits of the upstream code, 3-epoch warm-up, cosine decay to
1% (held for the last 15 epochs), batch 4, loss = class-weighted focal loss (alpha 0.5,
gamma 2; --no-focal: weighted cross-entropy instead) + dice loss. Inputs are scaled to
[0, 1]; augmentation (utils/dataloader.py) is random scaling with aspect jitter, horizontal
flip, placement on a gray canvas, Gaussian blur, rotation up to 10 degrees and HSV jitter.

Weights are saved every 5 epochs and as best_epoch_weights.pth / last_epoch_weights.pth.
Training code is from bubbliiiing/deeplabv3-plus-pytorch (single-node DataParallel path).
"""
import argparse
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch


def fit_lr(batch_size, init_lr, min_lr, optimizer_type, backbone):
    """Scale lr by batch / 16 within the limits of the upstream code."""
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    if backbone == 'xception':
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    return init_lr_fit, min_lr_fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='folder with Images/, Labels/, ImageSets/Segmentation/')
    ap.add_argument('save_dir')
    ap.add_argument('--num-classes', type=int, default=3)
    ap.add_argument('--backbone', default='xception', choices=['xception', 'mobilenet'])
    ap.add_argument('--pretrained', action='store_true',
                    help='download ImageNet backbone weights (ignored for keys loaded from --model-path)')
    ap.add_argument('--model-path', default='', help='initial weights, e.g. the VOC deeplab_xception.pth')
    ap.add_argument('--cls-weights', default='1,1,1', help='cross-entropy class weights')
    ap.add_argument('--miou-out', default=None, help='temporary folder for validation predictions')
    ap.add_argument('--input-size', type=int, default=256)
    ap.add_argument('--downsample-factor', type=int, default=16)
    ap.add_argument('--freeze-epoch', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--freeze-batch-size', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=7e-3)
    ap.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    ap.add_argument('--lr-decay', default='cos', choices=['cos', 'step'])
    ap.add_argument('--save-period', type=int, default=5)
    ap.add_argument('--eval-period', type=int, default=5)
    ap.add_argument('--no-dice', action='store_true')
    ap.add_argument('--no-focal', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    cuda = not args.cpu and torch.cuda.is_available()
    input_shape = [args.input_size, args.input_size]
    init_lr = args.lr
    min_lr = init_lr * 0.01
    momentum = 0.9
    weight_decay = 1e-4
    dice_loss = not args.no_dice
    focal_loss = not args.no_focal
    cls_weights = np.array([float(v) for v in args.cls_weights.split(',')], np.float32)
    if len(cls_weights) != args.num_classes:
        raise ValueError('--cls-weights needs %d values' % args.num_classes)
    miou_out = args.miou_out or os.path.join(args.save_dir, 'miou_out')
    device = torch.device('cuda' if cuda else 'cpu')
    local_rank = 0
    os.makedirs(args.save_dir, exist_ok=True)

    if args.pretrained:
        download_weights(args.backbone)
    model = DeepLab(num_classes=args.num_classes, backbone=args.backbone,
                    downsample_factor=args.downsample_factor, pretrained=args.pretrained)
    if not args.pretrained:
        weights_init(model)
    if args.model_path:
        # load every key whose shape matches; the classification head is expected to differ
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        temp_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print('Loaded %d of %d keys from %s' % (len(temp_dict), len(pretrained_dict), args.model_path))

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, 'loss_' + time_str)
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if args.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(os.path.join(args.dataset, 'ImageSets/Segmentation/train.txt')) as f:
        train_lines = f.readlines()
    with open(os.path.join(args.dataset, 'ImageSets/Segmentation/val.txt')) as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    show_config(num_classes=args.num_classes, backbone=args.backbone, model_path=args.model_path,
                input_shape=input_shape, Freeze_Epoch=args.freeze_epoch, UnFreeze_Epoch=args.epochs,
                Freeze_batch_size=args.freeze_batch_size, Unfreeze_batch_size=args.batch_size,
                Init_lr=init_lr, Min_lr=min_lr, optimizer_type=args.optimizer, momentum=momentum,
                lr_decay_type=args.lr_decay, cls_weights=cls_weights.tolist(), dice_loss=dice_loss,
                focal_loss=focal_loss, save_dir=args.save_dir, num_workers=args.workers,
                num_train=num_train, num_val=num_val)

    freeze = args.freeze_epoch > 0
    unfreeze_flag = False
    if freeze:
        for param in model.backbone.parameters():
            param.requires_grad = False
    batch_size = args.freeze_batch_size if freeze else args.batch_size

    init_lr_fit, min_lr_fit = fit_lr(batch_size, init_lr, min_lr, args.optimizer, args.backbone)
    optimizer = {
        'adam': optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay),
    }[args.optimizer]
    lr_scheduler_func = get_lr_scheduler(args.lr_decay, init_lr_fit, min_lr_fit, args.epochs)

    train_dataset = DeeplabDataset(train_lines, input_shape, args.num_classes, True, args.dataset)
    val_dataset = DeeplabDataset(val_lines, input_shape, args.num_classes, False, args.dataset)

    def loaders(bs):
        epoch_step, epoch_step_val = num_train // bs, num_val // bs
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('Dataset too small for batch size %d.' % bs)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=bs, num_workers=args.workers, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=bs, num_workers=args.workers, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)
        return gen, gen_val, epoch_step, epoch_step_val

    gen, gen_val, epoch_step, epoch_step_val = loaders(batch_size)
    eval_callback = EvalCallback(model, input_shape, args.num_classes, val_lines, args.dataset, log_dir, cuda,
                                 miou_out, eval_flag=True, period=args.eval_period)

    for epoch in range(args.epochs):
        if epoch >= args.freeze_epoch and not unfreeze_flag and freeze:
            batch_size = args.batch_size
            init_lr_fit, min_lr_fit = fit_lr(batch_size, init_lr, min_lr, args.optimizer, args.backbone)
            lr_scheduler_func = get_lr_scheduler(args.lr_decay, init_lr_fit, min_lr_fit, args.epochs)
            for param in model.backbone.parameters():
                param.requires_grad = True
            gen, gen_val, epoch_step, epoch_step_val = loaders(batch_size)
            unfreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                      gen, gen_val, args.epochs, cuda, dice_loss, focal_loss, cls_weights, args.num_classes,
                      args.fp16, scaler, args.save_period, args.save_dir, local_rank)

    loss_history.writer.close()


if __name__ == '__main__':
    main()
