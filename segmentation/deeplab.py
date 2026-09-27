"""DeepLab-v3+ inference wrapper (from bubbliiiing/deeplabv3-plus-pytorch, trimmed).

One 3-class model (0 background, 1 SCC, 2 ADC), xception backbone, output stride 16.
Tiles are 256 x 256 level-0 patches and are fed to the network without resizing.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from labels import labels_from_probs
from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input


class DeeplabV3(object):
    _defaults = {
        "num_classes": 3,
        "backbone": "xception",
        "input_shape": [256, 256],
        "downsample_factor": 16,
        "cuda": True,
    }

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.cuda = self.cuda and torch.cuda.is_available()
        self.generate()

    def generate(self):
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone,
                           downsample_factor=self.downsample_factor, pretrained=False)
        device = torch.device('cuda' if self.cuda else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def predict_probs(self, image):
        """Softmax probabilities (C, H, W), float32, for one tile of the network input size."""
        image = np.array(cvtColor(image), np.float32)
        if list(image.shape[:2]) != list(self.input_shape):
            raise ValueError('tile is %s, expected %s (no resizing)' % (image.shape[:2], self.input_shape))
        image_data = np.expand_dims(np.transpose(preprocess_input(image), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = F.softmax(self.net(images)[0], dim=0)
        return pr.cpu().numpy()

    def get_miou_png(self, image):
        """Label prediction (softmax < 0.5 -> background) for mIoU evaluation."""
        from PIL import Image
        return Image.fromarray(labels_from_probs(self.predict_probs(image)))
