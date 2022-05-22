import torch
import torch.nn as nn
import transforms as transforms
from prototype.prototype.model.vit.swin_transformer import swin_tiny
import os
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
dict = [0, 1, 12, 23, 34, 45, 56, 67, 78, 89, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = swin_tiny(drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.0, num_classes=100)
        self.load_params()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            normalize,
        ])

    def load_params(self):
        params = torch.load(os.path.join(os.path.dirname(__file__), 'ckpt_swinTiny.pth'), map_location=torch.device('cpu'))
        self.model.load_state_dict(params)

    def forward(self, x):
        device = x.device
        self.model = self.model.to(device)

        x_cvtcolor = x[:, [2, 1, 0], ...]
        x_trans = self.transform(x_cvtcolor)
        out = self.model(x_trans)
        out = out[:, dict]

        return F.softmax(out * 10000, dim=1)

