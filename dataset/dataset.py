import os

import cv2
import torch
from torchvision import transforms as TTR


class DIV2K(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()

        self.data_path = data_path
        self.images = os.listdir(data_path)

        self.preprocess = TTR.Compose(
            [
                TTR.ToTensor(),
                TTR.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.random_crop = TTR.RandomCrop(512)
        self.resize = TTR.Resize(256, TTR.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original = cv2.imread(f"{self.data_path}/{self.images[idx]}")
        original = self.preprocess(original)

        high_res = self.random_crop(original)
        low_res = self.resize(high_res)

        return low_res, high_res
