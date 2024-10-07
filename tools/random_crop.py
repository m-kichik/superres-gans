"""
https://github.com/Kright/mySmallProjects/blob/master/2023/ml_experiments/superresolution/models/random_crop.py
"""

import random
from typing import List

import torch


class RandomShiftCrop:
    """
    crops rectangle with random shift and size to max_shift less.
    """
    def __init__(self, max_shift: int):
        self.max_shift = max_shift

    def __call__(self, *xx: torch.Tensor) -> List[torch.Tensor]:
        initial_size = xx[0].size()[2]

        rx = random.randrange(0, self.max_shift)
        ry = random.randrange(0, self.max_shift)

        xx = [x[:, :, ry: initial_size - self.max_shift + ry, rx: initial_size - self.max_shift + rx] for x in xx]
        return xx
