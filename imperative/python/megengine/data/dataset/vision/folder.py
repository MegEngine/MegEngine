# -*- coding: utf-8 -*-
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
# ---------------------------------------------------------------------
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .meta_vision import VisionDataset
from .utils import is_img


class ImageFolder(VisionDataset):
    r"""ImageFolder is a class for loading image data and labels from a organized folder.
    
    The folder is expected to be organized as followed: root/cls/xxx.img_ext
    
    Labels are indices of sorted classes in the root directory.

    Args:
        root: root directory of an image folder.
        loader: a function used to load image from path,
            if ``None``, default function that loads
            images with PIL will be called.
        check_valid_func: a function used to check if files in folder are
            expected image files, if ``None``, default function
            that checks file extensions will be called.
        class_name: if ``True``, return class name instead of class index.
    """

    def __init__(self, root: str, check_valid_func=None, class_name: bool = False):
        super().__init__(root, order=("image", "image_category"))

        self.root = root

        if check_valid_func is not None:
            self.check_valid = check_valid_func
        else:
            self.check_valid = is_img

        self.class_name = class_name

        self.class_dict = self.collect_class()
        self.samples = self.collect_samples()

    def collect_samples(self) -> List:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filename):
                    path = os.path.join(r, name)
                    if self.check_valid(path):
                        if self.class_name:
                            samples.append((path, key))
                        else:
                            samples.append((path, self.class_dict[key]))
        return samples

    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def __getitem__(self, index: int) -> Tuple:
        path, label = self.samples[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img, label

    def __len__(self):
        return len(self.samples)
