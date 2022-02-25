# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Part of the following code in this file refs to torchvision
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
# ---------------------------------------------------------------------
import json
import os

import cv2
import numpy as np

from .meta_vision import VisionDataset


class Cityscapes(VisionDataset):
    r"""`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset."""

    supported_order = (
        "image",
        "mask",
        "info",
    )

    def __init__(self, root, image_set, mode, *, order=None):
        super().__init__(root, order=order, supported_order=self.supported_order)

        city_root = self.root
        if not os.path.isdir(city_root):
            raise RuntimeError("Dataset not found or corrupted.")

        self.mode = mode
        self.images_dir = os.path.join(city_root, "leftImg8bit", image_set)
        self.masks_dir = os.path.join(city_root, self.mode, image_set)
        self.images, self.masks = [], []
        # self.target_type = ["instance", "semantic", "polygon", "color"]

        # for semantic segmentation
        if mode == "gtFine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            mask_dir = os.path.join(self.masks_dir, city)
            for file_name in os.listdir(img_dir):
                mask_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0],
                    self._get_target_suffix(self.mode, "semantic"),
                )
                self.images.append(os.path.join(img_dir, file_name))
                self.masks.append(os.path.join(mask_dir, mask_name))

    def __getitem__(self, index):
        target = []
        for k in self.order:
            if k == "image":
                image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "mask":
                mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
                mask = self._trans_mask(mask)
                mask = mask[:, :, np.newaxis]
                target.append(mask)
            elif k == "info":
                if image is None:
                    image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
                info = [image.shape[0], image.shape[1], self.images[index]]
                target.append(info)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.images)

    def _trans_mask(self, mask):
        trans_labels = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        label = np.ones(mask.shape) * 255
        for i, tl in enumerate(trans_labels):
            label[mask == tl] = i
        return label.astype(np.uint8)

    def _get_target_suffix(self, mode, target_type):
        if target_type == "instance":
            return "{}_instanceIds.png".format(mode)
        elif target_type == "semantic":
            return "{}_labelIds.png".format(mode)
        elif target_type == "color":
            return "{}_color.png".format(mode)
        else:
            return "{}_polygons.json".format(mode)

    def _load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

    class_names = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
