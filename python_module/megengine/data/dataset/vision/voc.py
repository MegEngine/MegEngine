# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ---------------------------------------------------------------------
# Part of the following code in this file refs to torchvision
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
# ---------------------------------------------------------------------
import collections.abc
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .meta_vision import VisionDataset


class PascalVOC(VisionDataset):
    r"""`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.
    """

    supported_order = (
        "image",
        # "boxes",
        # "boxes_category",
        "mask",
        "info",
    )

    def __init__(self, root, image_set, *, order=None):
        super().__init__(root, order=order, supported_order=self.supported_order)

        voc_root = self.root
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.")

        self.image_set = image_set
        image_dir = os.path.join(voc_root, "JPEGImages")

        # for segmentation
        if "aug" in image_set:
            mask_dir = os.path.join(voc_root, "SegmentationClass_aug")
        else:
            mask_dir = os.path.join(voc_root, "SegmentationClass")
        splitmask_dir = os.path.join(voc_root, "ImageSets/Segmentation")
        split_f = os.path.join(splitmask_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in self.file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in self.file_names]

        # TODO: for detection
        # splitdet_dir = os.path.join(voc_root, "ImageSets/Main")
        # split_f = os.path.join(splitdet_dir, image_set.rstrip("\n") + ".txt")
        # with open(os.path.join(split_f), "r") as f:
        #     self.file_names = [x.strip() for x in f.readlines()]
        # self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in self.file_names]

        # assert (len(self.images) == len(self.masks)) and (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        target = []
        for k in self.order:
            if k == "image":
                image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "mask":
                if "aug" in self.image_set:
                    mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
                else:
                    mask = np.array(cv2.imread(self.masks[index], cv2.IMREAD_COLOR))
                    mask = self._trans_mask(mask)
                mask = mask[:, :, np.newaxis]
                target.append(mask)
            # elif k == "boxes":
            #     boxes = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
            #     target.append(boxes)
            elif k == "info":
                if image is None:
                    image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
                info = [image.shape[0], image.shape[1], self.file_names[index]]
                target.append(info)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.images)

    def _trans_mask(self, mask):
        label = np.ones(mask.shape[:2]) * 255
        for i in range(len(self.class_colors)):
            b, g, r = self.class_colors[i]
            label[
                (mask[:, :, 0] == b) & (mask[:, :, 1] == g) & (mask[:, :, 2] == r)
            ] = i
        return label.astype("uint8")

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {
                node.tag: {
                    ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()
                }
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    class_names = (
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    class_colors = [
        [0, 0, 0],
        [0, 0, 128],
        [0, 128, 0],
        [0, 128, 128],
        [128, 0, 0],
        [128, 0, 128],
        [128, 128, 0],
        [128, 128, 128],
        [0, 0, 64],
        [0, 0, 192],
        [0, 128, 64],
        [0, 128, 192],
        [128, 0, 64],
        [128, 0, 192],
        [128, 128, 64],
        [128, 128, 192],
        [0, 64, 0],
        [0, 64, 128],
        [0, 192, 0],
        [0, 192, 128],
        [128, 64, 0],
    ]
