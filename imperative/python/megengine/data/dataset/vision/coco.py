# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Part of the following code in this file refs to maskrcnn-benchmark
# MIT License
#
# Copyright (c) 2018 Facebook
# ---------------------------------------------------------------------
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from .meta_vision import VisionDataset

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def has_valid_annotation(anno, order):
    # if it"s empty, there is no annotation
    if len(anno) == 0:
        return False
    if "boxes" in order or "boxes_category" in order:
        if "bbox" not in anno[0]:
            return False
    if "keypoints" in order:
        if "keypoints" not in anno[0]:
            return False
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) < min_keypoints_per_image:
            return False
    return True


class COCO(VisionDataset):
    r"""`MS COCO <http://cocodataset.org/#home>`_ Dataset."""

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
        "keypoints",
        # TODO: need to check
        # "polygons",
        "info",
    )

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, *, order=None
    ):
        super().__init__(root, order=order, supported_order=self.supported_order)

        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            # for saving memory
            if "license" in img:
                del img["license"]
            if "coco_url" in img:
                del img["coco_url"]
            if "date_captured" in img:
                del img["date_captured"]
            if "flickr_url" in img:
                del img["flickr_url"]
            self.imgs[img["id"]] = img

        self.img_to_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            # for saving memory
            if (
                "boxes" not in self.order
                and "boxes_category" not in self.order
                and "bbox" in ann
            ):
                del ann["bbox"]
            if "polygons" not in self.order and "segmentation" in ann:
                del ann["segmentation"]
            self.img_to_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.img_to_anns[img_id]
                # filter crowd annotations
                anno = [obj for obj in anno if obj["iscrowd"] == 0]
                anno = [
                    obj for obj in anno if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
                ]
                if has_valid_annotation(anno, order):
                    ids.append(img_id)
                    self.img_to_anns[img_id] = anno
                else:
                    del self.imgs[img_id]
                    del self.img_to_anns[img_id]
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(sorted(self.cats.keys()))
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]

        target = []
        for k in self.order:
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                path = os.path.join(self.root, file_name)
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = [
                    self.json_category_id_to_contiguous_id[c] for c in boxes_category
                ]
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == "keypoints":
                keypoints = [obj["keypoints"] for obj in anno]
                keypoints = np.array(keypoints, dtype=np.float32).reshape(
                    -1, len(self.keypoint_names), 3
                )
                target.append(keypoints)
            elif k == "polygons":
                polygons = [obj["segmentation"] for obj in anno]
                polygons = [
                    [np.array(p, dtype=np.float32).reshape(-1, 2) for p in ps]
                    for ps in polygons
                ]
                target.append(polygons)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["file_name"]]
                target.append(info)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    classes_originID = {
        "person": 1,
        "bicycle": 2,
        "car": 3,
        "motorcycle": 4,
        "airplane": 5,
        "bus": 6,
        "train": 7,
        "truck": 8,
        "boat": 9,
        "traffic light": 10,
        "fire hydrant": 11,
        "stop sign": 13,
        "parking meter": 14,
        "bench": 15,
        "bird": 16,
        "cat": 17,
        "dog": 18,
        "horse": 19,
        "sheep": 20,
        "cow": 21,
        "elephant": 22,
        "bear": 23,
        "zebra": 24,
        "giraffe": 25,
        "backpack": 27,
        "umbrella": 28,
        "handbag": 31,
        "tie": 32,
        "suitcase": 33,
        "frisbee": 34,
        "skis": 35,
        "snowboard": 36,
        "sports ball": 37,
        "kite": 38,
        "baseball bat": 39,
        "baseball glove": 40,
        "skateboard": 41,
        "surfboard": 42,
        "tennis racket": 43,
        "bottle": 44,
        "wine glass": 46,
        "cup": 47,
        "fork": 48,
        "knife": 49,
        "spoon": 50,
        "bowl": 51,
        "banana": 52,
        "apple": 53,
        "sandwich": 54,
        "orange": 55,
        "broccoli": 56,
        "carrot": 57,
        "hot dog": 58,
        "pizza": 59,
        "donut": 60,
        "cake": 61,
        "chair": 62,
        "couch": 63,
        "potted plant": 64,
        "bed": 65,
        "dining table": 67,
        "toilet": 70,
        "tv": 72,
        "laptop": 73,
        "mouse": 74,
        "remote": 75,
        "keyboard": 76,
        "cell phone": 77,
        "microwave": 78,
        "oven": 79,
        "toaster": 80,
        "sink": 81,
        "refrigerator": 82,
        "book": 84,
        "clock": 85,
        "vase": 86,
        "scissors": 87,
        "teddy bear": 88,
        "hair drier": 89,
        "toothbrush": 90,
    }

    keypoint_names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
