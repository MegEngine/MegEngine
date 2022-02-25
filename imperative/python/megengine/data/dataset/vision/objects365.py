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


class Objects365(VisionDataset):
    r"""`Objects365 <https://www.objects365.org/overview.html>`_ Dataset."""

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
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
                if len(anno) > 0:
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
        "sneakers",
        "chair",
        "hat",
        "lamp",
        "bottle",
        "cabinet/shelf",
        "cup",
        "car",
        "glasses",
        "picture/frame",
        "desk",
        "handbag",
        "street lights",
        "book",
        "plate",
        "helmet",
        "leather shoes",
        "pillow",
        "glove",
        "potted plant",
        "bracelet",
        "flower",
        "tv",
        "storage box",
        "vase",
        "bench",
        "wine glass",
        "boots",
        "bowl",
        "dining table",
        "umbrella",
        "boat",
        "flag",
        "speaker",
        "trash bin/can",
        "stool",
        "backpack",
        "couch",
        "belt",
        "carpet",
        "basket",
        "towel/napkin",
        "slippers",
        "barrel/bucket",
        "coffee table",
        "suv",
        "toy",
        "tie",
        "bed",
        "traffic light",
        "pen/pencil",
        "microphone",
        "sandals",
        "canned",
        "necklace",
        "mirror",
        "faucet",
        "bicycle",
        "bread",
        "high heels",
        "ring",
        "van",
        "watch",
        "sink",
        "horse",
        "fish",
        "apple",
        "camera",
        "candle",
        "teddy bear",
        "cake",
        "motorcycle",
        "wild bird",
        "laptop",
        "knife",
        "traffic sign",
        "cell phone",
        "paddle",
        "truck",
        "cow",
        "power outlet",
        "clock",
        "drum",
        "fork",
        "bus",
        "hanger",
        "nightstand",
        "pot/pan",
        "sheep",
        "guitar",
        "traffic cone",
        "tea pot",
        "keyboard",
        "tripod",
        "hockey",
        "fan",
        "dog",
        "spoon",
        "blackboard/whiteboard",
        "balloon",
        "air conditioner",
        "cymbal",
        "mouse",
        "telephone",
        "pickup truck",
        "orange",
        "banana",
        "airplane",
        "luggage",
        "skis",
        "soccer",
        "trolley",
        "oven",
        "remote",
        "baseball glove",
        "paper towel",
        "refrigerator",
        "train",
        "tomato",
        "machinery vehicle",
        "tent",
        "shampoo/shower gel",
        "head phone",
        "lantern",
        "donut",
        "cleaning products",
        "sailboat",
        "tangerine",
        "pizza",
        "kite",
        "computer box",
        "elephant",
        "toiletries",
        "gas stove",
        "broccoli",
        "toilet",
        "stroller",
        "shovel",
        "baseball bat",
        "microwave",
        "skateboard",
        "surfboard",
        "surveillance camera",
        "gun",
        "life saver",
        "cat",
        "lemon",
        "liquid soap",
        "zebra",
        "duck",
        "sports car",
        "giraffe",
        "pumpkin",
        "piano",
        "stop sign",
        "radiator",
        "converter",
        "tissue ",
        "carrot",
        "washing machine",
        "vent",
        "cookies",
        "cutting/chopping board",
        "tennis racket",
        "candy",
        "skating and skiing shoes",
        "scissors",
        "folder",
        "baseball",
        "strawberry",
        "bow tie",
        "pigeon",
        "pepper",
        "coffee machine",
        "bathtub",
        "snowboard",
        "suitcase",
        "grapes",
        "ladder",
        "pear",
        "american football",
        "basketball",
        "potato",
        "paint brush",
        "printer",
        "billiards",
        "fire hydrant",
        "goose",
        "projector",
        "sausage",
        "fire extinguisher",
        "extension cord",
        "facial mask",
        "tennis ball",
        "chopsticks",
        "electronic stove and gas stove",
        "pie",
        "frisbee",
        "kettle",
        "hamburger",
        "golf club",
        "cucumber",
        "clutch",
        "blender",
        "tong",
        "slide",
        "hot dog",
        "toothbrush",
        "facial cleanser",
        "mango",
        "deer",
        "egg",
        "violin",
        "marker",
        "ship",
        "chicken",
        "onion",
        "ice cream",
        "tape",
        "wheelchair",
        "plum",
        "bar soap",
        "scale",
        "watermelon",
        "cabbage",
        "router/modem",
        "golf ball",
        "pine apple",
        "crane",
        "fire truck",
        "peach",
        "cello",
        "notepaper",
        "tricycle",
        "toaster",
        "helicopter",
        "green beans",
        "brush",
        "carriage",
        "cigar",
        "earphone",
        "penguin",
        "hurdle",
        "swing",
        "radio",
        "CD",
        "parking meter",
        "swan",
        "garlic",
        "french fries",
        "horn",
        "avocado",
        "saxophone",
        "trumpet",
        "sandwich",
        "cue",
        "kiwi fruit",
        "bear",
        "fishing rod",
        "cherry",
        "tablet",
        "green vegetables",
        "nuts",
        "corn",
        "key",
        "screwdriver",
        "globe",
        "broom",
        "pliers",
        "volleyball",
        "hammer",
        "eggplant",
        "trophy",
        "dates",
        "board eraser",
        "rice",
        "tape measure/ruler",
        "dumbbell",
        "hamimelon",
        "stapler",
        "camel",
        "lettuce",
        "goldfish",
        "meat balls",
        "medal",
        "toothpaste",
        "antelope",
        "shrimp",
        "rickshaw",
        "trombone",
        "pomegranate",
        "coconut",
        "jellyfish",
        "mushroom",
        "calculator",
        "treadmill",
        "butterfly",
        "egg tart",
        "cheese",
        "pig",
        "pomelo",
        "race car",
        "rice cooker",
        "tuba",
        "crosswalk sign",
        "papaya",
        "hair drier",
        "green onion",
        "chips",
        "dolphin",
        "sushi",
        "urinal",
        "donkey",
        "electric drill",
        "spring rolls",
        "tortoise/turtle",
        "parrot",
        "flute",
        "measuring cup",
        "shark",
        "steak",
        "poker card",
        "binoculars",
        "llama",
        "radish",
        "noodles",
        "yak",
        "mop",
        "crab",
        "microscope",
        "barbell",
        "bread/bun",
        "baozi",
        "lion",
        "red cabbage",
        "polar bear",
        "lighter",
        "seal",
        "mangosteen",
        "comb",
        "eraser",
        "pitaya",
        "scallop",
        "pencil case",
        "saw",
        "table tennis paddle",
        "okra",
        "starfish",
        "eagle",
        "monkey",
        "durian",
        "game board",
        "rabbit",
        "french horn",
        "ambulance",
        "asparagus",
        "hoverboard",
        "pasta",
        "target",
        "hotair balloon",
        "chainsaw",
        "lobster",
        "iron",
        "flashlight",
    )
