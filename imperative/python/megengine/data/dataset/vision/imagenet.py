# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
# ---------------------------------------------------------------------
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------
import os
import shutil

from tqdm import tqdm

from ....distributed.group import is_distributed
from ....logger import get_logger
from ....serialization import load, save
from .folder import ImageFolder
from .utils import _default_dataset_root, calculate_md5, untar, untargz

logger = get_logger(__name__)


class ImageNet(ImageFolder):
    r"""Load ImageNet from raw files or folder. Expected folder looks like:
    
    .. code-block:: shell
    
        ${root}/
        |       [REQUIRED TAR FILES]
        |-  ILSVRC2012_img_train.tar
        |-  ILSVRC2012_img_val.tar
        |-  ILSVRC2012_devkit_t12.tar.gz
        |       [OPTIONAL IMAGE FOLDERS]
        |-  train/cls/xxx.${img_ext}
        |-  val/cls/xxx.${img_ext}
        |-  ILSVRC2012_devkit_t12/data/meta.mat
        |-  ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
    
    If the image folders don't exist, raw tar files are required to get extracted and processed.

        * if ``root`` contains ``self.target_folder`` depending on ``train``:

          * initialize ImageFolder with target_folder.

        * else:

          * if all raw files are in ``root``:

            * parse ``self.target_folder`` from raw files.
            * initialize ImageFolder with ``self.target_folder``.

          * else:

            * raise error.

    Args:
        root: root directory of imagenet data, if root is ``None``, use default_dataset_root.
        train: if ``True``, load the train split, otherwise load the validation split.

    """

    raw_file_meta = {
        "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
        "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
        "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
    }  # ImageNet raw files
    default_train_dir = "train"
    default_val_dir = "val"
    default_devkit_dir = "ILSVRC2012_devkit_t12"

    def __init__(self, root: str = None, train: bool = True, **kwargs):
        # process the root path
        if root is None:
            self.root = self._default_root
        else:
            self.root = root

        if not os.path.exists(self.root):
            raise FileNotFoundError("dir %s does not exist" % self.root)

        self.devkit_dir = os.path.join(self.root, self.default_devkit_dir)

        if not os.path.exists(self.devkit_dir):
            logger.warning("devkit directory %s does not exists", self.devkit_dir)
            self._prepare_devkit()

        self.train = train

        if train:
            self.target_folder = os.path.join(self.root, self.default_train_dir)
        else:
            self.target_folder = os.path.join(self.root, self.default_val_dir)

        if not os.path.exists(self.target_folder):
            logger.warning(
                "expected image folder %s does not exist, try to load from raw file",
                self.target_folder,
            )
            if not self.check_raw_file():
                raise FileNotFoundError(
                    "expected image folder %s does not exist, and raw files do not exist in %s"
                    % (self.target_folder, self.root)
                )
            elif is_distributed():
                raise RuntimeError(
                    "extracting raw file shouldn't be done in distributed mode, use single process instead"
                )
            elif train:
                self._prepare_train()
            else:
                self._prepare_val()

        super().__init__(self.target_folder, **kwargs)

    @property
    def _default_root(self):
        return os.path.join(_default_dataset_root(), self.__class__.__name__)

    @property
    def valid_ground_truth(self):
        groud_truth_path = os.path.join(
            self.devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt"
        )
        if os.path.exists(groud_truth_path):
            with open(groud_truth_path, "r") as f:
                val_labels = f.readlines()
                return [int(val_label) for val_label in val_labels]
        else:
            raise FileNotFoundError(
                "valid ground truth file %s does not exist" % groud_truth_path
            )

    @property
    def meta(self):
        try:
            return load(os.path.join(self.devkit_dir, "meta.pkl"))
        except FileNotFoundError:
            import scipy.io

            meta_path = os.path.join(self.devkit_dir, "data", "meta.mat")
            if not os.path.exists(meta_path):
                raise FileNotFoundError("meta file %s does not exist" % meta_path)
            meta = scipy.io.loadmat(meta_path, squeeze_me=True)["synsets"]
            nums_children = list(zip(*meta))[4]
            meta = [
                meta[idx]
                for idx, num_children in enumerate(nums_children)
                if num_children == 0
            ]
            idcs, wnids, classes = list(zip(*meta))[:3]
            classes = [tuple(clss.split(", ")) for clss in classes]
            idx_to_wnid = dict(zip(idcs, wnids))
            wnid_to_classes = dict(zip(wnids, classes))
            logger.info(
                "saving cached meta file to %s",
                os.path.join(self.devkit_dir, "meta.pkl"),
            )
            save(
                (idx_to_wnid, wnid_to_classes),
                os.path.join(self.devkit_dir, "meta.pkl"),
            )
            return idx_to_wnid, wnid_to_classes

    def check_raw_file(self) -> bool:
        return all(
            [
                os.path.exists(os.path.join(self.root, value[0]))
                for _, value in self.raw_file_meta.items()
            ]
        )

    def _organize_val_data(self):
        id2wnid = self.meta[0]
        val_idcs = self.valid_ground_truth
        val_wnids = [id2wnid[idx] for idx in val_idcs]

        val_images = sorted(
            [
                os.path.join(self.target_folder, image)
                for image in os.listdir(self.target_folder)
            ]
        )

        logger.debug("mkdir for val set wnids")
        for wnid in set(val_wnids):
            os.makedirs(os.path.join(self.root, self.default_val_dir, wnid))

        logger.debug("mv val images into wnids dir")
        for wnid, img_file in tqdm(zip(val_wnids, val_images)):
            shutil.move(
                img_file,
                os.path.join(
                    self.root, self.default_val_dir, wnid, os.path.basename(img_file)
                ),
            )

    def _prepare_val(self):
        assert not self.train
        raw_filename, checksum = self.raw_file_meta["val"]
        raw_file = os.path.join(self.root, raw_filename)
        logger.info("checksum valid tar file %s ...", raw_file)
        assert (
            calculate_md5(raw_file) == checksum
        ), "checksum mismatch, {} may be damaged".format(raw_file)
        logger.info("extract valid tar file... this may take 10-20 minutes")
        untar(raw_file, self.target_folder)
        self._organize_val_data()

    def _prepare_train(self):
        assert self.train
        raw_filename, checksum = self.raw_file_meta["train"]
        raw_file = os.path.join(self.root, raw_filename)
        logger.info("checksum train tar file %s ...", raw_file)
        assert (
            calculate_md5(raw_file) == checksum
        ), "checksum mismatch, {} may be damaged".format(raw_file)
        logger.info("extract train tar file.. this may take several hours")
        untar(raw_file, self.target_folder)
        paths = [
            os.path.join(self.target_folder, child_dir)
            for child_dir in os.listdir(self.target_folder)
        ]
        for path in tqdm(paths):
            untar(path, os.path.splitext(path)[0], remove=True)

    def _prepare_devkit(self):
        raw_filename, checksum = self.raw_file_meta["devkit"]
        raw_file = os.path.join(self.root, raw_filename)
        logger.info("checksum devkit tar file %s ...", raw_file)
        assert (
            calculate_md5(raw_file) == checksum
        ), "checksum mismatch, {} may be damaged".format(raw_file)
        logger.info("extract devkit file..")
        untargz(os.path.join(self.root, self.raw_file_meta["devkit"][0]))
