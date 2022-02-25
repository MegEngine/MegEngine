# -*- coding: utf-8 -*-
import os
import pickle
import tarfile
from typing import Tuple

import numpy as np

from ....logger import get_logger
from .meta_vision import VisionDataset
from .utils import _default_dataset_root, load_raw_data_from_url

logger = get_logger(__name__)


class CIFAR10(VisionDataset):
    r""":class:`~.Dataset` for CIFAR10 meta data."""

    url_path = "http://www.cs.utoronto.ca/~kriz/"
    raw_file_name = "cifar-10-python.tar.gz"
    raw_file_md5 = "c58f30108f718f92721af3b95e74349a"
    raw_file_dir = "cifar-10-batches-py"
    train_batch = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_batch = ["test_batch"]
    meta_info = {"name": "batches.meta"}

    def __init__(
        self,
        root: str = None,
        train: bool = True,
        download: bool = True,
        timeout: int = 500,
    ):
        super().__init__(root, order=("image", "image_category"))

        self.timeout = timeout

        # process the root path
        if root is None:
            self.root = self._default_root
            if not os.path.exists(self.root):
                os.makedirs(self.root)
        else:
            self.root = root
            if not os.path.exists(self.root):
                if download:
                    logger.debug(
                        "dir %s does not exist, will be automatically created",
                        self.root,
                    )
                    os.makedirs(self.root)
                else:
                    raise ValueError("dir %s does not exist" % self.root)

        self.target_file = os.path.join(self.root, self.raw_file_dir)

        # check existence of target pickle dir, if exists load the
        # pickle file no matter what download is set
        if os.path.exists(self.target_file):
            if train:
                self.arrays = self.bytes2array(self.train_batch)
            else:
                self.arrays = self.bytes2array(self.test_batch)
        else:
            if download:
                self.download()
                if train:
                    self.arrays = self.bytes2array(self.train_batch)
                else:
                    self.arrays = self.bytes2array(self.test_batch)
            else:
                raise ValueError(
                    "dir does not contain target file %s, please set download=True"
                    % (self.target_file)
                )

    def __getitem__(self, index: int) -> Tuple:
        return tuple(array[index] for array in self.arrays)

    def __len__(self) -> int:
        return len(self.arrays[0])

    @property
    def _default_root(self):
        return os.path.join(_default_dataset_root(), self.__class__.__name__)

    @property
    def meta(self):
        meta_path = os.path.join(self.root, self.raw_file_dir, self.meta_info["name"])
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        return meta

    def download(self):
        url = self.url_path + self.raw_file_name
        load_raw_data_from_url(url, self.raw_file_name, self.raw_file_md5, self.root)
        self.process()

    def untar(self, file_path, dirs):
        assert file_path.endswith(".tar.gz")
        logger.debug("untar file %s to %s", file_path, dirs)
        t = tarfile.open(file_path)
        t.extractall(path=dirs)

    def bytes2array(self, filenames):
        data = []
        label = []
        for filename in filenames:
            path = os.path.join(self.root, self.raw_file_dir, filename)
            logger.debug("unpickle file %s", path)
            with open(path, "rb") as fo:
                dic = pickle.load(fo, encoding="bytes")
                batch_data = dic[b"data"].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
                data.extend(list(batch_data[..., [2, 1, 0]]))
                label.extend(dic[b"labels"])
        label = np.array(label, dtype=np.int32)
        return (data, label)

    def process(self):
        logger.info("process raw data ...")
        self.untar(os.path.join(self.root, self.raw_file_name), self.root)


class CIFAR100(CIFAR10):
    r""":class:`~.Dataset` for CIFAR100 meta data."""

    url_path = "http://www.cs.utoronto.ca/~kriz/"
    raw_file_name = "cifar-100-python.tar.gz"
    raw_file_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    raw_file_dir = "cifar-100-python"
    train_batch = ["train"]
    test_batch = ["test"]
    meta_info = {"name": "meta"}

    @property
    def meta(self):
        meta_path = os.path.join(self.root, self.raw_file_dir, self.meta_info["name"])
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        return meta

    def bytes2array(self, filenames):
        data = []
        fine_label = []
        coarse_label = []
        for filename in filenames:
            path = os.path.join(self.root, self.raw_file_dir, filename)
            logger.debug("unpickle file %s", path)
            with open(path, "rb") as fo:
                dic = pickle.load(fo, encoding="bytes")
                batch_data = dic[b"data"].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
                data.extend(list(batch_data[..., [2, 1, 0]]))
                fine_label.extend(dic[b"fine_labels"])
                coarse_label.extend(dic[b"coarse_labels"])
        fine_label = np.array(fine_label, dtype=np.int32)
        coarse_label = np.array(coarse_label, dtype=np.int32)
        return data, fine_label, coarse_label
