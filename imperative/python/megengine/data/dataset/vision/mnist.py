# -*- coding: utf-8 -*-
import gzip
import os
import struct
from typing import Tuple

import numpy as np
from tqdm import tqdm

from ....logger import get_logger
from .meta_vision import VisionDataset
from .utils import _default_dataset_root, load_raw_data_from_url

logger = get_logger(__name__)


class MNIST(VisionDataset):
    r"""MNIST dataset.
    The MNIST_ database (Modified National Institute of Standards and Technology database)
    is a large database of handwritten digits that is commonly used for training various image processing systems.
    The database is also widely used for training and testing in the field of machine learning.
    It was created by "re-mixing" the samples from `NIST`_'s original datasets.
    Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel
    bounding box and anti-aliased, which introduced grayscale levels.
    The MNIST database contains 60,000 training images and 10,000 testing images.

    The above introduction comes from `MNIST database - Wikipedia
    <https://en.wikipedia.org/wiki/MNIST_database>`_.

    Args:
        root:  Path for MNIST dataset downloading or loading. If it's ``None``,
            it will be set to ``~/.cache/megengine`` (the default root path). 
        train: If ``True``, use traning dataset; Otherwise use the test set.
        download: If ``True``, downloads the dataset from the internet and puts it in ``root`` directory.
            If dataset is already downloaded, it is not downloaded again.

    Returns:
        The MNIST :class:`~.Dataset` that can work with :class:`~.DataLoader`.

    Example:

       >>> from megengine.data.dataset import MNIST   # doctest: +SKIP
       >>> mnist = MNIST("/data/datasets/MNIST")  # Set the root path   # doctest: +SKIP
       >>> image, label = mnist[0]  # doctest: +SKIP
       >>> image.shape   # doctest: +SKIP
       (28, 28, 1)

    .. versionchanged:: 1.11 The original URL has been updated to a mirror URL

       *"Please refrain from accessing these files from automated scripts with high frequency. Make copies!"*
       As requested by the original provider of the MNIST dataset,
       now the dataset will be downloaded from the mirror site:
       https://ossci-datasets.s3.amazonaws.com/mnist/

    .. seealso::

       * MNIST dataset is used in :ref:`megengine-quick-start` tutorial as an example.
       * You can find a lot of machine learning projects using MNIST dataset on the internet.
    
    .. _MNIST: http://yann.lecun.com/exdb/mnist/
    .. _NIST: https://www.nist.gov/data
    """

    url_path = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    raw_file_name = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    raw_file_md5 = [
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "d53e105ee54ea40749a09fcbcd1e9432",
        "9fb629c4189551a2d022fa330f9573f3",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ]

    def __init__(
        self, root: str = None, train: bool = True, download: bool = True,
    ):
        super().__init__(root, order=("image", "image_category"))

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

        if self._check_raw_files():
            self.process(train)
        elif download:
            self.download()
            self.process(train)
        else:
            raise ValueError(
                "root does not contain valid raw files, please set download=True"
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
        return self._meta_data

    def _check_raw_files(self):
        return all(
            [
                os.path.exists(os.path.join(self.root, path))
                for path in self.raw_file_name
            ]
        )

    def download(self):
        for file_name, md5 in zip(self.raw_file_name, self.raw_file_md5):
            url = self.url_path + file_name
            load_raw_data_from_url(url, file_name, md5, self.root)

    def process(self, train):
        # load raw files and transform them into meta data and datasets Tuple(np.array)
        logger.info("process the raw files of %s set...", "train" if train else "test")
        if train:
            meta_data_images, images = parse_idx3(
                os.path.join(self.root, self.raw_file_name[0])
            )
            meta_data_labels, labels = parse_idx1(
                os.path.join(self.root, self.raw_file_name[1])
            )
        else:
            meta_data_images, images = parse_idx3(
                os.path.join(self.root, self.raw_file_name[2])
            )
            meta_data_labels, labels = parse_idx1(
                os.path.join(self.root, self.raw_file_name[3])
            )

        self._meta_data = {
            "images": meta_data_images,
            "labels": meta_data_labels,
        }
        self.arrays = (images, labels.astype(np.int32))


def parse_idx3(idx3_file):
    # parse idx3 file to meta data and data in numpy array (images)
    logger.debug("parse idx3 file %s ...", idx3_file)
    assert idx3_file.endswith(".gz")
    with gzip.open(idx3_file, "rb") as f:
        bin_data = f.read()

    #  parse meta data
    offset = 0
    fmt_header = ">iiii"
    magic, imgs, height, width = struct.unpack_from(fmt_header, bin_data, offset)
    meta_data = {"magic": magic, "imgs": imgs, "height": height, "width": width}

    # parse images
    image_size = height * width
    offset += struct.calcsize(fmt_header)
    fmt_image = ">" + str(image_size) + "B"
    images = []
    bar = tqdm(total=meta_data["imgs"], ncols=80)
    for image in struct.iter_unpack(fmt_image, bin_data[offset:]):
        images.append(np.array(image, dtype=np.uint8).reshape((height, width, 1)))
        bar.update()
    bar.close()
    return meta_data, images


def parse_idx1(idx1_file):
    # parse idx1 file to meta data and data in numpy array (labels)
    logger.debug("parse idx1 file %s ...", idx1_file)
    assert idx1_file.endswith(".gz")
    with gzip.open(idx1_file, "rb") as f:
        bin_data = f.read()

    # parse meta data
    offset = 0
    fmt_header = ">ii"
    magic, imgs = struct.unpack_from(fmt_header, bin_data, offset)
    meta_data = {"magic": magic, "imgs": imgs}

    # parse labels
    offset += struct.calcsize(fmt_header)
    fmt_image = ">B"
    labels = np.empty(imgs, dtype=int)
    bar = tqdm(total=meta_data["imgs"], ncols=80)
    for i, label in enumerate(struct.iter_unpack(fmt_image, bin_data[offset:])):
        labels[i] = label[0]
        bar.update()
    bar.close()
    return meta_data, labels
