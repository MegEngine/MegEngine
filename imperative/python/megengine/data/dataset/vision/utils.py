# -*- coding: utf-8 -*-
import hashlib
import os
import tarfile

from ....distributed.group import is_distributed
from ....logger import get_logger
from ....utils.http_download import download_from_url

IMG_EXT = (".jpg", ".png", ".jpeg", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

logger = get_logger(__name__)


def _default_dataset_root():
    default_dataset_root = os.path.expanduser(
        os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "megengine")
    )

    return default_dataset_root


def load_raw_data_from_url(url: str, filename: str, target_md5: str, raw_data_dir: str):
    cached_file = os.path.join(raw_data_dir, filename)
    logger.debug(
        "load_raw_data_from_url: downloading to or using cached %s ...", cached_file
    )
    if not os.path.exists(cached_file):
        if is_distributed():
            logger.warning(
                "Downloading raw data in DISTRIBUTED mode\n"
                "    File may be downloaded multiple times. We recommend\n"
                "    users to download in single process first."
            )
        md5 = download_from_url(url, cached_file)
    else:
        md5 = calculate_md5(cached_file)
    if target_md5 == md5:
        logger.debug("%s exists with correct md5: %s", filename, target_md5)
    else:
        os.remove(cached_file)
        raise RuntimeError("{} exists but fail to match md5".format(filename))


def calculate_md5(filename):
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def is_img(filename):
    return filename.lower().endswith(IMG_EXT)


def untar(path, to=None, remove=False):
    if to is None:
        to = os.path.dirname(path)
    with tarfile.open(path, "r") as tar:
        tar.extractall(path=to)

    if remove:
        os.remove(path)


def untargz(path, to=None, remove=False):
    if path.endswith(".tar.gz"):
        if to is None:
            to = os.path.dirname(path)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=to)
    else:
        raise ValueError("path %s does not end with .tar" % path)

    if remove:
        os.remove(path)
