# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import hashlib
import os
import shutil
from tempfile import NamedTemporaryFile

import requests
from tqdm import tqdm

from ..logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1024
HTTP_CONNECTION_TIMEOUT = 5


class HTTPDownloadError(BaseException):
    """The class that represents http request error."""


def download_from_url(url: str, dst: str, http_read_timeout=120):
    """
    Downloads file from given url to ``dst``.

    :param url: source URL.
    :param dst: saving path.
    :param http_read_timeout: how many seconds to wait for data before giving up.
    """
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)

    resp = requests.get(
        url, timeout=(HTTP_CONNECTION_TIMEOUT, http_read_timeout), stream=True
    )
    if resp.status_code != 200:
        raise HTTPDownloadError("An error occured when downloading from {}".format(url))

    md5 = hashlib.md5()
    total_size = int(resp.headers.get("Content-Length", 0))
    bar = tqdm(
        total=total_size, unit="iB", unit_scale=True, ncols=80
    )  # pylint: disable=blacklisted-name
    try:
        with NamedTemporaryFile("w+b", delete=False, suffix=".tmp", dir=dst_dir) as f:
            logger.info("Download file to temp file %s", f.name)
            for chunk in resp.iter_content(CHUNK_SIZE):
                if not chunk:
                    break
                bar.update(len(chunk))
                f.write(chunk)
                md5.update(chunk)
            bar.close()
        shutil.move(f.name, dst)
    finally:
        # ensure tmp file is removed
        if os.path.exists(f.name):
            os.remove(f.name)
    return md5.hexdigest()
