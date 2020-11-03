# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import struct

import numpy as np


def load_tensor_binary(fobj):
    """
    Load a tensor dumped by the :class:`BinaryOprIODump` plugin; the actual
    tensor value dump is implemented by ``mgb::debug::dump_tensor``.

    Multiple values can be compared by ``tools/compare_binary_iodump.py``.

    :param fobj: file object, or a string that contains the file name.
    :return: tuple ``(tensor_value, tensor_name)``.
    """
    if isinstance(fobj, str):
        with open(fobj, "rb") as fin:
            return load_tensor_binary(fin)

    DTYPE_LIST = {
        0: np.float32,
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        # 5: _mgb.intb1,
        # 6: _mgb.intb2,
        # 7: _mgb.intb4,
        8: None,
        9: np.float16,
        # quantized dtype start from 100000
        # see MEGDNN_PARAMETERIZED_DTYPE_ENUM_BASE in
        # dnn/include/megdnn/dtype.h
        100000: np.uint8,
        100001: np.int32,
        100002: np.int8,
    }

    header_fmt = struct.Struct("III")
    name_len, dtype, max_ndim = header_fmt.unpack(fobj.read(header_fmt.size))
    assert (
        DTYPE_LIST[dtype] is not None
    ), "Cannot load this tensor: dtype Byte is unsupported."

    shape = list(struct.unpack("I" * max_ndim, fobj.read(max_ndim * 4)))
    while shape[-1] == 0:
        shape.pop(-1)
    name = fobj.read(name_len).decode("ascii")
    return np.fromfile(fobj, dtype=DTYPE_LIST[dtype]).reshape(shape), name
