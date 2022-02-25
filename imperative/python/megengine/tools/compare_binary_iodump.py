#! /usr/bin/env python3
import argparse
import os
import struct
import textwrap
from pathlib import Path

import numpy as np


def load_tensor_binary(fobj):
    """Load a tensor dumped by the :class:`BinaryOprIODump` plugin; the actual
    tensor value dump is implemented by ``mgb::debug::dump_tensor``.

    Args:
      fobj: file object, or a string that contains the file name.

    Returns:
      tuple ``(tensor_value, tensor_name)``.
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


def check(v0, v1, name, max_err):
    v0 = np.ascontiguousarray(v0, dtype=np.float32)
    v1 = np.ascontiguousarray(v1, dtype=np.float32)
    assert np.isfinite(v0.sum()) and np.isfinite(
        v1.sum()
    ), "{} not finite: sum={} vs sum={}".format(name, v0.sum(), v1.sum())
    assert v0.shape == v1.shape, "{} shape mismatch: {} vs {}".format(
        name, v0.shape, v1.shape
    )
    vdiv = np.max([np.abs(v0), np.abs(v1), np.ones_like(v0)], axis=0)
    err = np.abs(v0 - v1) / vdiv
    rst = err > max_err
    if rst.sum():
        idx = tuple(i[0] for i in np.nonzero(rst))
        raise AssertionError(
            "{} not equal: "
            "shape={} nonequal_idx={} v0={} v1={} err={}".format(
                name, v0.shape, idx, v0[idx], v1[idx], err[idx]
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "compare tensor dumps generated BinaryOprIODump plugin, "
            "it can compare two dirs or two single files"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input0", help="dirname or filename")
    parser.add_argument("input1", help="dirname or filename")
    parser.add_argument(
        "-e", "--max-err", type=float, default=1e-3, help="max allowed error"
    )
    parser.add_argument(
        "-s", "--stop-on-error", action="store_true", help="do not compare "
    )
    args = parser.parse_args()

    files0 = set()
    files1 = set()
    if os.path.isdir(args.input0):
        assert os.path.isdir(args.input1)
        name0 = set()
        name1 = set()
        for i in os.listdir(args.input0):
            files0.add(str(Path(args.input0) / i))
            name0.add(i)

        for i in os.listdir(args.input1):
            files1.add(str(Path(args.input1) / i))
            name1.add(i)

        assert name0 == name1, "dir files mismatch: a-b={} b-a={}".format(
            name0 - name1, name1 - name0
        )
    else:
        files0.add(args.input0)
        files1.add(args.input1)
    files0 = sorted(files0)
    files1 = sorted(files1)

    for i, j in zip(files0, files1):
        val0, name0 = load_tensor_binary(i)
        val1, name1 = load_tensor_binary(j)
        name = "{}: \n{}\n{}\n".format(
            i, "\n  ".join(textwrap.wrap(name0)), "\n  ".join(textwrap.wrap(name1))
        )
        try:
            check(val0, val1, name, args.max_err)
        except Exception as exc:
            if args.stop_on_error:
                raise exc
            print(exc)


if __name__ == "__main__":
    main()
