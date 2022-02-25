# -*- coding: utf-8 -*-
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
# ---------------------------------------------------------------------
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
# ----------------------------------------------------------------------
import collections.abc
import re

import numpy as np

np_str_obj_array_pattern = re.compile(r"[aO]")
default_collate_err_msg_format = (
    "default_collator: inputs must contain numpy arrays, numbers, "
    "Unicode strings, bytes, dicts or lists; found {}"
)


class Collator:
    r"""Used for merging a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a dataset.
    Modified from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    """

    def apply(self, inputs):
        elem = inputs[0]
        elem_type = type(elem)
        if (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            elem = inputs[0]
            if elem_type.__name__ == "ndarray":
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return np.ascontiguousarray(np.stack(inputs))
            elif elem.shape == ():  # scalars
                return np.array(inputs)
        elif isinstance(elem, float):
            return np.array(inputs, dtype=np.float64)
        elif isinstance(elem, int):
            return np.array(inputs)
        elif isinstance(elem, (str, bytes)):
            return inputs
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.apply([d[key] for d in inputs]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(self.apply(samples) for samples in zip(*inputs)))
        elif isinstance(elem, collections.abc.Sequence):
            transposed = zip(*inputs)
            return [self.apply(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
