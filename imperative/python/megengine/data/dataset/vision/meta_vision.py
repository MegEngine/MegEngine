# -*- coding: utf-8 -*-
import collections.abc
import os

from ..meta_dataset import Dataset


class VisionDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root, *, order=None, supported_order=None):
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        if order is None:
            order = ("image",)
        if not isinstance(order, collections.abc.Sequence):
            raise ValueError(
                "order should be a sequence, but got order={}".format(order)
            )

        if supported_order is not None:
            assert isinstance(supported_order, collections.abc.Sequence)
            for k in order:
                if k not in supported_order:
                    raise NotImplementedError("{} is unsupported data type".format(k))
        self.order = order

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
