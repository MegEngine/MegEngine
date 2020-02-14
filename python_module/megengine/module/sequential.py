# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from collections import OrderedDict

from .module import Module


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example:

    .. testcode::

        import numpy as np
        import megengine.nn as nn
        import megengine.nn.functional as F

        batch_size = 64
        data = nn.Input("data", shape=(batch_size, 1, 28, 28), dtype=np.float32, value=np.zeros((batch_size, 1, 28, 28)))
        label = nn.Input("label", shape=(batch_size,), dtype=np.int32, value=np.zeros(batch_size,))

        data = data.reshape(batch_size, -1)
        net = nn.Sequential(
                nn.Linear(28 * 28, 320),
                nn.Linear(320, 500),
                nn.Linear(500, 320),
                nn.Linear(320, 10)
            )
        pred = net(data)

        loss = F.cross_entropy_with_softmax(pred, label)

    """

    def __init__(self, *args):
        super().__init__()
        self.layer_keys = []
        self.layer_values = []
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                # self.add_module(key, module)
                setattr(self, key, module)
                self.layer_keys.append(key)
                self.layer_values.append(module)
        else:
            for idx, module in enumerate(args):
                # self.add_module(str(idx), module)
                setattr(self, str(idx), module)
                self.layer_keys.append(str(idx))
                self.layer_values.append(module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(
                OrderedDict(zip(self.layer_keys[idx], self.layer_values[idx]))
            )
        else:
            return self.layer_values[idx]

    def __setitem__(self, idx, module):
        key = self.layer_keys[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.layer_keys[idx]:
                delattr(self, key)
                del self.layer_keys[idx]
                del self.layer_values[idx]
        else:
            delattr(self, self.layer_keys[idx])
            del self.layer_keys[idx]
            del self.layer_values[idx]

    def __len__(self):
        return len(self.layer_keys)

    def __iter__(self):
        return iter(self.layer_values)

    def forward(self, inp):
        for layer in self.layer_values:
            inp = layer(inp)
        return inp
