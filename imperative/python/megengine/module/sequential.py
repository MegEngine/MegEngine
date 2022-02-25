# -*- coding: utf-8 -*-
from collections import OrderedDict

from .module import Module


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    Examples:

        .. testcode::

            import numpy as np
            import megengine as mge
            import megengine.module as M
            import megengine.functional as F
            from collections import OrderedDict

            batch_size = 64
            data = mge.tensor(np.zeros((batch_size, 28 * 28)), dtype=np.float32)
            label = mge.tensor(np.zeros(batch_size,), dtype=np.int32)

            net0 = M.Sequential(
                    M.Linear(28 * 28, 320),
                    M.Linear(320, 10)
                )
            pred0 = net0(data)

            modules = OrderedDict()
            modules["fc0"] = M.Linear(28 * 28, 320)
            modules["fc1"] = M.Linear(320, 10)
            net1 = M.Sequential(modules)
            pred1 = net1(data)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.layer_keys = []
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                # self.add_module(key, module)
                setattr(self, key, module)
                self.layer_keys.append(key)
        else:
            for idx, module in enumerate(args):
                # self.add_module(str(idx), module)
                setattr(self, str(idx), module)
                self.layer_keys.append(str(idx))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(
                OrderedDict(zip(self.layer_keys[idx], self.layer_values[idx]))
            )
        else:
            return getattr(self, self.layer_keys[idx])

    def __setitem__(self, idx, module):
        key = self.layer_keys[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.layer_keys[idx]:
                delattr(self, key)
                del self.layer_keys[idx]
        else:
            delattr(self, self.layer_keys[idx])
            del self.layer_keys[idx]

    def __len__(self):
        return len(self.layer_keys)

    def __iter__(self):
        return iter(self.layer_values)

    @property
    def layer_values(self):
        return [getattr(self, key) for key in self.layer_keys]

    def forward(self, inp):
        # avoid layer_values as a name prefix, see Module.__getattribute__
        for layer in [getattr(self, key) for key in self.layer_keys]:
            inp = layer(inp)
        return inp
