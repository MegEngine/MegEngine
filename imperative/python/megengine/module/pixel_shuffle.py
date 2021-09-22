# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..functional.nn import pixel_shuffle
from .module import Module


class PixelShuffle(Module):
    r"""
    Rearranges elements in a tensor of shape (*, C x r^2, H, W) to a tensor of
    shape (*, C, H x r, W x r), where r is an upscale factor, where * is zero
    or more batch dimensions.
    """

    def __init__(self, upscale_factor: int, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.upscale_factor)
