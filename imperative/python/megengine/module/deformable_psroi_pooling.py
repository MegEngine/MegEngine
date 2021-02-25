# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..functional import deformable_psroi_pooling
from .module import Module


class DeformablePSROIPooling(Module):
    def __init__(
        self,
        no_trans,
        part_size,
        pooled_h,
        pooled_w,
        sample_per_part,
        spatial_scale,
        trans_std: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.no_trans = no_trans
        self.part_size = part_size
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.sample_per_part = sample_per_part
        self.spatial_scale = spatial_scale
        self.trans_std = trans_std

    def forward(self, inp, rois, trans):
        return deformable_psroi_pooling(
            inp,
            rois,
            trans,
            self.no_trans,
            self.part_size,
            self.pooled_h,
            self.pooled_w,
            self.sample_per_part,
            self.spatial_scale,
            self.trans_std,
        )
