# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .fake_quant import FakeQuantize
from .observer import Observer
from .qconfig import QConfig, ema_fakequant_qconfig, min_max_fakequant_qconfig
from .quantize import quantize, quantize_qat
