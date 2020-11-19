# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from .fake_quant import FakeQuantize
from .internal_fake_quant import *
from .observer import HistogramObserver, Observer
from .qconfig import (
    QConfig,
    calibration_qconfig,
    ema_fakequant_qconfig,
    ema_lowbit_fakequant_qconfig,
    min_max_fakequant_qconfig,
    passive_qconfig,
    sync_ema_fakequant_qconfig,
    tqt_qconfig,
)
from .utils import QuantMode
