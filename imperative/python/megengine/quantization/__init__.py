# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from .fake_quant import TQT, FakeQuantize
from .observer import (
    ExponentialMovingAverageObserver,
    HistogramObserver,
    MinMaxObserver,
    Observer,
    PassiveObserver,
    SyncExponentialMovingAverageObserver,
    SyncMinMaxObserver,
)
from .qconfig import (
    QConfig,
    calibration_qconfig,
    easyquant_qconfig,
    ema_fakequant_qconfig,
    ema_lowbit_fakequant_qconfig,
    min_max_fakequant_qconfig,
    passive_qconfig,
    sync_ema_fakequant_qconfig,
    tqt_qconfig,
)
from .quantize import (
    apply_easy_quant,
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
    propagate_qconfig,
    quantize,
    quantize_qat,
    reset_qconfig,
)
from .utils import (
    QParams,
    QuantMode,
    create_qparams,
    fake_quant_bias,
    fake_quant_tensor,
)
