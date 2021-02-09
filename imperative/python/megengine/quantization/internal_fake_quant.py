# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy
import math
from functools import partial

import numpy as np

from .. import functional as F
from ..autodiff import Function
from .fake_quant import _FakeQuantize
from .observer import MinMaxObserver
from .qconfig import QConfig
from .utils import QParams


