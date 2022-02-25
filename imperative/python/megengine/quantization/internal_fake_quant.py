from functools import partial

from .. import functional as F
from ..autodiff import Function
from .fake_quant import _FakeQuantize
from .observer import MinMaxObserver
from .qconfig import QConfig
from .utils import QParams
