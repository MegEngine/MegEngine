# -*- coding: utf-8 -*-
import os

from .base import *
from .base import version as __version__
from .global_setting import *
from .network import *
from .struct import *
from .tensor import *
from .utils import *


def config_env():
    """
    more detail: please check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    default config to LAZY, which means cuda module will be loaded when needed to save cu memory by load fatbin elf section
    please do not call any cuda api before this function
    """
    if not os.getenv("ALREADY_CONFIG_CUDA_LOADING_MODE", False):
        may_user_config = os.getenv("CUDA_MODULE_LOADING", "LAZY")
        os.environ["CUDA_MODULE_LOADING"] = may_user_config


config_env()
