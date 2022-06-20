# -*- coding: utf-8 -*-

import logging
from ctypes import *
from enum import Enum, IntEnum


class LiteBackend(IntEnum):
    """
    The backend type enum, default only
    """

    LITE_DEFAULT = 0


class LiteDeviceType(IntEnum):
    """
    The backend device type enum

    Note:
        compute and storage will base on the device
    """

    LITE_CPU = 0
    LITE_CUDA = 1
    LITE_ATLAS = 3
    LITE_NPU = 4
    LITE_CAMBRICON = 5
    LITE_DEVICE_DEFAULT = 7


class LiteDataType(IntEnum):
    """
    The tensor data type enum

    Note:
        half for float16, int for int32
    """

    LITE_FLOAT = 0
    LITE_HALF = 1
    LITE_INT = 2
    LITE_INT16 = 3
    LITE_INT8 = 4
    LITE_UINT8 = 5
    LITE_UINT16 = 6


class LiteTensorPhase(IntEnum):
    """
    The tensor type enum
    Note:
        LITE_IO for both LITE_INPUT and LITE_OUTPUT
    """

    LITE_IO = 0
    LITE_INPUT = 1
    LITE_OUTPUT = 2


class LiteIOType(IntEnum):
    """
    The input and output type enum, include SHAPE and VALUE
    sometimes user only need the shape of the output tensor
    """

    LITE_IO_VALUE = 0
    LITE_IO_SHAPE = 1


class LiteAlgoSelectStrategy(IntEnum):
    """
    Operation algorithm seletion strategy type enum, some operations have
    multi algorithms, different algorithm has different attribute, according to
    the strategy, the best algorithm will be selected.

    Note:
        These strategies can be combined
        LITE_ALGO_HEURISTIC | LITE_ALGO_PROFILE means: if profile cache not valid,
        use heuristic instead

        LITE_ALGO_HEURISTIC | LITE_ALGO_REPRODUCIBLE means: heuristic choice the
        reproducible algo

        LITE_ALGO_PROFILE | LITE_ALGO_REPRODUCIBLE means: profile the best
        algorithm from the reproducible algorithms set

        LITE_ALGO_PROFILE | LITE_ALGO_OPTIMIZED means: profile the best
        algorithm form the optimzed algorithms, thus profile will process fast

        LITE_ALGO_PROFILE | LITE_ALGO_OPTIMIZED | LITE_ALGO_REPRODUCIBLE means:
        profile the best algorithm form the optimzed and reproducible algorithms
    """

    LITE_ALGO_HEURISTIC = 1
    LITE_ALGO_PROFILE = 2
    LITE_ALGO_REPRODUCIBLE = 4
    LITE_ALGO_OPTIMIZED = 8


class LiteLogLevel(IntEnum):
    """
    Log level enum  

    Note:
        DEBUG: The most verbose level, printing debugging info  

        INFO: The default level  

        WARN: Printing warnings  

        ERROR: The least verbose level, printing errors only  
    """

    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
