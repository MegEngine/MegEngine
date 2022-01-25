# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ctypes import *

import numpy as np

from .base import _Ctensor, _lib, _LiteCObjBase
from .network import *
from .struct import LiteDataType, LiteDeviceType, LiteIOType, Structure
from .tensor import *

LiteDecryptionFunc = CFUNCTYPE(
    c_size_t, c_void_p, c_size_t, POINTER(c_uint8), c_size_t, c_void_p
)


class _GlobalAPI(_LiteCObjBase):
    """
    get the api from the lib
    """

    _api_ = [
        ("LITE_get_device_count", [c_int, POINTER(c_size_t)]),
        ("LITE_try_coalesce_all_free_memory", []),
        (
            "LITE_register_decryption_and_key",
            [c_char_p, LiteDecryptionFunc, POINTER(c_uint8), c_size_t],
        ),
        (
            "LITE_update_decryption_or_key",
            [c_char_p, c_void_p, POINTER(c_uint8), c_size_t],
        ),
        ("LITE_set_loader_lib_path", [c_char_p]),
        ("LITE_set_persistent_cache", [c_char_p, c_int]),
        # ('LITE_set_tensor_rt_cache', [c_char_p]),
        ("LITE_dump_persistent_cache", [c_char_p]),
        ("LITE_dump_tensor_rt_cache", [c_char_p]),
        ("LITE_register_memory_pair", [c_void_p, c_void_p, c_size_t, c_int, c_int]),
        ("LITE_clear_memory_pair", [c_void_p, c_void_p, c_int, c_int]),
    ]


def decryption_func(func):
    """the decryption function decorator
    :type func: a function accept three array, in_arr, key_arr and out_arr, if out_arr is None, just query the out array lenght in byte
    """

    @CFUNCTYPE(c_size_t, c_void_p, c_size_t, POINTER(c_uint8), c_size_t, c_void_p)
    def wrapper(c_in_data, in_length, c_key_data, key_length, c_out_data):
        in_arr = np.frombuffer(c_in_data, dtype=np.uint8, count=in_length)
        key_arr = np.frombuffer(c_key_data, dtype=np.uint8, count=key_length)
        if c_out_data:
            out_length = func(in_arr, None)
            out_arr = np.frombuffer(c_out_data, dtype=np.uint8, count=out_length)
            return func(in_arr, key_arr, out_arr)
        # just query the output length
        else:
            return func(in_arr, key_arr, None)

    return wrapper


class LiteGlobal(object):
    """
    some global config in lite
    """

    _api = _GlobalAPI()._lib

    @staticmethod
    def register_decryption_and_key(decryption_name, decryption_func, key):
        c_name = c_char_p(decryption_name.encode("utf-8"))
        key_length = len(key)
        c_key = (c_uint8 * key_length)(*key)
        LiteGlobal._api.LITE_register_decryption_and_key(
            c_name, decryption_func, c_key, key_length
        )

    @staticmethod
    def update_decryption_key(decryption_name, key):
        c_name = c_char_p(decryption_name.encode("utf-8"))
        key_length = len(key)
        c_key = (c_uint8 * key_length)(*key)
        LiteGlobal._api.LITE_update_decryption_or_key(c_name, None, c_key, key_length)

    @staticmethod
    def set_loader_lib_path(path):
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_set_loader_lib_path(c_path)

    @staticmethod
    def set_persistent_cache(path, always_sync=False):
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_set_persistent_cache(c_path, always_sync)

    @staticmethod
    def set_tensorrt_cache(path):
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_set_tensorrt_cache(c_path)

    @staticmethod
    def dump_persistent_cache(path):
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_dump_persistent_cache(c_path)

    @staticmethod
    def dump_tensorrt_cache():
        LiteGlobal._api.LITE_dump_tensorrt_cache()

    @staticmethod
    def get_device_count(device_type):
        count = c_size_t()
        LiteGlobal._api.LITE_get_device_count(device_type, byref(count))
        return count.value

    @staticmethod
    def try_coalesce_all_free_memory():
        LiteGlobal._api.LITE_try_coalesce_all_free_memory()

    @staticmethod
    def register_memory_pair(
        vir_ptr, phy_ptr, length, device, backend=LiteBackend.LITE_DEFAULT
    ):
        assert isinstance(vir_ptr, c_void_p) and isinstance(
            phy_ptr, c_void_p
        ), "clear memory pair only accept c_void_p type."
        LiteGlobal._api.LITE_register_memory_pair(
            vir_ptr, phy_ptr, length, device, backend
        )

    @staticmethod
    def clear_memory_pair(vir_ptr, phy_ptr, device, backend=LiteBackend.LITE_DEFAULT):
        assert isinstance(vir_ptr, c_void_p) and isinstance(
            phy_ptr, c_void_p
        ), "clear memory pair only accept c_void_p type."
        LiteGlobal._api.LITE_clear_memory_pair(vir_ptr, phy_ptr, device, backend)
