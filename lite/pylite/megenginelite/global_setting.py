# -*- coding: utf-8 -*-

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
    Get APIs from the lib
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
        ("LITE_lookup_physic_ptr", [c_void_p, POINTER(c_void_p), c_int, c_int]),
    ]


def decryption_func(func):
    """the decryption function decorator.
    
    .. note::

       The function accept three array: ``in_arr``, ``key_arr`` and ``out_arr``.
       If ``out_arr`` is None, just query the out array length in byte.
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
    Some global config in lite
    """

    _api = _GlobalAPI()._lib

    @staticmethod
    def register_decryption_and_key(decryption_name, decryption_func, key):
        """Register a custom decryption method and key to lite

        Args:
            decryption_name: the name of the decryption, which will act as the hash 
                key to find the decryption method.
            decryption_func: the decryption function, which will decrypt the model with
                the registered key, then return the decrypted model.
                See :py:func:`~.decryption_func` for more details.
            key: the decryption key of the method.
        """
        c_name = c_char_p(decryption_name.encode("utf-8"))
        key_length = len(key)
        c_key = (c_uint8 * key_length)(*key)
        LiteGlobal._api.LITE_register_decryption_and_key(
            c_name, decryption_func, c_key, key_length
        )

    @staticmethod
    def update_decryption_key(decryption_name, key):
        """Update decryption key of a custom decryption method.

        Args:
            decrypt_name:  the name of the decryption, 
                which will act as the hash key to find the decryption method.
            key:  the decryption key of the method,
                if the length of key is zero, the key will not be updated.
        """
        c_name = c_char_p(decryption_name.encode("utf-8"))
        key_length = len(key)
        c_key = (c_uint8 * key_length)(*key)
        LiteGlobal._api.LITE_update_decryption_or_key(c_name, None, c_key, key_length)

    @staticmethod
    def set_loader_lib_path(path):
        """Set the loader path to be used in lite.

        Args:
            path: the file path which store the loader library.
        """
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_set_loader_lib_path(c_path)

    @staticmethod
    def set_persistent_cache(path, always_sync=False):
        """Set the algo policy cache file for CPU/CUDA,
        the algo policy cache is produced by MegEngine fast-run.
        
        Args:
            path: the file path which store the cache.
            always_sync: always update the cache file when model runs.
        """
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_set_persistent_cache(c_path, always_sync)

    @staticmethod
    def set_tensorrt_cache(path):
        """Set the TensorRT engine cache path for serialized prebuilt ICudaEngine.

        Args:
            path: the cache file path to set
        """
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_set_tensorrt_cache(c_path)

    @staticmethod
    def dump_persistent_cache(path):
        """Dump the PersistentCache policy cache to the specific file.
        If the network is set to profile when forward, 
        though this the algo policy will dump to file.

        Args:
            path: the cache file path to be dumped.
        """
        c_path = c_char_p(path.encode("utf-8"))
        LiteGlobal._api.LITE_dump_persistent_cache(c_path)

    @staticmethod
    def dump_tensorrt_cache():
        """Dump the TensorRT cache to the file set in :py:func:`~.set_tensorrt_cache`."""
        LiteGlobal._api.LITE_dump_tensorrt_cache()

    @staticmethod
    def get_device_count(device_type):
        """Get the number of device of the given device type in current context.

        Args:
            device_type: the device type to be counted.

        Returns:
            the number of device.
        """
        count = c_size_t()
        LiteGlobal._api.LITE_get_device_count(device_type, byref(count))
        return count.value

    @staticmethod
    def try_coalesce_all_free_memory():
        """Try to coalesce all free memory in MegEngine.
        When call it MegEnine Lite will try to free all the unused memory
        thus decrease the runtime memory usage.
        """
        LiteGlobal._api.LITE_try_coalesce_all_free_memory()

    @staticmethod
    def register_memory_pair(
        vir_ptr, phy_ptr, length, device, backend=LiteBackend.LITE_DEFAULT
    ):
        """Register the physical and virtual address pair to the MegEngine,
        some device need the map from physical to virtual.

        Args:
            vir_ptr: the virtual ptr to set to MegEngine.
            phy_ptr: the physical ptr to set to MegEngine.
            length: the length of bytes to set pair memory.
            device: the the device to set the pair memory.
            backend: the backend to set the pair memory

        Return:
            Whether the register operation is successful.
        """
        assert isinstance(vir_ptr, c_void_p) and isinstance(
            phy_ptr, c_void_p
        ), "clear memory pair only accept c_void_p type."
        LiteGlobal._api.LITE_register_memory_pair(
            vir_ptr, phy_ptr, length, device, backend
        )

    @staticmethod
    def clear_memory_pair(vir_ptr, phy_ptr, device, backend=LiteBackend.LITE_DEFAULT):
        """Clear the physical and virtual address pair in MegEngine.

        Args:
            vir_ptr: the virtual ptr to set to MegEngine.
            phy_ptr: the physical ptr to set to MegEngine.
            device: the the device to set the pair memory.
            backend: the backend to set the pair memory.

        Return:
            Whether the clear is operation successful.
        """
        assert isinstance(vir_ptr, c_void_p) and isinstance(
            phy_ptr, c_void_p
        ), "clear memory pair only accept c_void_p type."
        LiteGlobal._api.LITE_clear_memory_pair(vir_ptr, phy_ptr, device, backend)

    @staticmethod
    def lookup_physic_ptr(vir_ptr, device, backend=LiteBackend.LITE_DEFAULT):
        """Get the physic address by the virtual address in MegEngine.

        Args:
            vir_ptr: the virtual ptr to set to MegEngine.
            device: the the device to set the pair memory.
            backend: the backend to set the pair memory.

        Return:
            The physic address to lookup.
        """
        assert isinstance(
            vir_ptr, c_void_p
        ), "lookup physic ptr only accept c_void_p type."
        mem = c_void_p()
        LiteGlobal._api.LITE_lookup_physic_ptr(vir_ptr, byref(mem), device, backend)
        return mem
