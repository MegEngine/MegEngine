# -*- coding: utf-8 -*-

import ctypes
import glob
import logging
import os
import sys
from ctypes import *

if sys.platform == "win32":
    lib_path = os.path.join(os.path.dirname(__file__), "../megengine/core/lib")
    dll_paths = list(filter(os.path.exists, [lib_path,]))
    assert len(dll_paths) > 0

    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    has_load_library_attr = hasattr(kernel32, "AddDllDirectory")
    old_error_mode = kernel32.SetErrorMode(0x0001)

    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    if has_load_library_attr:
        kernel32.AddDllDirectory.restype = ctypes.c_void_p
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    for dll_path in dll_paths:
        if sys.version_info >= (3, 8):
            os.add_dll_directory(dll_path)
        elif has_load_library_attr:
            res = kernel32.AddDllDirectory(dll_path)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += ' Error adding "{}" to the DLL search PATH.'.format(
                    dll_path
                )
                raise err
        else:
            print("WARN: python or OS env have some issue, may load DLL failed!!!")

    import glob

    dlls = glob.glob(os.path.join(lib_path, "*.dll"))
    path_patched = False
    for dll in dlls:
        is_loaded = False
        if has_load_library_attr:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res is None and last_error != 126:
                err = ctypes.WinError(last_error)
                err.strerror += ' Error loading "{}" or one of its dependencies.'.format(
                    dll
                )
                err.strerror += " \nplease install VC runtime from: "
                err.strerror += " \nhttps://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160"
                raise err
            elif res is not None:
                is_loaded = True
        if not is_loaded:
            if not path_patched:
                os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                path_patched = True
            res = kernel32.LoadLibraryW(dll)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += ' Error loading "{}" or one of its dependencies.'.format(
                    dll
                )
                err.strerror += " \nplease install VC runtime from: "
                err.strerror += " \nhttps://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160"
                raise err

    kernel32.SetErrorMode(old_error_mode)


class _LiteCLib:
    def __init__(self):
        cwd = os.getcwd()
        package_dir = os.path.dirname(os.path.realpath(__file__))
        debug_path = os.getenv("LITE_LIB_PATH")
        os.chdir(package_dir)
        lite_libs = glob.glob("libs/liblite*")
        os.chdir(cwd)

        if debug_path is None:
            assert len(lite_libs) == 1
            self._lib = CDLL(os.path.join(package_dir, lite_libs[0]))
        else:
            self._lib = CDLL(debug_path)
        self._register_api(
            "LITE_get_version", [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        )
        self.lib.LITE_get_version.restype = None
        self._register_api("LITE_set_log_level", [c_int])
        self._register_api("LITE_get_log_level", [])
        self._register_api("LITE_get_last_error", [], False)
        self.lib.LITE_get_last_error.restype = c_char_p

    def _errcheck(self, result, func, args):
        if result:
            error = self.lib.LITE_get_last_error()
            msg = error.decode("utf-8")
            logging.error("{}".format(msg))
            raise RuntimeError("{}".format(msg))
        return result

    def _register_api(self, api_name, arg_types, error_check=True):
        func = getattr(self.lib, api_name)
        func.argtypes = arg_types
        func.restype = c_int
        if error_check:
            func.errcheck = self._errcheck

    @property
    def lib(self):
        return self._lib

    @property
    def version(self):
        major = c_int()
        minor = c_int()
        patch = c_int()
        self.lib.LITE_get_version(byref(major), byref(minor), byref(patch))
        return "{}.{}.{}".format(major.value, minor.value, patch.value)

    def set_log_level(self, level):
        self.lib.LITE_set_log_level(level)

    def get_log_level(self):
        return self.lib.LITE_get_log_level()


_lib = _LiteCLib()
version = _lib.version
set_log_level = _lib.set_log_level
get_log_level = _lib.get_log_level

_Cnetwork = c_void_p
_Ctensor = c_void_p


class _LiteCObjMetaClass(type):
    """metaclass for lite object"""

    def __new__(cls, name, bases, attrs):
        for api in attrs["_api_"]:
            _lib._register_api(*api)
        del attrs["_api_"]
        attrs["_lib"] = _lib.lib
        return super().__new__(cls, name, bases, attrs)


class _LiteCObjBase(metaclass=_LiteCObjMetaClass):
    _api_ = []
