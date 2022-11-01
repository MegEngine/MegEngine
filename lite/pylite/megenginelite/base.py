# -*- coding: utf-8 -*-

import ctypes
import glob
import logging
import os
import sys
from ctypes import *

from ._env_initlization import check_misc

# check misc as soon as possible
check_misc()


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
