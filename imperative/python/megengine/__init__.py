# -*- coding: utf-8 -*-
import atexit
import ctypes
import re
import os
import platform
import sys

if os.getenv("TERMUX_VERSION"):
    try:
        import cv2
    except Exception as exc:
        print("Run MegEngine python interface at Android/Termux env")
        print("!!!You need build opencv-python manually!!!, by run sh:")
        print(
            "https://github.com/MegEngine/MegEngine/blob/master/scripts/whl/android/android_opencv_python.sh"
        )
        raise exc

if sys.platform == "win32":
    lib_path = os.path.join(os.path.dirname(__file__), "core/lib")
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

from .core._imperative_rt.core2 import close as _close
from .core._imperative_rt.core2 import full_sync as _full_sync
from .core._imperative_rt.core2 import sync as _sync
from .core._imperative_rt.common import (
    get_supported_sm_versions as _get_supported_sm_versions,
)
from .config import *
from .device import *
from .logger import enable_debug_log, get_logger, set_log_file, set_log_level
from .serialization import load, save
from .tensor import Parameter, Tensor, tensor
from .utils import comp_graph_tools as cgtools
from .utils.persistent_cache import PersistentCacheOnServer as _PersistentCacheOnServer
from .version import __version__


_exit_handlers = []


def _run_exit_handlers():
    for handler in reversed(_exit_handlers):
        handler()
    _exit_handlers.clear()


atexit.register(_run_exit_handlers)


def _exit(code):
    _run_exit_handlers()
    sys.exit(code)


def _atexit(handler):
    _exit_handlers.append(handler)


_atexit(_close)

_persistent_cache = _PersistentCacheOnServer()
_persistent_cache.reg()

_atexit(_persistent_cache.flush)

# subpackages
import megengine.amp
import megengine.autodiff
import megengine.config
import megengine.data
import megengine.distributed
import megengine.dtr
import megengine.functional
import megengine.hub
import megengine.jit
import megengine.module
import megengine.optimizer
import megengine.quantization
import megengine.random
import megengine.utils
import megengine.traced_module
