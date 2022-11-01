from ._env_initlization import check_misc

# check misc as soon as possible
check_misc()

import atexit
import re
import sys

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
