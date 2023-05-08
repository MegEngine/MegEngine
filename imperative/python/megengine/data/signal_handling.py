import signal

from ..core._imperative_rt.utils import (
    _check_any_worker_failed,
    _reset_worker_pids,
    _set_worker_pids,
    _set_worker_signal_handlers,
)

_sigchld_handler_set = False


def _set_sigchld_handler():
    global _sigchld_handler_set
    if _sigchld_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        previous_handler = None

    def handler(signum, frame):
        _check_any_worker_failed()
        if previous_handler is not None:
            assert callable(previous_handler)
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _sigchld_handler_set = True
