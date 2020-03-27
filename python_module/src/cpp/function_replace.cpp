/**
 * \file python_module/src/cpp/function_replace.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief replace functions in megbrain core
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./megbrain_wrap.h"
#include "./python_helper.h"

#include "megbrain/utils/debug.h"
#include "megbrain/common.h"
#include "megbrain/system.h"

#include <stdexcept>
#include <cstring>
#include <cstdarg>

#include <Python.h>
#include <unistd.h>

namespace {

PyObject *logger = nullptr;

#if MGB_ENABLE_DEBUG_UTIL
void throw_fork_cuda_exc() {
    // set python error state, so when returning to parent process that calls
    // fork(), an exception could be raised
    //
    // call chain:
    // python -> fork() -> pthread_atfork -> CudaCheckOnFork ->
    // ForkAfterCudaError::throw_
    mgb_log_warn("try to raise python exception for fork after cuda");
    PyErr_SetString(PyExc_SystemError, "fork after cuda has been initialized");
}
#endif

class Init {
    static Init inst;
    Init() {
#if MGB_ENABLE_DEBUG_UTIL
        mgb::debug::ForkAfterCudaError::throw_ = throw_fork_cuda_exc;
#endif
    }
};
Init Init::inst;

int fork_exec_impl(const std::string &arg0, const std::string &arg1,
        const std::string &arg2) {
    auto pid = fork();
    if (!pid) {
        execl(arg0.c_str(), arg0.c_str(), arg1.c_str(), arg2.c_str(), nullptr);
        fprintf(stderr, "[megbrain] failed to execl %s [%s, %s]: %s\n",
                arg0.c_str(), arg1.c_str(), arg2.c_str(),
                std::strerror(errno));
        std::terminate();
    }
    mgb_assert(pid > 0, "failed to fork: %s", std::strerror(errno));
    return pid;
}

} // anonymous namespace

// called from swig/misc.i
void _timed_func_set_fork_exec_path(const char *arg0, const char *arg1) {
    using namespace std::placeholders;
    mgb::sys::TimedFuncInvoker::ins().set_fork_exec_impl(
            std::bind(fork_exec_impl, std::string{arg0}, std::string{arg1},
                _1));
}

void _timed_func_exec_cb(const char *user_data) {
    mgb::sys::TimedFuncInvoker::ins().fork_exec_impl_mainloop(user_data);
}

void _register_logger(PyObject *l) {
    logger = l;
}

namespace {
void py_log_handler(mgb::LogLevel level,
        const char *file, const char *func, int line, const char *fmt,
        va_list ap) {
    if (global_finalized()) {
        return;
    }

    using mgb::LogLevel;

    MGB_MARK_USED_VAR(file);
    MGB_MARK_USED_VAR(func);
    MGB_MARK_USED_VAR(line);

    if (!logger)
        return;

    PYTHON_GIL;

    const char *py_type;
    switch (level) {
        case LogLevel::DEBUG:
            py_type = "debug";
            break;
        case LogLevel::INFO:
            py_type = "info";
            break;
        case LogLevel::WARN:
            py_type = "warning";
            break;
        case LogLevel::ERROR:
            py_type = "error";
            break;
        default:
            throw std::runtime_error("bad log level");
    }

    std::string msg = mgb::svsprintf(fmt, ap);
    PyObject *py_msg = Py_BuildValue("s", msg.c_str());
    PyObject_CallMethod(logger,
            const_cast<char*>(py_type), const_cast<char*>("O"),
            py_msg);
    Py_DECREF(py_msg);
}

class LogHandlerSetter {
    static LogHandlerSetter ins;
    public:
        LogHandlerSetter() {
            mgb::set_log_handler(py_log_handler);
        }
};
LogHandlerSetter LogHandlerSetter::ins;
} // anobymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
