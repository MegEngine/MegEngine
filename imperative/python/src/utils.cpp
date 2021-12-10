/**
 * \file imperative/python/src/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "utils.h"
#ifdef WIN32
#include <stdio.h>
#include <windows.h>
#endif

#include <pybind11/operators.h>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include "./imperative_rt.h"
#include "megbrain/common.h"
#include "megbrain/comp_node.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/persistent_cache.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/tensor_sanity_check.h"
#include "megbrain/serialization/helper.h"
#include "megbrain/utils/persistent_cache.h"

#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/mm_handler.h"
#endif

namespace py = pybind11;

namespace {

bool g_global_finalized = false;

class LoggerWrapper {
public:
    using LogLevel = mgb::LogLevel;
    using LogHandler = mgb::LogHandler;
    static void set_log_handler(py::object logger_p) {
        logger = logger_p;
        mgb::set_log_handler(py_log_handler);
    }
    static LogLevel set_log_level(LogLevel log_level) {
        return mgb::set_log_level(log_level);
    }

private:
    static py::object logger;
    static void py_log_handler(
            mgb::LogLevel level, const char* file, const char* func, int line,
            const char* fmt, va_list ap) {
        using mgb::LogLevel;

        MGB_MARK_USED_VAR(file);
        MGB_MARK_USED_VAR(func);
        MGB_MARK_USED_VAR(line);

        if (g_global_finalized)
            return;

        const char* py_type;
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
        auto do_log = [msg = msg, py_type]() {
            if (logger.is_none())
                return;
            py::object _call = logger.attr(py_type);
            _call(py::str(msg));
        };
        if (PyGILState_Check()) {
            do_log();
        } else {
            py_task_q.add_task(do_log);
        }
    }
};
py::object LoggerWrapper::logger = py::none{};

uint32_t _get_dtype_num(py::object dtype) {
    return static_cast<uint32_t>(npy::dtype_np2mgb(dtype.ptr()).enumv());
}

py::bytes _get_serialized_dtype(py::object dtype) {
    std::string sdtype;
    auto write = [&sdtype](const void* data, size_t size) {
        auto pos = sdtype.size();
        sdtype.resize(pos + size);
        memcpy(&sdtype[pos], data, size);
    };
    mgb::serialization::serialize_dtype(npy::dtype_np2mgb(dtype.ptr()), write);
    return py::bytes(sdtype.data(), sdtype.size());
}

int fork_exec_impl(
        const std::string& arg0, const std::string& arg1, const std::string& arg2) {
#ifdef WIN32
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    auto args_str = " " + arg1 + " " + arg2;

    // Start the child process.
    if (!CreateProcess(
                arg0.c_str(),                         // exe name
                const_cast<char*>(args_str.c_str()),  // Command line
                NULL,                                 // Process handle not inheritable
                NULL,                                 // Thread handle not inheritable
                FALSE,                                // Set handle inheritance to FALSE
                0,                                    // No creation flags
                NULL,                                 // Use parent's environment block
                NULL,                                 // Use parent's starting directory
                &si,  // Pointer to STARTUPINFO structure
                &pi)  // Pointer to PROCESS_INFORMATION structure
    ) {
        mgb_log_warn("CreateProcess failed (%lu).\n", GetLastError());
        fprintf(stderr, "[megbrain] failed to execl %s [%s, %s]\n", arg0.c_str(),
                arg1.c_str(), arg2.c_str());
        __builtin_trap();
    }
    return pi.dwProcessId;
#else
    auto pid = fork();
    if (!pid) {
        execl(arg0.c_str(), arg0.c_str(), arg1.c_str(), arg2.c_str(), nullptr);
        fprintf(stderr, "[megbrain] failed to execl %s [%s, %s]: %s\n", arg0.c_str(),
                arg1.c_str(), arg2.c_str(), std::strerror(errno));
        std::terminate();
    }
    mgb_assert(pid > 0, "failed to fork: %s", std::strerror(errno));
    return pid;
#endif
}

}  // namespace

void init_utils(py::module m) {
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() { g_global_finalized = true; }));

    py::class_<std::atomic<uint64_t>>(m, "AtomicUint64")
            .def(py::init<>())
            .def(py::init<uint64_t>())
            .def("load", [](const std::atomic<uint64_t>& self) { return self.load(); })
            .def("store", [](std::atomic<uint64_t>& self,
                             uint64_t value) { return self.store(value); })
            .def("fetch_add", [](std::atomic<uint64_t>& self,
                                 uint64_t value) { return self.fetch_add(value); })
            .def("fetch_sub", [](std::atomic<uint64_t>& self,
                                 uint64_t value) { return self.fetch_sub(value); })
            .def(py::self += uint64_t())
            .def(py::self -= uint64_t());

    // FIXME!!! Should add a submodule instead of using a class for logger
    py::class_<LoggerWrapper> logger(m, "Logger");
    logger.def(py::init<>())
            .def_static("set_log_level", &LoggerWrapper::set_log_level)
            .def_static("set_log_handler", &LoggerWrapper::set_log_handler);

    py::enum_<LoggerWrapper::LogLevel>(logger, "LogLevel")
            .value("Debug", LoggerWrapper::LogLevel::DEBUG)
            .value("Info", LoggerWrapper::LogLevel::INFO)
            .value("Warn", LoggerWrapper::LogLevel::WARN)
            .value("Error", LoggerWrapper::LogLevel::ERROR);

    m.def("_get_dtype_num", &_get_dtype_num, "Convert numpy dtype to internal dtype");

    m.def("_get_serialized_dtype", &_get_serialized_dtype,
          "Convert numpy dtype to internal dtype for serialization");

    m.def("_get_device_count", &mgb::CompNode::get_device_count,
          "Get total number of specific devices on this system");

    m.def("_try_coalesce_all_free_memory", &mgb::CompNode::try_coalesce_all_free_memory,
          "This function will try it best to free all consecutive free chunks back to "
          "operating system");

    using mgb::imperative::TensorSanityCheck;
    py::class_<TensorSanityCheck>(m, "TensorSanityCheckImpl")
            .def(py::init<>())
            .def("enable",
                 [](TensorSanityCheck& checker) -> TensorSanityCheck& {
                     checker.enable();
                     return checker;
                 })
            .def("disable", [](TensorSanityCheck& checker) { checker.disable(); });

#if MGB_ENABLE_OPR_MM
    m.def("create_mm_server", &create_zmqrpc_server, py::arg("addr"),
          py::arg("port") = 0);
#else
    m.def("create_mm_server", []() {});
#endif

    // Debug code, internal only
    m.def("_defrag", [](const mgb::CompNode& cn) {
        mgb::imperative::BlobManager::inst()->defrag(cn);
    });
    m.def("_set_fork_exec_path_for_timed_func",
          [](const std::string& arg0, const ::std::string arg1) {
              using namespace std::placeholders;
              mgb::sys::TimedFuncInvoker::ins().set_fork_exec_impl(std::bind(
                      fork_exec_impl, std::string{arg0}, std::string{arg1}, _1));
          });
    m.def("_timed_func_exec_cb", [](const std::string& user_data) {
        mgb::sys::TimedFuncInvoker::ins().fork_exec_impl_mainloop(user_data.c_str());
    });

    using PersistentCache = mgb::PersistentCache;
    using ExtendedPersistentCache =
            mgb::imperative::persistent_cache::ExtendedPersistentCache;

    struct PersistentCacheManager {
        std::shared_ptr<ExtendedPersistentCache> instance;

        bool try_reg(std::shared_ptr<ExtendedPersistentCache> cache) {
            if (cache) {
                instance = cache;
                PersistentCache::set_impl(cache);
                return true;
            }
            return false;
        }
        bool open_file(std::string path) {
            return try_reg(mgb::imperative::persistent_cache::make_in_file(path));
        }
        std::optional<size_t> clean() {
            if (instance) {
                return instance->clear();
            }
            return {};
        }
        void put(std::string category, std::string key, std::string value) {
            PersistentCache::inst().put(
                    category, {key.data(), key.size()}, {value.data(), value.size()});
        }
        py::object get(std::string category, std::string key) {
            auto value =
                    PersistentCache::inst().get(category, {key.data(), key.size()});
            if (value.valid()) {
                return py::bytes(std::string((const char*)value->ptr, value->size));
            } else {
                return py::none();
            }
        }
    };

    py::class_<PersistentCacheManager>(m, "PersistentCacheManager")
            .def(py::init<>())
            .def("try_open_file", &PersistentCacheManager::open_file)
            .def("clean", &PersistentCacheManager::clean)
            .def("put", &PersistentCacheManager::put)
            .def("get", &PersistentCacheManager::get);
}
