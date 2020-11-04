/**
 * \file src/core/impl/exception.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/exception.h"
#include "megbrain/common.h"
#include "megbrain/utils/debug.h"
#include "megbrain/comp_node_env.h"

using namespace mgb;

namespace {
    class MegDNNErrorHandler final: public megdnn::ErrorHandler {
        static MegDNNErrorHandler inst;
        void do_on_megdnn_error(const std::string &msg) override {
            mgb_throw_raw(MegDNNError{msg});
        }

        void do_on_tensor_reshape_error(const std::string &msg) override {
            mgb_throw_raw(TensorReshapeError{msg});
        }

        public:
            MegDNNErrorHandler() {
                set_handler(this);
            }
    };
    MegDNNErrorHandler MegDNNErrorHandler::inst;
}

void MegBrainError::init()
{
    m_msg.append("\n");
#if MGB_ENABLE_DEBUG_UTIL
    debug::backtrace(2).fmt_to_str(m_msg);
    static bool print_exc = MGB_GETENV("MGB_PRINT_EXC");
    if (print_exc) {
        fprintf(stderr, "mgb: exception occurred: %s\n", m_msg.c_str());
    }
#endif
}

CudaError::CudaError(const std::string &msg):
    SystemError(msg)
{
    m_msg.append(get_cuda_extra_info());
}

std::string CudaError::get_cuda_extra_info() {
#if MGB_CUDA
    // get last error and clear error
    auto err = cudaGetLastError();
    int dev = -1;
    cudaGetDevice(&dev);
    size_t free_byte = 0, total_byte = 0;
    cudaMemGetInfo(&free_byte, &total_byte);
    constexpr double SIZE2MB = 1.0 / 1024 / 1024;
    return ssprintf("(last_err=%d(%s) "
            "device=%d mem_free=%.3fMiB mem_tot=%.3fMiB)",
            err, cudaGetErrorString(err),
            dev, free_byte * SIZE2MB, total_byte * SIZE2MB);
#else
    return "cuda disabled at compile time";
#endif
}

AtlasError::AtlasError(const std::string &msg):
    SystemError(msg)
{
}



ROCmError::ROCmError(const std::string &msg):
    SystemError(msg)
{
    m_msg.append(get_rocm_extra_info());
}

std::string ROCmError::get_rocm_extra_info() {
#if MGB_ROCM
    // get last error and clear error
    auto err = hipGetLastError();
    int dev = -1;
    hipGetDevice(&dev);
    size_t free_byte = 0, total_byte = 0;
    hipMemGetInfo(&free_byte, &total_byte);
    constexpr double SIZE2MB = 1.0 / 1024 / 1024;
    return ssprintf("(last_err=%d(%s) "
            "device=%d mem_free=%.3fMiB mem_tot=%.3fMiB)",
            err, hipGetErrorString(err),
            dev, free_byte * SIZE2MB, total_byte * SIZE2MB);
#else
    return "rocm disabled at compile time";
#endif
}

CnrtError::CnrtError(const std::string& msg) : SystemError(msg) {
    m_msg.append(get_cnrt_extra_info());
}

std::string CnrtError::get_cnrt_extra_info() {
#if MGB_CAMBRICON
    // get last error
    auto err = cnrtGetLastErr();
    return ssprintf("(last_err=%d(%s))", err, cnrtGetErrorStr(err));
#else
    return "cnrt disabled at compile time";
#endif
}

CndevError::CndevError(const std::string& msg) : SystemError(msg) {}

CnmlError::CnmlError(const std::string& msg) : SystemError(msg) {}

bool mgb::has_uncaught_exception() {
#if MGB_ENABLE_EXCEPTION
#if __cplusplus > 201402L
    // C++17; see https://stackoverflow.com/questions/38456127/what-is-the-value-of-cplusplus-for-c17
    return std::uncaught_exceptions() != 0;
#else
    return std::uncaught_exception();
#endif
#else
    return false;
#endif
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

