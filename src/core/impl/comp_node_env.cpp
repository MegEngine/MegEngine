/**
 * \file src/core/impl/comp_node_env.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/comp_node_env.h"
#include "megbrain/exception.h"
#include "megbrain/system.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/version_symbol.h"

#include "megdnn/version.h"
#if MGB_CUDA
#include "megcore_cuda.h"
#if MGB_ENABLE_DEBUG_UTIL
#include <nvToolsExtCudaRt.h>
#endif
#endif


using namespace mgb;

/* =================== MegDNNHandle =================== */
MGB_TYPEINFO_OBJ_IMPL(MegDNNHandle);

int MegDNNHandle::sm_default_dbg_level = 0;

MegDNNHandle& MegDNNHandle::get(const CompNodeEnv& env) {
    auto maker = [&]() { return std::make_shared<MegDNNHandle>(env); };
    return env.get_user_data<MegDNNHandle>(maker);
}

MegDNNHandle::MegDNNHandle(const CompNodeEnv& env) {
    auto megdnn_version = megdnn::get_version();
    mgb_throw_if(
            megdnn_version.major != MEGDNN_MAJOR ||
                    megdnn_version.minor < MEGDNN_MINOR,
            SystemError,
            "incompatible megdnn version: compiled with %d.%d, get %d.%d.%d "
            "at runtime",
            MEGDNN_MAJOR, MEGDNN_MINOR, megdnn_version.major,
            megdnn_version.minor, megdnn_version.patch);
    bool init = false;
#if MGB_CUDA
    if (env.property().type == CompNode::DeviceType::CUDA) {
        megcoreCreateDeviceHandle(&m_dev_hdl, megcorePlatformCUDA,
                                  env.cuda_env().device, 0);
        megcore::createComputingHandleWithCUDAContext(&m_comp_hdl, m_dev_hdl, 0,
                {env.cuda_env().stream, make_async_error_info(env)});
        init = true;
    }
#endif

    if (env.property().type == CompNode::DeviceType::CPU) {
        megcoreCreateDeviceHandle(&m_dev_hdl, megcorePlatformCPU);
        megcoreCreateComputingHandleWithCPUDispatcher(&m_comp_hdl, m_dev_hdl,
                                                      env.cpu_env().dispatcher);
        init = true;
    }

    mgb_assert(init);
    int level = sm_default_dbg_level;
    if (auto set = MGB_GETENV("MGB_USE_MEGDNN_DBG")) {
        level = std::stol(set);
        mgb_log_warn("use megdnn handle with debug level: %d", level);
    }
    // handle may have been implemented when device type is cadence.
    if (!m_megdnn_handle) {
        m_megdnn_handle = megdnn::Handle::make(m_comp_hdl, level);
    }
}

MegDNNHandle::~MegDNNHandle() noexcept {
    m_megdnn_handle.reset();
#if MGB_NEED_MEGDNN_ASYNC_ERROR
    m_async_error_info_devptr.reset();
#endif
    if (m_comp_hdl) {
        megcoreDestroyComputingHandle(m_comp_hdl);
    }
    if (m_dev_hdl) {
        megcoreDestroyDeviceHandle(m_dev_hdl);
    }
}

#if MGB_NEED_MEGDNN_ASYNC_ERROR
megcore::AsyncErrorInfo* MegDNNHandle::make_async_error_info(
        const CompNodeEnv& env) {
    auto cn = env.comp_node();
    auto del = [cn](megcore::AsyncErrorInfo* ptr) {
        if (ptr) {
            cn.free_device(ptr);
        }
    };
    megcore::AsyncErrorInfo zero_info{0, nullptr, "", {0,0,0,0}};
    auto ptr = static_cast<megcore::AsyncErrorInfo*>(
            env.comp_node().alloc_device(sizeof(zero_info)));
    cn.copy_to_device(ptr, &zero_info, sizeof(zero_info));
    cn.sync();
    m_async_error_info_devptr = {ptr, del};
    return m_async_error_info_devptr.get();
}
#endif

/* =================== misc =================== */

#if MGB_CUDA

void mgb::_on_cuda_error(const char* expr, cudaError_t err, const char* file,
                         const char* func, int line) {
    mgb_throw(CudaError, "cuda error %d: %s (%s at %s:%s:%d)", int(err),
              cudaGetErrorString(err), expr, file, func, line);
}

void CompNodeEnv::init_cuda_async(int dev, CompNode comp_node,
                                  const ContinuationCtx<cudaStream_t>& cont) {
    m_comp_node = comp_node;

    mgb_assert(!m_user_data_container && !m_async_init_need_wait);
    m_cuda_env.device = dev;
    m_property.type = DeviceType::CUDA;
    MGB_CUDA_CHECK(cudaGetDeviceProperties(&m_cuda_env.device_prop, dev));
    {
        auto&& prop = m_cuda_env.device_prop;
        m_property.mem_alignment =
                std::max(prop.textureAlignment, prop.texturePitchAlignment);
    }

    std::atomic_bool tid_set{false};
    auto worker = [this, cont, &tid_set]() {
        sys::set_thread_name("async_cuda_init");
        m_async_init_tid = std::this_thread::get_id();
        tid_set.store(true);
        bool stream_done = false;
        MGB_MARK_USED_VAR(stream_done);
        MGB_TRY {
            m_cuda_env.activate();
            MGB_CUDA_CHECK(cudaStreamCreateWithFlags(&m_cuda_env.stream,
                                                     cudaStreamNonBlocking));
            stream_done = true;

            m_user_data_container = std::make_unique<UserDataContainer>();

#if MGB_ENABLE_DEBUG_UTIL
            nvtxNameCudaStreamA(m_cuda_env.stream,
                                m_comp_node.to_string().c_str());
#endif
            cont.next(m_cuda_env.stream);

            // megdnn is initialized here; must be placed after cont.next()
            // which handles comp node init
            mgb_assert(
                    m_property.mem_alignment ==
                    MegDNNHandle::get(*this).handle()->alignment_requirement());
        }
        MGB_CATCH(std::exception & exc, {
            mgb_log_error("async cuda init failed: %s", exc.what());
            if (stream_done) {
                cudaStreamDestroy(m_cuda_env.stream);
            }
            cont.err(exc);
            throw;
        })
    };

    m_async_init_need_wait = true;
    m_async_init_future = std::async(std::launch::async, worker);
    while (!tid_set.load())
        std::this_thread::yield();
    mgb_assert(m_async_init_tid != std::this_thread::get_id());
}
#endif


void CompNodeEnv::init_cpu(const CpuEnv& env, CompNode comp_node) {
    m_comp_node = comp_node;

    mgb_assert(!m_user_data_container);
    m_property.type = DeviceType::CPU;
    m_cpu_env = env;
    m_user_data_container = std::make_unique<UserDataContainer>();
    m_property.mem_alignment =
            MegDNNHandle::get(*this).handle()->alignment_requirement();
}


void CompNodeEnv::fini() {
    ensure_async_init_finished();
    m_user_data_container.reset();
#if MGB_CUDA
    if (m_property.type == DeviceType::CUDA) {
        m_cuda_env.activate();
        MGB_CUDA_CHECK(cudaStreamDestroy(m_cuda_env.stream));
    }
#endif
}

#if MGB_ENABLE_COMP_NODE_ASYNC_INIT
void CompNodeEnv::wait_async_init() {
    if (std::this_thread::get_id() == m_async_init_tid)
        return;

    MGB_LOCK_GUARD(m_async_init_mtx);
    if (m_async_init_need_wait.load()) {
        m_async_init_future.wait();
        m_async_init_need_wait.store(false);
        m_async_init_future.get();
    }
}
#endif

void CompNodeEnv::on_bad_device_type(DeviceType expected) const {
    mgb_throw(MegBrainError, "bad device type: expected=%d actual=%d",
              static_cast<int>(expected), static_cast<int>(m_property.type));
}

MGB_VERSION_SYMBOL3(MEGDNN, MEGDNN_MAJOR, MEGDNN_MINOR, MEGDNN_PATCH);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
