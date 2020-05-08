/**
 * \file src/core/include/megbrain/comp_node_env.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/common.h"
#include "megbrain/comp_node.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thread.h"
#include "megbrain_build_config.h"

#include "megdnn/handle.h"


#if MGB_CUDA
#include <cuda_runtime.h>

#if MGB_ENABLE_LOGGING
#define MGB_CUDA_CHECK(expr)                                          \
    do {                                                              \
        cudaError_t __cuda_check_code = (expr);                       \
        if (!mgb_likely(__cuda_check_code == cudaSuccess)) {          \
            ::mgb::_on_cuda_error(#expr, __cuda_check_code, __FILE__, \
                                  __func__, __LINE__);                \
        }                                                             \
    } while (0)
#else
#define MGB_CUDA_CHECK(expr)                                            \
    do {                                                                \
        cudaError_t __cuda_check_code = (expr);                         \
        if (!mgb_likely(__cuda_check_code == cudaSuccess)) {            \
            ::mgb::_on_cuda_error(#expr, __cuda_check_code, "", "", 1); \
        }                                                               \
    } while (0)

#endif  // MGB_ENABLE_LOGGING

#endif

//! whether to enable asynchronous initialization for CompNode and CompNodeEnv
#define MGB_ENABLE_COMP_NODE_ASYNC_INIT (MGB_CUDA)

//! whether AsyncErrorInfo is needed
#define MGB_NEED_MEGDNN_ASYNC_ERROR (MGB_CUDA)

#if MGB_ENABLE_COMP_NODE_ASYNC_INIT
#include <atomic>
#include <future>
#endif

#include <memory>
#include <type_traits>
#include "megbrain/utils/thin/function.h"

namespace mgb {

#if MGB_CUDA
[[noreturn]] void _on_cuda_error(const char* expr, cudaError_t err,
                                 const char* file, const char* func, int line);
#endif


class CPUDispatcher : public MegcoreCPUDispatcher {
public:
    using AffinityCallBack = thin_function<void(size_t)>;
    //! get number of tasks already dispatched
    virtual size_t get_nr_dispatched_tasks() const = 0;
    //! set the cpu affinity callback, the callback is
    //! thin_function<void(size_t)>
    virtual void set_affinity(AffinityCallBack&& /*affinity_cb*/) {
        mgb_assert(0, "The CompNode set_affinity is not implement");
    }
};


/*!
 * \brief CompNode environment
 *
 * CompNodeEnv contains necessary information to launch a kernel on a comp node,
 * or calling other libraries on a comp node. It has common fields for all comp
 * nodes and also specific fields for a given comp node type.
 *
 * Each CompNode is associated with a CompNodeEnv that could be retrieved by
 * CompNodeEnv::from_comp_node.
 *
 * Note: CUDA CompNodeEnv is initialized asynchronously. The env and property is
 * set synchronously, but m_lib_handle_manager would be initialized in the
 * future.
 */
class CompNodeEnv final : public NonCopyableObj {
public:
    using DeviceType = CompNode::DeviceType;
    using MemEventHandler =
            thin_function<void(size_t alloc_size, bool is_host, void* ptr)>;

    //! extra properties for a CompNodeEnv
    struct Property {
        //! type of the underlying device
        DeviceType type;

        //! alignment requirement in bytes, for memory allocating
        size_t mem_alignment = 0;
    };

    //! get user data by calling UserDataContainer::get_user_data_or_create;
    //! this method is thread-safe
    template <typename T, typename Maker>
    T& get_user_data(Maker&& maker) const {
        ensure_async_init_finished();
        MGB_LOCK_GUARD(m_user_data_container_mtx);
        return *m_user_data_container->get_user_data_or_create<T>(
                std::forward<Maker>(maker));
    }

    template <typename T>
    T& get_user_data() const {
        ensure_async_init_finished();
        MGB_LOCK_GUARD(m_user_data_container_mtx);
        return *m_user_data_container->get_user_data_or_create<T>(
                std::make_shared<T>);
    }

    //! check whether a user data object has been registered
    template <typename T>
    bool has_user_data() const {
        ensure_async_init_finished();
        MGB_LOCK_GUARD(m_user_data_container_mtx);
        return m_user_data_container->get_user_data<T>().second;
    }

    //! get property
    const Property& property() const { return m_property; }

    //! get the comp node to which this env belongs
    CompNode comp_node() const { return m_comp_node; }

    /*!
     * \brief create CompNodeEnv from comp_node
     */
    static inline const CompNodeEnv& from_comp_node(const CompNode& node);

    /*!
     * \brief activate this env for current thread
     *
     * Currently only calls cuda_env().activate() if type is cuda
     */
    void activate() const {
#if MGB_CUDA
        if (m_property.type == DeviceType::CUDA) {
            m_cuda_env.activate();
        }
#endif
    }

    /*!
     * \brief set a callback to be invoked on alloc/free events
     * \param[in,out] handler the new handler to be set; the previous handler
     *      would be returned
     */
    void mem_event_handler(MemEventHandler& handler) {
        m_mem_event_handler.swap(handler);
    }

    //! invoke mem event handler on a mem event; only be called from CompNode
    void on_mem_event(size_t alloc_size, bool is_host, void* ptr) {
        if (m_mem_event_handler) {
            m_mem_event_handler(alloc_size, is_host, ptr);
        }
    }

        // following are impls for various envs

#if MGB_CUDA
    struct CudaEnv {
        int device = -1;
        cudaStream_t stream = 0;
        cudaDeviceProp device_prop;

        void activate() const { MGB_CUDA_CHECK(cudaSetDevice(device)); }
    };

    const CudaEnv& cuda_env() const {
        if (mgb_unlikely(m_property.type != DeviceType::CUDA))
            on_bad_device_type(DeviceType::CUDA);
        ensure_async_init_finished();
        return m_cuda_env;
    }

    //! init this as a cuda env asynchronously
    void init_cuda_async(int dev, CompNode comp_node,
                         const ContinuationCtx<cudaStream_t>& cont);
#endif


    struct CpuEnv {
        using Task = CPUDispatcher::Task;
        using MultiThreadingTask = CPUDispatcher::MultiThreadingTask;
        using AffinityCallBack = thin_function<void(size_t)>;

        std::shared_ptr<CPUDispatcher> dispatcher;

        void dispatch(Task&& task) const {
            dispatcher->dispatch(std::move(task));
        }

        void dispatch(MultiThreadingTask&& task, size_t parallelism) const {
            dispatcher->dispatch(std::move(task), parallelism);
        }

        void set_affinity(AffinityCallBack&& cb) const {
            dispatcher->set_affinity(std::move(cb));
        }
    };

    const CpuEnv& cpu_env() const {
        if (mgb_unlikely(m_property.type != DeviceType::CPU))
            on_bad_device_type(DeviceType::CPU);
        return m_cpu_env;
    }

    //! init this as a cpu env
    void init_cpu(const CpuEnv& env, CompNode comp_node);

    void fini();

private:
    CompNode m_comp_node;
    Property m_property;
    MemEventHandler m_mem_event_handler;

#if MGB_CUDA
    CudaEnv m_cuda_env;
#endif
    CpuEnv m_cpu_env;

    std::unique_ptr<UserDataContainer> m_user_data_container;
    mutable RecursiveSpinlock m_user_data_container_mtx;

    [[noreturn]] void on_bad_device_type(DeviceType expected) const;

#if MGB_ENABLE_COMP_NODE_ASYNC_INIT
    //! whether async init is in future; set by init*_async methods
    std::atomic_bool m_async_init_need_wait{false};
    std::mutex m_async_init_mtx;
    std::future<void> m_async_init_future;
    std::thread::id m_async_init_tid;

    void ensure_async_init_finished() const {
        if (m_async_init_need_wait.load()) {
            const_cast<CompNodeEnv*>(this)->wait_async_init();
        }
    }

    void wait_async_init();
#else
    void ensure_async_init_finished() const {}
#endif
};

//! megdnn handle stored in a CompNodeEnv
class MegDNNHandle final : public UserDataContainer::UserData,
                           public std::enable_shared_from_this<MegDNNHandle> {
    MGB_TYPEINFO_OBJ_DECL;

    static int sm_default_dbg_level;
    megcoreDeviceHandle_t m_dev_hdl = nullptr;
    megcoreComputingHandle_t m_comp_hdl = nullptr;
    std::unique_ptr<megdnn::Handle> m_megdnn_handle;

#if MGB_NEED_MEGDNN_ASYNC_ERROR
    std::shared_ptr<megcore::AsyncErrorInfo> m_async_error_info_devptr;
    megcore::AsyncErrorInfo* make_async_error_info(const CompNodeEnv& env);
#endif

public:
    MegDNNHandle(const CompNodeEnv& env);
    ~MegDNNHandle() noexcept;

    static MegDNNHandle& get(const CompNodeEnv& env);

    megdnn::Handle* operator->() const { return handle(); }

    megdnn::Handle* handle() const { return m_megdnn_handle.get(); }

    //! set the default debug level; return original setting
    static int exchange_default_dbg_level(int level) {
        auto ret = sm_default_dbg_level;
        sm_default_dbg_level = level;
        return ret;
    }

#if MGB_NEED_MEGDNN_ASYNC_ERROR
    /*!
     * \brief get pointer to underlying AsyncErrorInfo
     *
     * return nullptr if the device does not need async error report.
     */
    megcore::AsyncErrorInfo* async_error_info_devptr() const {
        return m_async_error_info_devptr.get();
    }
#endif
};

class CompNode::Impl : public CompNode::ImplBase {
protected:
    CompNodeEnv m_env;

    using ImplBase::ImplBase;
    ~Impl() = default;

public:
    CompNodeEnv& env() { return m_env; }
};

const CompNodeEnv& CompNodeEnv::from_comp_node(const CompNode& node) {
    mgb_assert(node.valid());
    return static_cast<CompNode::Impl*>(node.m_impl)->env();
}

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
