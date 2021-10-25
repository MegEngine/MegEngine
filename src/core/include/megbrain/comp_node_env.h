/**
 * \file src/core/include/megbrain/comp_node_env.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>

#if MGB_ENABLE_LOGGING
#define MGB_CUDA_CHECK(expr)                                                 \
    do {                                                                     \
        cudaError_t __cuda_check_code = (expr);                              \
        if (!mgb_likely(__cuda_check_code == cudaSuccess)) {                 \
            ::mgb::_on_cuda_error(                                           \
                    #expr, __cuda_check_code, __FILE__, __func__, __LINE__); \
        }                                                                    \
    } while (0)

#define MGB_CUDA_CU_CHECK(expr)                                              \
    do {                                                                     \
        CUresult __cuda_check_code = (expr);                                 \
        if (!mgb_likely(__cuda_check_code == CUDA_SUCCESS)) {                \
            ::mgb::_on_cuda_cu_error(                                        \
                    #expr, __cuda_check_code, __FILE__, __func__, __LINE__); \
        }                                                                    \
    } while (0)

#else
#define MGB_CUDA_CHECK(expr)                                            \
    do {                                                                \
        cudaError_t __cuda_check_code = (expr);                         \
        if (!mgb_likely(__cuda_check_code == cudaSuccess)) {            \
            ::mgb::_on_cuda_error(#expr, __cuda_check_code, "", "", 1); \
        }                                                               \
    } while (0)

#define MGB_CUDA_CU_CHECK(expr)                                            \
    do {                                                                   \
        CUresult __cuda_check_code = (expr);                               \
        if (!mgb_likely(__cuda_check_code == CUDA_SUCCESS)) {              \
            ::mgb::_on_cuda_cu_error(#expr, __cuda_check_code, "", "", 1); \
        }                                                                  \
    } while (0)

#endif  // MGB_ENABLE_LOGGING
#endif  // MGB_CUDA

#if MGB_ATLAS
#include <atomic>
#include "megcore_atlas.h"

#if MGB_ENABLE_LOGGING
#define MGB_ATLAS_CHECK(expr)                                               \
    do {                                                                    \
        aclError __acl_check_code = (expr);                                 \
        if (!mgb_likely(__acl_check_code == ACL_ERROR_NONE)) {              \
            ::mgb::_on_atlas_error(                                         \
                    #expr, __acl_check_code, __FILE__, __func__, __LINE__); \
        }                                                                   \
    } while (0)
#else
#define MGB_ATLAS_CHECK(expr)                                           \
    do {                                                                \
        aclError __acl_check_code = (expr);                             \
        if (!mgb_likely(__acl_check_code == ACL_ERROR_NONE)) {          \
            ::mgb::_on_atlas_error(#expr, __acl_check_code, "", "", 1); \
        }                                                               \
    } while (0)

#endif  // MGB_ENABLE_LOGGING

#endif  // MGB_ATLAS

#if MGB_ROCM
#include "hcc_detail/hcc_defs_prologue.h"
#include "megcore_rocm.h"

#if MGB_ENABLE_LOGGING
#define MGB_ROCM_CHECK(expr)                                                \
    do {                                                                    \
        hipError_t __hip_check_code = (expr);                               \
        if (!mgb_likely(__hip_check_code == hipSuccess)) {                  \
            ::mgb::_on_hip_error(                                           \
                    #expr, __hip_check_code, __FILE__, __func__, __LINE__); \
        }                                                                   \
    } while (0)
#else
#define MGB_ROCM_CHECK(expr)                                          \
    do {                                                              \
        hipError_t __hip_check_code = (expr);                         \
        if (!mgb_likely(__hip_check_code == hipSuccess)) {            \
            ::mgb::_on_hip_error(#expr, __hip_check_code, "", "", 1); \
        }                                                             \
    } while (0)

#endif  // MGB_ENABLE_LOGGING

#endif

#if MGB_CAMBRICON
#include <cndev.h>
#include <cnml.h>
#include <cnrt.h>

#if MGB_ENABLE_LOGGING
#define MGB_CNRT_CHECK(expr)                                                 \
    do {                                                                     \
        cnrtRet_t __cnrt_check_code = (expr);                                \
        if (mgb_unlikely(__cnrt_check_code != CNRT_RET_SUCCESS)) {           \
            ::mgb::_on_cnrt_error(                                           \
                    #expr, __cnrt_check_code, __FILE__, __func__, __LINE__); \
        }                                                                    \
    } while (0)
#define MGB_CNDEV_CHECK(expr)                                                 \
    do {                                                                      \
        cndevRet_t __cndev_check_code = (expr);                               \
        if (mgb_unlikely(__cndev_check_code != CNDEV_SUCCESS)) {              \
            ::mgb::_on_cndev_error(                                           \
                    #expr, __cndev_check_code, __FILE__, __func__, __LINE__); \
        }                                                                     \
    } while (0)
#define MGB_CNML_CHECK(expr)                                                 \
    do {                                                                     \
        cnmlStatus_t __cnml_check_code = (expr);                             \
        if (mgb_unlikely(__cnml_check_code != CNML_STATUS_SUCCESS)) {        \
            ::mgb::_on_cnml_error(                                           \
                    #expr, __cnml_check_code, __FILE__, __func__, __LINE__); \
        }                                                                    \
    } while (0)
#else
#define MGB_CNRT_CHECK(expr)                                            \
    do {                                                                \
        cnrtRet_t __cnrt_check_code = (expr);                           \
        if (mgb_unlikely(__cnrt_check_code != CNRT_RET_SUCCESS)) {      \
            ::mgb::_on_cnrt_error(#expr, __cnrt_check_code, "", "", 1); \
        }                                                               \
    } while (0)
#define MGB_CNDEV_CHECK(expr)                                                       \
    do {                                                                            \
        cndevRet_t __cndev_check_code = (expr);                                     \
        if (mgb_unlikely(__cndev_check_code != CNDEV_SUCCESS)) {                    \
            ::mgb::_on_cndev_error(#expr, __cndev_check_code, __FILE__, "", "", 1); \
        }                                                                           \
    } while (0)
#define MGB_CNML_CHECK(expr)                                                      \
    do {                                                                          \
        cnmlStatus_t __cnml_check_code = (expr);                                  \
        if (mgb_unlikely(__cnml_check_code != CNML_STATUS_SUCCESS)) {             \
            ::mgb::_on_cnml_error(#expr, __cnml_check_code, __FILE__, "", "", 1); \
        }                                                                         \
    } while (0)
#endif  // MGB_ENABLE_LOGGING
#endif  // MGB_CAMBRICON

//! whether to enable asynchronous initialization for CompNode and CompNodeEnv
#define MGB_ENABLE_COMP_NODE_ASYNC_INIT (MGB_CUDA || MGB_ROCM)

//! whether AsyncErrorInfo is needed
#define MGB_NEED_MEGDNN_ASYNC_ERROR (MGB_CUDA || MGB_ROCM)

#if MGB_ENABLE_COMP_NODE_ASYNC_INIT
#include <atomic>
#include <future>
#endif

#include <memory>
#include <type_traits>
#include "megbrain/utils/thin/function.h"

namespace mgb {
#if MGB_ATLAS
[[noreturn]] void _on_atlas_error(
        const char* expr, aclError err, const char* file, const char* func, int line);
#endif

#if MGB_CUDA
[[noreturn]] void _on_cuda_error(
        const char* expr, cudaError_t err, const char* file, const char* func,
        int line);
[[noreturn]] void _on_cuda_cu_error(
        const char* expr, CUresult err, const char* file, const char* func, int line);
#endif

#if MGB_ROCM
[[noreturn]] void _on_hip_error(
        const char* expr, hipError_t err, const char* file, const char* func, int line);
#endif

#if MGB_CAMBRICON
const char* cnml_get_error_string(cnmlStatus_t err);
[[noreturn]] void _on_cnrt_error(
        const char* expr, cnrtRet_t err, const char* file, const char* func, int line);
[[noreturn]] void _on_cndev_error(
        const char* expr, cndevRet_t err, const char* file, const char* func, int line);
[[noreturn]] void _on_cnml_error(
        const char* expr, cnmlStatus_t err, const char* file, const char* func,
        int line);
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
using AtlasDispatcher = CPUDispatcher;

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
        return *m_user_data_container->get_user_data_or_create<T>(std::make_shared<T>);
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
#if MGB_ROCM
        if (m_property.type == DeviceType::ROCM) {
            m_rocm_env.activate();
        }
#endif
#if MGB_CAMBRICON
        if (m_property.type == DeviceType::CAMBRICON) {
            m_cnrt_env.activate();
        }
#endif
#if MGB_ATLAS
        if (m_property.type == DeviceType::ATLAS) {
            m_atlas_env.activate();
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
    void init_cuda_async(
            int dev, CompNode comp_node, const ContinuationCtx<cudaStream_t>& cont);
#endif

#if MGB_ATLAS
    struct AtlasEnv {
        int device = -1;
        aclrtStream stream = 0;

        struct InitStatus {
            bool initialized;
            Spinlock mtx;
            InitStatus() : initialized{false} {}
            void init() {
                MGB_LOCK_GUARD(mtx);
                if (!initialized) {
                    const char* config_path = MGB_GETENV("MGB_ATLAS_PROFILE_JSON");
                    auto acl_err = aclInit(config_path);
                    initialized = acl_err == ACL_ERROR_NONE;
                    mgb_throw_if(
                            !initialized, AtlasError,
                            "acl initialize failed: (acl: %s)",
                            megcore::atlas::get_error_str(acl_err));
                }
            }
            ~InitStatus() {
                MGB_LOCK_GUARD(mtx);
                if (initialized) {
                    initialized = false;
                }
            }
        };
        static InitStatus init_status;

        static void init() { init_status.init(); }

        void activate() const {
            init();
            int32_t device_id = -1;
            auto err = aclrtGetDevice(&device_id);
            if (err == ACL_ERROR_INVALID_DEVICE || device != device_id) {
                MGB_ATLAS_CHECK(aclrtSetDevice(device));
            } else {
                MGB_ATLAS_CHECK(err);
                mgb_assert(
                        err == ACL_ERROR_NONE,
                        "Failed to invoke aclrtGetDevice, get %s(%d)",
                        megcore::atlas::get_error_str(err), err);
            }
        }
    };

    const AtlasEnv& atlas_env() const {
        if (mgb_unlikely(m_property.type != DeviceType::ATLAS))
            on_bad_device_type(DeviceType::ATLAS);
        ensure_async_init_finished();
        return m_atlas_env;
    }

    //! init this as a atlas env synchronously
    void init_atlas(CompNode comp_node, const AtlasEnv& env);
#endif

#if MGB_ROCM
    struct ROCmEnv {
        int device = -1;
        hipStream_t stream = 0;
        hipDeviceProp_t device_prop;

        void activate() const { MGB_ROCM_CHECK(hipSetDevice(device)); }
    };

    const ROCmEnv& rocm_env() const {
        if (mgb_unlikely(m_property.type != DeviceType::ROCM))
            on_bad_device_type(DeviceType::ROCM);
        ensure_async_init_finished();
        return m_rocm_env;
    }

    //! init this as a rocm env asynchronously
    void init_rocm_async(
            int dev, CompNode comp_node, const ContinuationCtx<hipStream_t>& cont);

#endif

#if MGB_CAMBRICON
    struct CnrtEnv {
        int device = -1;
        cnrtQueue_t queue = nullptr;
        cnrtDeviceInfo_t device_info;
        struct InitStatus {
            bool initialized;
            Spinlock mtx;
            InitStatus() : initialized{false} {}
            void init() {
                MGB_LOCK_GUARD(mtx);
                if (!initialized) {
                    auto cnrt_err = cnrtInit(0);
                    initialized = cnrt_err == CNRT_RET_SUCCESS;
                    auto cndev_err = cndevInit(0);
                    initialized &= cndev_err == CNDEV_SUCCESS;
                    auto cnml_err = cnmlInit(0);
                    initialized &= cnml_err == CNML_STATUS_SUCCESS;
                    mgb_throw_if(
                            !initialized, CnrtError,
                            "cnrt/cndev/cnml initialize failed: (cnrt:%d, "
                            "cndev:%d, cnml: %d)",
                            static_cast<int>(cnrt_err), static_cast<int>(cndev_err),
                            static_cast<int>(cnml_err));
                }
            }
            ~InitStatus() {
                if (initialized) {
                    MGB_CNML_CHECK(cnmlExit());
                    MGB_CNDEV_CHECK(cndevRelease());
                    cnrtDestroy();
                    initialized = false;
                }
            }
        };
        static InitStatus init_status;

        static void init() { init_status.init(); }

        void activate() const {
            init();
            cnrtDev_t dev;
            MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev, device));
            MGB_CNRT_CHECK(cnrtSetCurrentDevice(dev));
        }
    };

    const CnrtEnv& cnrt_env() const {
        if (mgb_unlikely(m_property.type != DeviceType::CAMBRICON))
            on_bad_device_type(DeviceType::CAMBRICON);
        return m_cnrt_env;
    }

    void init_cnrt(
            int dev, CompNode comp_node, const ContinuationCtx<cnrtQueue_t>& cont);
#endif

    struct CpuEnv {
        using Task = CPUDispatcher::Task;
        using MultiThreadingTask = CPUDispatcher::MultiThreadingTask;
        using AffinityCallBack = thin_function<void(size_t)>;

        std::shared_ptr<CPUDispatcher> dispatcher;
#if MGB_HAVE_THREAD
        static MGB_THREAD_LOCAL_PTR(bool) do_task_inplace;
#else
        bool* do_task_inplace = nullptr;
#endif

        void enable_dispatch();

        void disable_dispatch(bool* flag);

        void dispatch(Task&& task) const;

        void dispatch(MultiThreadingTask&& task, size_t parallelism) const;

        void set_affinity(AffinityCallBack&& cb) const {
            dispatcher->set_affinity(std::move(cb));
        }
    };

    const CpuEnv& cpu_env() const {
        if (mgb_unlikely(m_property.type != DeviceType::CPU))
            on_bad_device_type(DeviceType::CPU);
        return m_cpu_env;
    }

    CpuEnv& cpu_env() {
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
#if MGB_ATLAS
    AtlasEnv m_atlas_env;
#endif
#if MGB_ROCM
    ROCmEnv m_rocm_env;
#endif
#if MGB_CAMBRICON
    CnrtEnv m_cnrt_env;
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
