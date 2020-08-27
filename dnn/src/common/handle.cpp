/**
 * \file dnn/src/common/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"

#include "src/common/handle_impl.h"
#include "src/common/utils.h"
#include "src/fallback/handle.h"
#include "src/naive/handle.h"

#include "midout.h"

#if MEGDNN_X86
#include "src/x86/handle.h"
#endif

#if MEGDNN_ARMV7
#include "src/armv7/handle.h"
#endif

#if MEGDNN_AARCH64
#include "src/aarch64/handle.h"
#endif


#if MEGDNN_WITH_CUDA
#include "src/cuda/handle.h"
#endif


#if MEGDNN_WITH_CAMBRICON
#include "src/cambricon/handle.h"
#endif

#ifdef MEGDNN_WITH_ATLAS
#include "src/atlas/handle.h"
#endif

using namespace megdnn;

MIDOUT_DECL(HandlePlatform);
MIDOUT_DECL(HandleOpr);

Handle::Handle(megcoreComputingHandle_t computing_handle, HandleType type)
        : m_computing_handle(computing_handle), m_handle_type(type) {}

std::unique_ptr<Handle> Handle::make(megcoreComputingHandle_t computing_handle,
                                     int debug_level) {
    (void)debug_level;
    megcoreDeviceHandle_t device_handle;
    megcorePlatform_t platform;
    megcoreGetDeviceHandle(computing_handle, &device_handle);

    megcoreGetPlatform(device_handle, &platform);
    if (platform == megcorePlatformCPU) {
        // only enable midout for CPU, becuase CPU might be unused when some
        // other platforms are used
        MIDOUT_BEGIN(HandlePlatform, midout_iv(megcorePlatformCPU)) {
        // CPU
#if MEGDNN_NAIVE
            return make_unique<naive::HandleImpl>(computing_handle);
#else
            if (debug_level == 0) {
#if MEGDNN_X86
                // Because of ICC bug, we cannot use make_unique here. It will
                // trigger an internal compiler error.
                return std::unique_ptr<x86::HandleImpl>(
                        new x86::HandleImpl(computing_handle));
                // return make_unique<x86::HandleImpl>(computing_handle);
#elif MEGDNN_ARMV7
                return make_unique<armv7::HandleImpl>(computing_handle);
#elif MEGDNN_AARCH64
                return make_unique<aarch64::HandleImpl>(computing_handle);
#else
                return make_unique<fallback::HandleImpl>(computing_handle);
#endif
            } else if (debug_level == 1) {
                return make_unique<fallback::HandleImpl>(computing_handle);
            } else if (debug_level == 2) {
                return make_unique<naive::HandleImpl>(computing_handle);
            } else {
                megdnn_throw(megdnn_mangle("Debug level must be 0/1/2."));
            }
        }
        MIDOUT_END();
#endif
        }
        else if (platform == megcorePlatformROCM) {
#if MEGDNN_WITH_ROCM
            return make_rocm_handle(computing_handle);
#else
            return nullptr;
#endif
        }
        else if (platform == megcorePlatformCambricon) {
#if MEGDNN_WITH_CAMBRICON
            return make_unique<cambricon::HandleImpl>(computing_handle);
#else
            return nullptr;
#endif
        }
        else if (platform == megcorePlatformAtlas) {
#if MEGDNN_WITH_ATLAS
            return make_unique<atlas::HandleImpl>(computing_handle);
#else
            return nullptr;
#endif
        }
        else {
            // CUDA
            megdnn_assert_internal(platform == megcorePlatformCUDA);
#if MEGDNN_WITH_CUDA
            return make_unique<cuda::HandleImpl>(computing_handle);
#else
            return nullptr;
#endif
        }
        return nullptr;
    }


    void Handle::set_destructor(const thin_function<void()>& d) {
        megdnn_assert(!m_destructor, "destructor can be set only once");
        m_destructor = d;
    }

    Handle::~Handle() {
        if (m_destructor)
            m_destructor();
        m_alive_magic = 0;
    }

    size_t Handle::alignment_requirement() const {
        // default to 32
        return 32;
    }

    size_t Handle::image2d_pitch_alignment() const {
        megdnn_throw("image2d tensor format not supported on this handle");
    }

    bool Handle::check_cross_dev_copy_constraint(const TensorLayout& src) {
        return src.is_contiguous();
    }

    void Handle::on_opr_destructed(OperatorBase * opr) {
        if (m_alive_magic != ALIVE_MAGIC) {
            megdnn_log_error(
                    "Handle is destructed before opr gets destructed. "
                    "Please fix the destruction order as this would cause "
                    "undefined memory access. "
                    "Abort now to avoid further problems.");
            abort();
        }
        if (m_on_opr_destructed) {
            m_on_opr_destructed(opr);
        }
    }

    OperatorBase::~OperatorBase() { m_handle->on_opr_destructed(this); }

    template <typename Opr>
    std::unique_ptr<Opr> Handle::create_operator() {
#define CASE(etype, nm)                                                        \
    case HandleType::etype: {                                                  \
        MIDOUT_BEGIN(HandleOpr, Opr, midout_iv(HandleType::etype)) {           \
            return static_cast<nm::HandleImpl*>(this)->create_operator<Opr>(); \
        }                                                                      \
        MIDOUT_END();                                                          \
    }

        switch (m_handle_type) {
            CASE(NAIVE, naive);
#if !MEGDNN_NAIVE
            CASE(FALLBACK, fallback);
#if MEGDNN_X86
            CASE(X86, x86);
#endif
#if MEGDNN_ARMV7
            CASE(ARMV7, armv7);
#endif
#if MEGDNN_AARCH64
            CASE(AARCH64, aarch64);
#endif
#if MEGDNN_ARMV7 || MEGDNN_AARCH64
            CASE(ARM_COMMON, arm_common);
#endif
#endif  // !MEGDNN_NAIVE
#if MEGDNN_WITH_CUDA
            CASE(CUDA,cuda);
#endif
#if MEGDNN_WITH_ATLAS
            CASE(ATLAS, atlas);
#endif
#if MEGDNN_WITH_ROCM
            case HandleType::ROCM: {
                MIDOUT_BEGIN(HandleOpr, Opr, midout_iv(HandleType::ROCM)) {
                    return create_rocm_operator<Opr>();
                }
                MIDOUT_END();
            }
#endif
#if MEGDNN_WITH_CAMBRICON
            CASE(CAMBRICON, cambricon);
#endif
            default:
                megdnn_throw(megdnn_mangle("bad handle type"));
        }
#undef CASE
    }

#define INST(opr) template std::unique_ptr<opr> Handle::create_operator();
        MEGDNN_FOREACH_OPR_CLASS(INST)
#undef INST
// vim: syntax=cpp.doxygen

