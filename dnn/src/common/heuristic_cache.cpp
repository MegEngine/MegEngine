/**
 * \file dnn/src/common/heuristic_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/heuristic_cache.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#if MEGDNN_WITH_CUDA
#include "src/cuda/utils.h"
#endif

#if MEGDNN_WITH_ROCM
#include "hcc_detail/hcc_defs_prologue.h"
#include "megcore_rocm.h"
#include "src/rocm/utils.h"
#endif

using namespace megdnn;

HeuristicCache& HeuristicCache::instance() {
    static HeuristicCache ins;
    return ins;
}

HeuristicCache::KeyStorage HeuristicCache::Key::build_key_storage() const {
    auto&& ctg = m_category;
    auto&& inp = m_input;

    if (!m_category.empty() && !m_input.empty())
        return {ctg, inp};

    inp.reserve(sizeof(TensorLayout) * 3 * m_inp_layouts_size + m_param_size);
    for (size_t i = 0; i < m_inp_layouts_size; i++) {
        auto&& ly = m_inp_layouts_ptr[i];
        for (size_t j = 0; j < ly.ndim; j++) {
            if (j)
                inp.push_back(',');
            inp.append(std::to_string(ly.shape[j]));
        }
        inp.push_back(';');
        for (size_t j = 0; j < ly.ndim; j++) {
            if (j)
                inp.push_back(',');
            inp.append(std::to_string(ly.stride[j]));
        }
        inp.push_back(';');
        inp.append(ly.dtype.name());
        inp.push_back(';');
        inp.append(ly.format.to_string().c_str());
        inp.push_back('|');
    }
    if (m_param_size) {
        inp.append(reinterpret_cast<const char*>(m_param_ptr), m_param_size);
    }

    ctg = "plat:";
    ctg.append(std::to_string(static_cast<uint32_t>(m_handle->type())));
    switch (m_handle->type()) {
#if MEGDNN_WITH_CUDA
        case Handle::HandleType::CUDA: {
            int cuda_rt = -1;
            cuda_check(cudaRuntimeGetVersion(&cuda_rt));
            cuda_rt /= 1000;
            auto&& handle = static_cast<megdnn::cuda::HandleImpl*>(m_handle);
            auto&& prop = handle->device_prop();
            ctg.append(ssprintf(";dev=%s;cap=%d.%d;runtime=%d;",
                            prop.name, prop.major, prop.minor, cuda_rt));
            break;
        }
#endif
#if MEGDNN_WITH_ROCM
        case Handle::HandleType::ROCM: {
            auto&& handle = static_cast<megdnn::rocm::HandleImpl*>(m_handle);
            auto&& prop = handle->device_prop();
            int drv = -1, hip_rt = -1;
            hip_check(hipDriverGetVersion(&drv));
            hip_check(hipRuntimeGetVersion(&hip_rt));
            ctg.append(ssprintf(";dev=%s;cap=%d.%d,drv=%d;runtime=%d;",
                            prop.name, prop.major, prop.minor, drv, hip_rt));
            break;
        }
#endif
        case Handle::HandleType::FALLBACK:
#if MEGDNN_X86
        case Handle::HandleType::X86:
#endif
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
        case Handle::HandleType::ARM_COMMON:
#endif
#if MEGDNN_AARCH64
        case Handle::HandleType::AARCH64:
#endif
#if MEGDNN_ARMV7
        case Handle::HandleType::ARMV7:
#endif
        {
            size_t nr_threads =
                    static_cast<megdnn::naive::HandleImpl*>(m_handle)
                            ->megcore_dispatcher()
                            ->nr_threads();
            ctg.append(";");
            ctg.append(std::to_string(nr_threads));
            ctg.append(";");
            break;
        }
        default:
            ctg.append(";");
    }
    ctg.append(std::to_string(m_opr_type));
    return {ctg, inp};
}

void HeuristicCache::put(const Key& key, Result& result) {
    MEGDNN_LOCK_GUARD(m_mtx);
    if (result.policy.algo.valid())
        m_heuristic_cache[key.build_key_storage()] = result;
}

HeuristicCache::Result HeuristicCache::get(const Key& key) {
    MEGDNN_LOCK_GUARD(m_mtx);
    KeyStorage ks = key.build_key_storage();
    auto iter = m_heuristic_cache.find(ks);
    if (iter == m_heuristic_cache.end()) {
        return {};
    } else {
        return iter->second;
    }
}

void HeuristicCache::clear() {
    MEGDNN_LOCK_GUARD(m_mtx);
    m_heuristic_cache.clear();
}