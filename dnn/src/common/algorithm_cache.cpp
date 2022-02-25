#include "megdnn/algorithm_cache.h"
#include "megdnn/tensor_format.h"
#include "src/common/hash_ct.h"
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

AlgorithmCache& AlgorithmCache::instance() {
    static AlgorithmCache ins;
    return ins;
}

AlgorithmCache::KeyStorage AlgorithmCache::Key::build_key_storage() const {
    size_t buf_size = 16 * m_inp_layouts_size + 6;
    size_t buf[buf_size];

    size_t pos = 0;
    for (size_t i = 0; i < m_inp_layouts_size; i++) {
        auto&& layout = m_inp_layouts_ptr[i];
        if (layout.dtype.valid()) {
            buf[pos++] = static_cast<size_t>(layout.dtype.enumv());
        } else {
            buf[pos++] = static_cast<size_t>(SIZE_MAX);
        }
        buf[pos++] = static_cast<size_t>(layout.format.type());
        for (size_t j = 0; j < layout.ndim; j++) {
            buf[pos++] = layout.shape[j];
            buf[pos++] = layout.stride[j];
        }
    }

    buf[pos++] = m_opr_type;
    buf[pos++] = static_cast<size_t>(m_handle->type());

    switch (m_handle->type()) {
#if MEGDNN_WITH_CUDA
        case Handle::HandleType::CUDA: {
            int cuda_rt = -1;
            cuda_check(cudaRuntimeGetVersion(&cuda_rt));
            cuda_rt /= 1000;
            auto&& handle = static_cast<megdnn::cuda::HandleImpl*>(m_handle);
            auto&& prop = handle->device_prop();
            buf[pos++] = prop.major;
            buf[pos++] = prop.minor;
            buf[pos++] = cuda_rt;
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
            buf[pos++] = prop.major;
            buf[pos++] = prop.minor;
            buf[pos++] = drv;
            buf[pos++] = hip_rt;
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
            size_t nr_threads = static_cast<megdnn::naive::HandleImpl*>(m_handle)
                                        ->megcore_dispatcher()
                                        ->nr_threads();
            buf[pos++] = nr_threads;
            break;
        }
        default:
            break;
    }

    m_buf.resize(pos);
    SmallVector<size_t> tmp(buf, buf + pos);
    m_buf = std::move(tmp);

    size_t k1 = XXHash64CT::hash((const char*)buf, pos * sizeof(size_t), 20220328);
    size_t k2 = XXHash64CT::hash((const char*)m_param_ptr, m_param_size, 20220328);

    return {k1, k2};
}

void AlgorithmCache::put(const Key& key, Result& result) {
    MEGDNN_LOCK_GUARD(m_mtx);
    if (result.policy.algo.valid())
        m_heuristic_cache[key.build_key_storage()] = result;
}

template <typename T>
bool is_same_buf(
        const T hash_buf[], const size_t buf_size, const T hash_buf_[],
        const size_t buf_size_) {
    if (buf_size != buf_size_) {
        return false;
    }
    for (size_t i = 0; i < buf_size; i++) {
        if (hash_buf[i] != hash_buf_[i]) {
            return false;
        }
    }
    return true;
}

AlgorithmCache::Result AlgorithmCache::get(const Key& key) {
    MEGDNN_LOCK_GUARD(m_mtx);
    KeyStorage ks = key.build_key_storage();
    auto iter = m_heuristic_cache.find(ks);
    if (iter != m_heuristic_cache.end()) {
        if (is_same_buf(
                    key.m_buf.data(), key.m_buf.size(), iter->second.m_buf.data(),
                    iter->second.m_buf.size()) &&
            is_same_buf(
                    (char*)(key.m_param_ptr), key.m_param_size,
                    iter->second.m_param_buf.data(), iter->second.m_param_buf.size())) {
            return iter->second;
        }
        megdnn_log_warn(
                "hash collision occurs in heuristic cache with key: (%zu, %zu)", ks.k1,
                ks.k2);
    }
    SmallVector<char> param_buf(
            (char*)key.m_param_ptr, (char*)key.m_param_ptr + key.m_param_size);
    return Result{{}, 0, key.m_buf, param_buf};
}

void AlgorithmCache::clear() {
    MEGDNN_LOCK_GUARD(m_mtx);
    m_heuristic_cache.clear();
}
