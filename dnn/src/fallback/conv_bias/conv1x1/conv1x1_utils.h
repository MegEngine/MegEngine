#pragma once

#include <unordered_map>
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {
namespace conv1x1 {
namespace utils {

struct StrategyHashKey {
    ConvBiasImpl::NCBKernSizeParam param;
    param::ConvBias::Format format;
    MatrixMulImpl::AlgoBase::PackMode packmode;
};

struct StrategyHasher {
    std::size_t operator()(const StrategyHashKey& key) const {
        constexpr size_t base = 1;  //! avoid hashkey is zero
        std::size_t result =
                static_cast<std::size_t>(key.param.src_type.enumv()) + base;
        result = result ^
                 ((static_cast<std::size_t>(key.param.dst_type.enumv()) + base) << 3);
        result =
                result ^
                ((static_cast<std::size_t>(key.param.filter_type.enumv()) + base) << 6);
        result = result ^
                 ((static_cast<std::size_t>(key.param.bias_type.enumv()) + base) << 9);
        result = result ^ ((static_cast<std::size_t>(key.format) + base) << 12);
        result = result ^ ((static_cast<std::size_t>(key.packmode) + base) << 15);
        return result;
    }
};

struct StrategyHashKeyEqual {
    bool operator()(const StrategyHashKey& key1, const StrategyHashKey& key2) const {
        return key1.param.src_type == key2.param.src_type &&
               key1.param.filter_type == key2.param.filter_type &&
               key1.param.bias_type == key2.param.bias_type &&
               key1.param.dst_type == key2.param.dst_type &&
               key1.format == key2.format && key1.packmode == key2.packmode;
    }
};

template <typename T>
class StrategyDelegationStorage {
    using creator = std::function<std::unique_ptr<T>(
            const ConvBiasImpl::NCBKernSizeParam&, MatrixMulImpl::AlgoBase::PackMode,
            param::ConvBias::Format)>;

public:
    T* get(const ConvBiasImpl::NCBKernSizeParam& param,
           MatrixMulImpl::AlgoBase::PackMode pack_mode, param::ConvBias::Format format,
           creator Fun) {
        MEGDNN_LOCK_GUARD(m_mtx);
        StrategyHashKey key;
        key.param = param;
        key.format = format;
        key.packmode = pack_mode;
        if (m_map_strategies.find(key) == m_map_strategies.end()) {
            auto strategy = Fun(param, pack_mode, format);
            m_map_strategies[key] = std::move(strategy);
        }
        return m_map_strategies[key].get();
    }

private:
    DNN_MUTEX m_mtx;
    std::unordered_map<
            StrategyHashKey, std::unique_ptr<T>, StrategyHasher, StrategyHashKeyEqual>
            m_map_strategies;
};

//! get_thread_bundle
WorkspaceBundle get_thread_bundle(
        const ConvBiasImpl::NCBKernSizeParam& param, size_t matmul_c_size,
        size_t oc_tile_size);
//! get_matmul_kern_param
MatrixMulImpl::KernSizeParam get_matmul_kern_param(
        const ConvBiasImpl::NCBKernSizeParam& param, size_t n, size_t m);

}  // namespace utils
}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
