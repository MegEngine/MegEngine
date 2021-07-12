/**
 * \file dnn/src/x86/pooling/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <unordered_map>
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/x86/pooling/opr_impl.h"
#include "src/x86/handle.h"

namespace megdnn {
namespace x86 {
using AlgoBase = PoolingImpl::AlgoBase;

class PoolingImpl::AlgoBase : public Algorithm {
public:
    enum class AlgoType : uint32_t {
        X86_MeanW2S2AVX,
        X86_MeanW2S2SSE3,
        X86_MaxW2S2SSE,
        X86_MaxW3S3SSE,
#if MEGDNN_X86_WITH_MKL_DNN
        X86_MKLDNNNCHW,
        X86_MKLDNNNCHW88,
#endif
        X86_Fallback
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::X86; }
    virtual ~AlgoBase() = default;
    struct SizeArgs {
        HandleImpl* handle;
        PoolingImpl* opr;
        const TensorLayout layout_src, layout_dst;

        std::string to_string() const;
        SizeArgs(PoolingImpl* opr, const TensorLayout& src,
                 const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *dst_tensor;
        Workspace workspace;

        ExecArgs(PoolingImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_out dst, _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    uint32_t type() const override { return INVALID_ALGO_TYPE; };
    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available(args);
    }
};

#define ALGO_IMPL(_name)                                                       \
    class PoolingImpl::Algo##_name final : public AlgoBase {                   \
        std::string m_algo_name;                                               \
                                                                               \
    public:                                                                    \
        Algo##_name() : m_algo_name(std::string(#_name).append("_POOLING")) {} \
        AlgoAttribute attribute() const override {                             \
            return AlgoAttribute::REPRODUCIBLE;                                \
        };                                                                     \
        const char* name() const override { return m_algo_name.c_str(); }      \
        bool is_available(const SizeArgs& args) const override;                \
        void exec(const ExecArgs& args) const override;                        \
        MEGDNN_DECL_ALGO_TYPE(X86_##_name)                                     \
    };

ALGO_IMPL(MeanW2S2AVX)
ALGO_IMPL(MeanW2S2SSE3)
ALGO_IMPL(MaxW2S2SSE)
ALGO_IMPL(MaxW3S3SSE)
#if MEGDNN_X86_WITH_MKL_DNN
ALGO_IMPL(MKLDNNNCHW)
ALGO_IMPL(MKLDNNNCHW88)
#endif

#undef ALGO_IMPL

class PoolingImpl::AlgoFallback final : public AlgoBase {
    std::string m_algo_name;
public:
    AlgoFallback() : m_algo_name("FALLBACK_POOLING") {}
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    };
    const char* name() const override { return m_algo_name.c_str(); }
    bool is_available(const SizeArgs&) const override { return true; }
    void exec(const ExecArgs&) const override {}
    MEGDNN_DECL_ALGO_TYPE(X86_Fallback)
};

class PoolingImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;
    AlgoMeanW2S2AVX algo_mean_w2s2_avx;
    AlgoMeanW2S2SSE3 algo_mean_w2s2_sse3;
    AlgoMaxW2S2SSE algo_max_w2s2_sse;
    AlgoMaxW3S3SSE algo_max_w3s3_sse;
#if MEGDNN_X86_WITH_MKL_DNN
    AlgoMKLDNNNCHW algo_mkldnn_nchw;
    AlgoMKLDNNNCHW88 algo_mkldnn_nchw88;
#endif
    AlgoFallback algo_fallback;

public:
    AlgoPack();

    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace x86
}  // namespace megdnn
