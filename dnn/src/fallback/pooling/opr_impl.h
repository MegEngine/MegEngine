/**
 * \file dnn/src/fallback/pooling/opr_impl.h
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
#include "megdnn/oprs/base.h"
#include "src/naive/pooling/opr_impl.h"

namespace megdnn {
namespace fallback {

class PoolingImpl : public naive::PoolingForwardImpl {
private:
    class AlgoGiFilterxModexStride1;
    class AlgoGiFilter2ModexStride2;
    class AlgoGiFilter3MaxStride2;
    class AlgoGiFilter3AverageStride2;
    class AlgoGiFilter4MaxStride2;
    class AlgoGiFilter5MaxStride2;
    class AlgoGiFp32ModexStridexNCHW44;
    class AlgoFallback;
    class AlgoPack;
    static AlgoPack sm_algo_pack;

    void exec_w3x3_s1x1(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const Param& param);
    void exec_w2x2_s2x2_int8(_megdnn_tensor_in src, _megdnn_tensor_out dst);
    void exec_w2x2_s2x2_avg_int8(_megdnn_tensor_in src, _megdnn_tensor_out dst);
    void exec_fallback(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace);

public:
    using naive::PoolingForwardImpl::PoolingForwardImpl;
    using Param = param::Pooling;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;

    static size_t constexpr MAX_SPATIAL_DIM = 2;

    struct PoolingKernSizeParam {
        uint32_t n, ic;
        std::array<uint32_t, MAX_SPATIAL_DIM> isz, osz;
        std::array<uint32_t, MAX_SPATIAL_DIM> padding, filter, stride;
        DType src_type, dst_type;
        Handle* handle;
        Param::Format format;
        Mode mode;
    };

    struct PoolingKernParam : public PoolingKernSizeParam {
        RefPtr src_ptr;
        RefPtr dst_ptr;
        void* workspace_ptr;
        size_t workspace_size;

        template <typename T>
        const T* src() const {
            src_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(src_ptr.get_ptr());
        }

        template <typename T>
        T* dst() const {
            dst_type.assert_is_compatible_ctype<T>();
            return static_cast<T*>(dst_ptr.get_ptr());
        }

        template <typename T>
        T* workspace() const {
            return static_cast<T*>(workspace_ptr);
        }
    };

    PoolingKernSizeParam make_pooling_kern_szie_param(
            fallback::PoolingImpl* opr, const TensorLayout& src,
            const TensorLayout& dst);

    PoolingKernParam make_pooling_kern_param(
            fallback::PoolingImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace);
    class AlgoBase : public detail::Algorithm {
    public:
        enum class AlgoType : uint32_t {
            GI_FilterxModexStride1,
            GI_Filter2ModexStride2,
            GI_Filter3MaxStride2,
            GI_Filter3AverageStride2,
            GI_Filter4MaxStride2,
            GI_Filter5MaxStride2,
            GI_Filter2ModexStridexNCHW44,
            GI_Filter3ModexStridexNCHW44,
            GI_Filter4ModexStridexNCHW44,
            GI_Filter5ModexStridexNCHW44,
            GI_Fp32ModexStridexNCHW44,
            FallbackNotGI
        };

        using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
        AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::FALLBACK; }
        virtual ~AlgoBase() = default;
        virtual bool usable(const PoolingKernSizeParam& param) const = 0;
        virtual void exec(const PoolingKernParam& param) const = 0;

        uint32_t type() const override { return INVALID_ALGO_TYPE; };
        bool is_available_attribute(
                const PoolingKernSizeParam& param,
                const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
                const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
            return contain_attribute_all(positive_attr) &&
                   !contain_attribute_any(negative_attr) && usable(param);
        }
    };

    const char* get_algorithm_set_name() const override {
        return "FALLBACK_POOLING_FORWARD";
    }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& dst) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) {
        return fallback::PoolingImpl::get_algorithm_heuristic(
                       src, dst, workspace_limit_in_bytes, positive_attr, negative_attr)
                ->info();
    }

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    bool is_fallback_non_gi_algo(Algorithm* algo) {
        return strcmp(algo->name(), "FALLBACK_NOT_GI_POOLING") == 0;
    }
};
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
