/**
 * \file dnn/src/cuda/conv_bias/opr_impl.h
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
#include "../elemwise/opr_impl.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class ConvBiasForwardImpl : public ConvBiasForward {
public:
    using ConvBiasForward::ConvBiasForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
            _megdnn_tensor_in z, _megdnn_tensor_out dst,
            const PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&,
            const PreprocessedFilter*) override;

    size_t get_preprocess_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override;
    void exec_preprocess(
            const TensorLayout&, _megdnn_tensor_in, _megdnn_tensor_in,
            const TensorLayout&, const TensorLayout&, PreprocessedFilter*,
            _megdnn_workspace) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoCUDNNConvBiasActivation;
    class AlgoChanwise;
    class AlgoChanwiseSmall;
    class AlgoChanwise8x8x32;
    class AlgoCUDNNConv;
    class AlgoFallbackNCHWQS8;
    class AlgoInplaceMatmul;
    class AlgoMatmul;
    class AlgoMatmul8x8x32;
    class Algo1x1;
    class AlgoBatchedMatmul;
    class AlgoGroupConvGeneral;
    class AlgoQUInt4x4x32WMMA;
    class AlgoCutlassConvolutionBase;
    class AlgoInt8CHWN4DotProdImplicitGemm;
    class AlgoInt8NCHW4DotProdImplicitGemm;
    class AlgoInt8CHWN4IMMAImplicitGemm;
    class AlgoInt8NCHW4IMMAImplicitGemm;
    class AlgoInt8CHWN4IMMAImplicitGemmReorderFilter;
    class AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth;
    class AlgoInt8NCHW32IMMAImplicitGemm;
    class AlgoInt8NHWCIMMAImplicitGemm;
    class AlgoInt4NCHW64IMMAImplicitGemmBase;
    class AlgoInt4Int4NCHW64IMMAImplicitGemm;
    class AlgoUInt4Int4NCHW64IMMAImplicitGemm;
    class AlgoInt4NHWCIMMAImplicitGemmBase;
    class AlgoInt4Int4NHWCIMMAImplicitGemm;
    class AlgoUInt4Int4NHWCIMMAImplicitGemm;
    class AlgoBFloat16;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
