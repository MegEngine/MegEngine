/**
 * \file dnn/src/naive/convolution/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class ConvolutionForwardImpl: public ConvolutionForward {
    public:
        using ConvolutionForward::ConvolutionForward;
        void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                  _megdnn_tensor_out dst,
                  const PreprocessedFilter* preprocessed_filter,
                  _megdnn_workspace workspace) override;
        std::vector<Algorithm *> get_all_algorithms(const TensorLayout &src,
                const TensorLayout &filter,
                const TensorLayout &dst) override;
        Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                           const TensorLayout& filter,
                                           const TensorLayout& dst,
                                           size_t workspace_limit_in_bytes,
                                           bool reproducible) override;
        size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                      const TensorLayout&,
                                      const PreprocessedFilter*) override {
            return 0;
        }

        size_t get_preprocess_workspace_in_bytes(const TensorLayout&,
                                                 const TensorLayout&,
                                                 const TensorLayout&) override {
            return 0;
        }

        void exec_preprocess(const TensorLayout&, _megdnn_tensor_in,
                             const TensorLayout&, PreprocessedFilter*,
                             _megdnn_workspace) override {}

        SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
                const TensorLayout& , const TensorLayout& ,
                const TensorLayout& )override{
            return {};
        }

        const char* get_algorithm_set_name() const override;
};

class ConvolutionBackwardDataImpl: public ConvolutionBackwardData {
    public:
        using ConvolutionBackwardData::ConvolutionBackwardData;
        void exec(_megdnn_tensor_in filter,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        std::vector<Algorithm *> get_all_algorithms(const TensorLayout &filter,
                const TensorLayout &diff,
                const TensorLayout &grad) override;
        Algorithm* get_algorithm_heuristic(const TensorLayout& filter,
                                           const TensorLayout& diff,
                                           const TensorLayout& grad,
                                           size_t workspace_limit_in_bytes,
                                           bool reproducible) override;
        size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                      const TensorLayout&) override;

        const char* get_algorithm_set_name() const override;
};

class ConvolutionBackwardFilterImpl: public ConvolutionBackwardFilter {
    public:
        using ConvolutionBackwardFilter::ConvolutionBackwardFilter;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        std::vector<Algorithm *> get_all_algorithms(const TensorLayout &src,
                const TensorLayout &diff,
                const TensorLayout &grad) override;
        Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                           const TensorLayout& diff,
                                           const TensorLayout& grad,
                                           size_t workspace_limit_in_bytes,
                                           bool reproducible) override;
        size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                      const TensorLayout&) override;

        const char* get_algorithm_set_name() const override;
};

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen
