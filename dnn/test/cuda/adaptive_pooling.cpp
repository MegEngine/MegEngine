/**
 * \file dnn/test/cuda/adaptive_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/tensor_iter.h"
#include "test/common/adaptive_pooling.h"
#include "test/common/checker.h"

#include "src/common/utils.h"
#include "test/cuda/utils.h"

#include <cudnn.h>
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, ADAPTIVE_POOLING_FORWARD) {
    auto args = adaptive_pooling::get_args();
    using Format = param::AdaptivePooling::Format;
    DType dtype = dtype::Float32();
    for (auto&& arg : args) {
        auto param = arg.param;
        auto src = arg.ishape;
        auto dst = arg.oshape;
        param.format = Format::NCHW;
        Checker<AdaptivePooling> checker(handle_cuda());
        checker.set_epsilon(1e-2);
        checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                TensorShapeArray{src, dst, {}});
    }
}

TEST_F(CUDA, ADAPTIVE_POOLING_BACKWARD) {
    auto args = adaptive_pooling::get_args();
    for (auto&& arg : args) {
        Checker<AdaptivePoolingBackward> checker(handle_cuda());
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout = TensorLayout(arg.oshape, dtype::Float32());

        auto constraint = [this,
                           arg](CheckerHelper::TensorValueArray& tensors_orig) {
            megdnn_assert(tensors_orig.size() == 4);
            auto opr = handle_cuda()->create_operator<AdaptivePoolingForward>();
            opr->param() = arg.param;

            auto tensors_cuda_storage = CheckerHelper::alloc_tensors(
                    handle_cuda(),
                    {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
            auto&& tensors_cuda = *tensors_cuda_storage;

            auto span = tensors_cuda[0].layout.span();
            auto dst = static_cast<dt_byte*>(tensors_cuda[0].raw_ptr) +
                       span.low_byte;
            auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr) +
                       span.low_byte;
            megdnn_memcpy_H2D(handle_cuda(), dst, src, span.dist_byte());

            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors_cuda[0].layout, tensors_cuda[1].layout);
            auto workspace_cuda = megdnn_malloc(handle_cuda(), workspace_size);
            Workspace workspace{static_cast<dt_byte*>(workspace_cuda),
                                workspace_size};
            opr->exec(tensors_cuda[0], tensors_cuda[1], workspace);
            megdnn_free(handle_cuda(), workspace_cuda);

            span = tensors_cuda[1].layout.span();
            dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr) +
                  span.low_byte;
            src = static_cast<const dt_byte*>(tensors_cuda[1].raw_ptr) +
                  span.low_byte;
            megdnn_memcpy_D2H(handle_cuda(), dst, src, span.dist_byte());
        };

        DType dtype = dtype::Float32();
        checker.set_tensors_constraint(constraint)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_param(arg.param)
                .exec(TensorShapeArray{ilayout, olayout, olayout, ilayout});
    }
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
