/**
 * \file dnn/src/naive/softmax/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/softmax/opr_impl.h"

#include <cstring>
#include "megdnn/dtype.h"
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_helper.cuh"
#include "src/common/opr_delegate.h"
#include "src/common/reduce_helper.h"
#include "src/common/utils.h"
#include "src/naive/elemwise/opr_impl.h"
#include "src/naive/handle.h"
#include "src/naive/lowbit_utils.h"

using namespace megdnn;

namespace {
template <typename T>
TensorND op_exec(_megdnn_tensor_in src, megdnn::dt_byte* workspace_ptr, const T& opr) {
    TensorLayout dst_layout;
    opr->deduce_layout(src.layout, dst_layout);
    TensorND dst{workspace_ptr, dst_layout};
    workspace_ptr += dst_layout.span().dist_byte();
    auto new_workspace = Workspace{
            workspace_ptr, opr->get_workspace_in_bytes(src.layout, dst_layout)};
    workspace_ptr += opr->get_workspace_in_bytes(src.layout, dst_layout);
    opr->exec(src, dst, new_workspace);
    return dst;
}

}  // namespace

namespace megdnn {
namespace naive {

//===============================Softmax Forward============================

void SoftmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto axis = param().axis;
    if (axis < 0)
        axis += src.layout.ndim;
    check_exec(src.layout, dst.layout, workspace.size);
    auto workspace_ptr = workspace.raw_ptr;

    auto reduce_opr = handle()->create_operator<ReduceForward>();
    reduce_opr->param().axis = axis;
    reduce_opr->param().mode = Reduce::Mode::MAX;
    reduce_opr->param().data_type = param::Reduce::DataType::DEFAULT;
    TensorND max_tensor = op_exec(src, workspace_ptr, reduce_opr);

    auto elemwise_opr = handle()->create_operator<Elemwise>();
    elemwise_opr->param().mode = Elemwise::Mode::SUB;
    elemwise_opr->exec({src, max_tensor}, dst);

    elemwise_opr->param().mode = Elemwise::Mode::EXP;
    TensorLayout exp_layout;
    elemwise_opr->deduce_layout({src.layout}, exp_layout);
    TensorND exp_tensor{workspace_ptr, exp_layout};
    workspace_ptr += exp_layout.span().dist_byte();
    elemwise_opr->exec({dst}, exp_tensor);

    reduce_opr->param().mode = Reduce::Mode::SUM;
    TensorND down_tensor = op_exec(exp_tensor, workspace_ptr, reduce_opr);

    elemwise_opr->param().mode = Elemwise::Mode::TRUE_DIV;
    elemwise_opr->exec({exp_tensor, down_tensor}, dst);
}

//=============================Softmax backward ============================

void SoftmaxBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    auto axis = param().axis;
    if (axis < 0)
        axis += src.layout.ndim;
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    auto workspace_ptr = workspace.raw_ptr;
    TensorLayout mulres = src.layout;
    mulres.dtype = src.layout.dtype;
    mulres.format = src.layout.format;
    mulres.init_contiguous_stride();

    TensorND mul_tensor{workspace_ptr, mulres};
    workspace_ptr += mulres.span().dist_byte();
    TensorND mul_tensor2{workspace_ptr, mulres};
    workspace_ptr += mulres.span().dist_byte();

    auto elemwise_opr = handle()->create_operator<Elemwise>();
    elemwise_opr->param().mode = Elemwise::Mode::MUL;
    elemwise_opr->exec({src, diff}, mul_tensor);

    auto reduce_opr = handle()->create_operator<ReduceForward>();
    reduce_opr->param().axis = axis;
    reduce_opr->param().mode = Reduce::Mode::SUM;
    reduce_opr->param().data_type = param::Reduce::DataType::DEFAULT;
    TensorND sum_tensor = op_exec(mul_tensor, workspace_ptr, reduce_opr);

    elemwise_opr->exec({sum_tensor, src}, mul_tensor2);

    elemwise_opr->param().mode = Elemwise::Mode::SUB;
    elemwise_opr->exec({mul_tensor, mul_tensor2}, grad);
}
}  // namespace naive
}  // namespace megdnn