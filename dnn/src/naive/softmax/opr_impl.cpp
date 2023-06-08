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
namespace megdnn {
namespace naive {

//===============================Softmax Forward============================

size_t SoftmaxForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    int32_t axis = param().axis;
    int32_t nidm = src.ndim;
    if (axis < 0)
        axis += nidm;
    megdnn_assert(axis >= 0, "is not a vaild axis=%d for dim=%d", axis, nidm);

    reduce_opr = handle()->create_operator<Reduce>();
    elemwise_opr = handle()->create_operator<Elemwise>();

    reduce_opr->param().axis = axis;
    reduce_opr->param().data_type = param::Reduce::DataType::DEFAULT;
    reduce_opr->param().mode = Reduce::Mode::MAX;

    reduce_opr->param().mode = Reduce::Mode::MAX;
    size_t max_workspace = reduce_opr->get_workspace_in_bytes(src, dst);
    reduce_opr->param().mode = Reduce::Mode::SUM;
    size_t sum_workspace = reduce_opr->get_workspace_in_bytes(src, dst);
    reduce_worksize = max_workspace > sum_workspace ? max_workspace : sum_workspace;

    return WorkspaceBundle(nullptr, {src.span().dist_byte(), reduce_worksize})
            .total_size_in_bytes();
}

void SoftmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(src.layout, dst.layout, workspace.size);

    WorkspaceBundle workspace_bundle{
            workspace.raw_ptr, {src.layout.span().dist_byte(), reduce_worksize}};

    TensorLayout tmp_layout;
    reduce_opr->param().mode = Reduce::Mode::MAX;
    reduce_opr->deduce_layout(src.layout, tmp_layout);
    TensorND max_tensor{workspace_bundle.get_workspace(0).raw_ptr, tmp_layout};
    reduce_opr->exec(src, max_tensor, workspace_bundle.get_workspace(1));

    elemwise_opr->param().mode = Elemwise::Mode::SUB;
    elemwise_opr->exec({src, max_tensor}, dst);

    // no broadcast
    elemwise_opr->param().mode = Elemwise::Mode::EXP;
    elemwise_opr->exec({dst}, dst);

    reduce_opr->param().mode = Reduce::Mode::SUM;
    reduce_opr->deduce_layout(src.layout, tmp_layout);

    TensorND deno_tensor{workspace_bundle.get_workspace(0).raw_ptr, tmp_layout};
    reduce_opr->exec(dst, deno_tensor, workspace_bundle.get_workspace(1));

    elemwise_opr->param().mode = Elemwise::Mode::TRUE_DIV;
    elemwise_opr->exec({dst, deno_tensor}, dst);
#else
    __builtin_trap();
#endif
}

//=============================Softmax backward ============================

size_t SoftmaxBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout&) {
    int32_t axis = param().axis;
    int32_t nidm = src.ndim;
    if (axis < 0)
        axis += nidm;
    megdnn_assert(axis >= 0, "is not a vaild axis=%d for dim=%d", axis, nidm);

    reduce_opr = handle()->create_operator<Reduce>();
    elemwise_opr = handle()->create_operator<Elemwise>();
    reduce_opr->param().axis = axis;
    reduce_opr->param().data_type = param::Reduce::DataType::DEFAULT;
    reduce_opr->param().mode = Reduce::Mode::SUM;
    reduce_worksize = reduce_opr->get_workspace_in_bytes(src, diff);

    return WorkspaceBundle(
                   nullptr,
                   {src.span().dist_byte(), src.span().dist_byte(), reduce_worksize})
            .total_size_in_bytes();
}

void SoftmaxBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);

    WorkspaceBundle workspace_bundle{
            workspace.raw_ptr,
            {src.layout.span().dist_byte(), src.layout.span().dist_byte(),
             reduce_worksize}};

    TensorLayout mul_layout = src.layout;
    mul_layout.dtype = src.layout.dtype;
    mul_layout.format = src.layout.format;
    mul_layout.init_contiguous_stride();

    TensorND mul_lhs_tensor{workspace_bundle.get_workspace(0).raw_ptr, mul_layout};
    TensorND mul_rhs_tensor{workspace_bundle.get_workspace(1).raw_ptr, mul_layout};

    elemwise_opr->param().mode = Elemwise::Mode::MUL;
    elemwise_opr->exec({src, diff}, mul_lhs_tensor);

    TensorLayout sum_layout;
    reduce_opr->deduce_layout(mul_lhs_tensor.layout, sum_layout);
    TensorND sum_tensor{grad.raw_ptr(), sum_layout};
    reduce_opr->exec(mul_lhs_tensor, sum_tensor, workspace_bundle.get_workspace(2));

    // there are broadcast occurring elemwsie mul
    elemwise_opr->exec({sum_tensor, src}, mul_rhs_tensor);

    elemwise_opr->param().mode = Elemwise::Mode::SUB;
    elemwise_opr->exec({mul_lhs_tensor, mul_rhs_tensor}, grad);
#else
    __builtin_trap();
#endif
}
}  // namespace naive
}  // namespace megdnn
