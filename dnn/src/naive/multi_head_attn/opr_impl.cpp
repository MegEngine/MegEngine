#include "src/naive/multi_head_attn/opr_impl.h"
#include "megdnn/oprs/linalg.h"
#include "src/common/utils.cuh"

namespace megdnn {
namespace naive {

using Param = MultiHeadAttnBase::Param;

size_t MultiHeadAttnForwardImpl::get_workspace_in_bytes(
        const TensorLayout& /*queries*/, const TensorLayout& /*keys*/,
        const TensorLayout& /*values*/, const TensorLayout& /*qkvo_weight_bias*/,
        const TensorLayout& /*attn_mask*/, const TensorLayout& /*bias_k*/,
        const TensorLayout& /*bias_v*/, const TensorLayout& /*out*/,
        const TensorLayout& /*attn_weight*/, const TensorLayout& /*mask_reservespace*/,
        const TensorLayout& /*othr_reservespace*/) {
    megdnn_throw("unsupported naive multiheadattn forward\n");
}

void MultiHeadAttnForwardImpl::exec(
        _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
        _megdnn_tensor_in qkvo_weight_bias, _megdnn_tensor_in attn_mask,
        _megdnn_tensor_in bias_k, _megdnn_tensor_in bias_v, _megdnn_tensor_out out,
        _megdnn_tensor_out attn_weight, _megdnn_tensor_out mask_reservespace,
        _megdnn_tensor_out othr_reservespace, _megdnn_workspace workspace) {
    check_exec(
            queries.layout, keys.layout, values.layout, qkvo_weight_bias.layout,
            attn_mask.layout, bias_k.layout, bias_v.layout, out.layout,
            attn_weight.layout, mask_reservespace.layout, othr_reservespace.layout,
            workspace.size);

    megdnn_throw("unsupported naive multiheadattn forward\n");
}

void MultiHeadAttnBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
        _megdnn_tensor_in values, _megdnn_tensor_in qkvo_weight_bias,
        _megdnn_tensor_in attn_mask, _megdnn_tensor_in attn_weight,
        _megdnn_tensor_in mask_reservespace, _megdnn_tensor_in othr_reservespace,
        _megdnn_tensor_out dqueries, _megdnn_tensor_out dkeys,
        _megdnn_tensor_out dvalues, _megdnn_tensor_out dqkvo_weight_bias,
        _megdnn_tensor_out dbias_k, _megdnn_tensor_out dbias_v,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, queries.layout, keys.layout, values.layout,
            qkvo_weight_bias.layout, attn_mask.layout, attn_weight.layout,
            mask_reservespace.layout, othr_reservespace.layout, dqueries.layout,
            dkeys.layout, dvalues.layout, dqkvo_weight_bias.layout, dbias_k.layout,
            dbias_v.layout, workspace.size);

    megdnn_throw("unsupported naive multiheadattn backward\n");
}

}  // namespace naive
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
