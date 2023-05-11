#pragma once
#include "megdnn/handle.h"
#include "megdnn/oprs.h"
#include "src/common/reduce_helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/handle.h"
#include "src/cuda/multi_head_attn/helper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class MultiHeadAttnForwardImpl final : public MultiHeadAttnForward {
public:
    using MultiHeadAttnForward::MultiHeadAttnForward;
#if CUDNN_VERSION >= 8004
    MultiHeadAttnStatus desc_status;
#endif

    void exec(
            _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
            _megdnn_tensor_in qkvo_weight_bias, _megdnn_tensor_in attn_mask,
            _megdnn_tensor_in bias_k, _megdnn_tensor_in bias_v, _megdnn_tensor_out out,
            _megdnn_tensor_out attn_weight, _megdnn_tensor_out mask_reservespace,
            _megdnn_tensor_out othr_reservespace, _megdnn_workspace workspace) override;
    void deduce_layout(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, TensorLayout& out, TensorLayout& attn_weight,
            TensorLayout& mask_reservespace, TensorLayout& othr_reservespace) override;
    size_t get_mask_reservespace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace) override;
    size_t get_othr_reservespace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace) override;
};

class MultiHeadAttnBackwardImpl final : public MultiHeadAttnBackward {
public:
    using MultiHeadAttnBackward::MultiHeadAttnBackward;
#if CUDNN_VERSION >= 8004
    MultiHeadAttnStatus desc_status;
#endif
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
            _megdnn_tensor_in values, _megdnn_tensor_in qkvo_weight_bias,
            _megdnn_tensor_in attn_mask, _megdnn_tensor_in attn_weight,
            _megdnn_tensor_in mask_reservespace, _megdnn_tensor_in othr_reservespace,
            _megdnn_tensor_out dqueries, _megdnn_tensor_out dkeys,
            _megdnn_tensor_out dvalues, _megdnn_tensor_out dqkvo_weight_bias,
            _megdnn_tensor_out dbias_k, _megdnn_tensor_out dbias_v,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& queries,
            const TensorLayout& keys, const TensorLayout& values,
            const TensorLayout& qkvo_weight_bias, const TensorLayout& attn_mask,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace, const TensorLayout& dqueries,
            const TensorLayout& dkeys, const TensorLayout& dvalues,
            const TensorLayout& dqkvo_weight_bias, const TensorLayout& dbias_k,
            const TensorLayout& dbias_v) override;
};
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
