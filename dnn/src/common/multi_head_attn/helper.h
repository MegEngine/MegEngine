#pragma once
#include "megdnn/dtype.h"

#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

namespace megdnn {

namespace multi_head_attn {

inline void matmul_deduce_layout(
        std::unique_ptr<MatrixMulForward>& opr, const TensorLayout& A,
        const TensorLayout& B, TensorLayout& C) {
    megdnn_assert(A.ndim == 3 && B.ndim == 2);
    auto m_param = opr->param();

    size_t A1, A2, B0, B1;
    A1 = A.shape[1];
    A2 = A.shape[2];
    B0 = B.shape[0];
    B1 = B.shape[1];
    if (m_param.transposeA) {
        std::swap(A1, A2);
    }
    if (m_param.transposeB) {
        std::swap(B0, B1);
    }
    C = TensorLayout(TensorShape({A.shape[0], A1, B1}), A.dtype);
}

inline void matmul_exec(
        std::unique_ptr<MatrixMulForward>& opr, _megdnn_tensor_in A,
        _megdnn_tensor_in B, _megdnn_tensor_out C, _megdnn_workspace workspace) {
    auto Batch = A.layout.shape[0];

    auto Astrd = A.layout.dtype.size() * A.layout.stride[0],
         Cstrd = C.layout.dtype.size() * C.layout.stride[0];

    auto Aref = A.get_ref_ptr();
    auto Bref = B.get_ref_ptr();
    auto Cref = C.get_ref_ptr();

    rep(b, Batch) {
        //! all tensors should share the same RefPtr
        auto A_ref = Aref;
        A_ref += b * Astrd;
        auto B_ref = Bref;
        auto C_ref = Cref;
        C_ref += b * Cstrd;
        TensorND A_{A.layout.remove_axis(0), A_ref};
        TensorND B_{B.layout, B_ref};
        TensorND C_{C.layout.remove_axis(0), C_ref};
        opr->exec(A_, B_, C_, workspace);
    }
}

using Param = MultiHeadAttnBase::Param;
using MaskType = Param::AttnMaskType;
using InputType = Param::TensorCombinationType;

/***************************** MHA base *****************************/
#define _MHA_FORWARD(INPUT_TYPE, OUTPUT_TYPE)                                     \
    INPUT_TYPE queries, INPUT_TYPE keys, INPUT_TYPE values,                       \
            INPUT_TYPE qkvo_weight_bias, INPUT_TYPE attn_mask, INPUT_TYPE bias_k, \
            INPUT_TYPE bias_v, OUTPUT_TYPE out, OUTPUT_TYPE attn_weight,          \
            OUTPUT_TYPE mask_reservespace, OUTPUT_TYPE othr_reservespace
#define _MHA_BACKWARD(INPUT_TYPE, OUTPUT_TYPE)                                         \
    INPUT_TYPE diff, INPUT_TYPE queries, INPUT_TYPE keys, INPUT_TYPE values,           \
            INPUT_TYPE qkvo_weight_bias, INPUT_TYPE attn_mask, INPUT_TYPE attn_weight, \
            INPUT_TYPE mask_reservespace, INPUT_TYPE othr_reservespace,                \
            OUTPUT_TYPE dqueries, OUTPUT_TYPE dkeys, OUTPUT_TYPE dvalues,              \
            OUTPUT_TYPE dqkvo_weight_bias, OUTPUT_TYPE dbias_k, OUTPUT_TYPE dbias_v
#define _MHA_PROXY_PRE(HANDLE_TYPE, PARAM_TYPE) HANDLE_TYPE handle, PARAM_TYPE param

#define MHA_EXEC_PARAM(cb) \
    cb(_megdnn_tensor_in, _megdnn_tensor_out), _megdnn_workspace workspace
#define MHA_LAYOUT_CONST_PARAM(cb) cb(const TensorLayout&, const TensorLayout&)
#define MHA_LAYOUT_PARAM(cb)       cb(const TensorLayout&, TensorLayout&)
#define MHA_CALL(cb)               cb(, )
#define MHA_PROXY_PRE_PARAM        _MHA_PROXY_PRE(Handle*, Param&)
#define MHA_PROXY_PRE_CALL         _MHA_PROXY_PRE(, )

/***************************** MHA forward *****************************/
#define MHA_FORWARD_EXEC_PARAM         MHA_EXEC_PARAM(_MHA_FORWARD)
#define MHA_FORWARD_LAYOUT_CONST_PARAM MHA_LAYOUT_CONST_PARAM(_MHA_FORWARD)
#define MHA_FORWARD_LAYOUT_PARAM       MHA_LAYOUT_PARAM(_MHA_FORWARD)
#define MHA_FORWARD_CALL               MHA_CALL(_MHA_FORWARD)

#define MHA_PROXY_FORWARD_EXEC_PARAM   MHA_PROXY_PRE_PARAM, MHA_FORWARD_EXEC_PARAM
#define MHA_PROXY_FORWARD_LAYOUT_PARAM MHA_PROXY_PRE_PARAM, MHA_FORWARD_LAYOUT_PARAM
#define MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM \
    MHA_PROXY_PRE_PARAM, MHA_FORWARD_LAYOUT_CONST_PARAM
#define MHA_PROXY_FORWARD_CALL MHA_PROXY_PRE_CALL, MHA_FORWARD_CALL

/***************************** MHA backward *****************************/
#define MHA_BACKWARD_EXEC_PARAM         MHA_EXEC_PARAM(_MHA_BACKWARD)
#define MHA_BACKWARD_LAYOUT_CONST_PARAM MHA_LAYOUT_CONST_PARAM(_MHA_BACKWARD)
#define MHA_BACKWARD_LAYOUT_PARAM       MHA_LAYOUT_PARAM(_MHA_BACKWARD)
#define MHA_BACKWARD_CALL               MHA_CALL(_MHA_BACKWARD)

#define MHA_PROXY_BACKWARD_EXEC_PARAM   MHA_PROXY_PRE_PARAM, MHA_BACKWARD_EXEC_PARAM
#define MHA_PROXY_BACKWARD_LAYOUT_PARAM MHA_PROXY_PRE_PARAM, MHA_BACKWARD_LAYOUT_PARAM
#define MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM \
    MHA_PROXY_PRE_PARAM, MHA_BACKWARD_LAYOUT_CONST_PARAM
#define MHA_PROXY_BACKWARD_CALL MHA_PROXY_PRE_CALL, MHA_BACKWARD_CALL

/***************************** MHA other *****************************/
#define MHA_FORWARD_TENSOR_TO_LAYOUT_CALL                                \
    queries.layout, keys.layout, values.layout, qkvo_weight_bias.layout, \
            attn_mask.layout, bias_k.layout, bias_v.layout, out.layout,  \
            attn_weight.layout, mask_reservespace.layout, othr_reservespace.layout
#define MHA_BACKWARD_TENSOR_TO_LAYOUT_CALL                                            \
    diff.layout, queries.layout, keys.layout, values.layout, qkvo_weight_bias.layout, \
            attn_mask.layout, attn_weight.layout, mask_reservespace.layout,           \
            othr_reservespace.layout, dqueries.layout, dkeys.layout, dvalues.layout,  \
            dqkvo_weight_bias.layout, dbias_k.layout, dbias_v.layout
#define MHA_PROXY_FORWARD_TENSOR_TO_LAYOUT_CALL \
    MHA_PROXY_PRE_CALL, MHA_FORWARD_TENSOR_TO_LAYOUT_CALL
#define MHA_PROXY_BACKWARD_TENSOR_TO_LAYOUT_CALL \
    MHA_PROXY_PRE_CALL, MHA_BACKWARD_TENSOR_TO_LAYOUT_CALL

}  // namespace multi_head_attn
}  // namespace megdnn
