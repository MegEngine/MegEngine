#include "src/cuda/multi_head_attn/opr_impl.h"
#include "src/common/utils.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

bool can_use_mha_cudnn(const Param& param) {
#if CUDNN_VERSION < 8004
    MEGDNN_MARK_USED_VAR(param);
    return false;
#else
    bool flag = true;
    size_t bias_num = 0;
    size_t weight_num = 0;
    bias_num += (param.qbias ? 1 : 0);
    bias_num += (param.kbias ? 1 : 0);
    bias_num += (param.vbias ? 1 : 0);
    bias_num += (param.obias ? 1 : 0);
    weight_num += (param.qproj_size > 0 ? 1 : 0);
    weight_num += (param.kproj_size > 0 ? 1 : 0);
    weight_num += (param.vproj_size > 0 ? 1 : 0);
    weight_num += (param.oproj_size > 0 ? 1 : 0);
    if (bias_num != weight_num && bias_num != 0) {
        flag = false;
    }
#if CUDNN_VERSION < 8600
    if (bias_num > 0 && param.training == true) {
        flag = false;
    }
    if (param.out_prob > 0) {
        flag = false;
    }
#endif
    if (param.need_weights) {
        flag = false;
    }
    if (param.attn_mask_type == MaskType::USER_DEFINED_MASK) {
        flag = false;
    }
    if (param.attn_mask_type == MaskType::CUDNN_STYLE_MASK) {
        megdnn_assert(
                flag == true,
                "maybe_cudnn_style_mask=True, but can not run cudnn impl, Please make "
                "sure that cuda is available, and check you parameter or do not use "
                "cudnn style mask.");
    }
    return flag;
#endif
}

void MultiHeadAttnForwardImpl::deduce_layout(MHA_FORWARD_LAYOUT_PARAM) {
    Param p = param();
#if CUDNN_VERSION < 8004
    proxy_opr.deduce_layout(
            this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
            bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
#else
    if (can_use_mha_cudnn(p)) {
        cudnn_opr.deduce_layout(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    } else {
        proxy_opr.deduce_layout(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    }
#endif
}

size_t MultiHeadAttnForwardImpl::get_workspace_in_bytes(
        MHA_FORWARD_LAYOUT_CONST_PARAM) {
    Param p = param();
#if CUDNN_VERSION < 8004
    return proxy_opr.get_workspace_in_bytes(
            this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
            bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
#else
    if (can_use_mha_cudnn(p)) {
        return cudnn_opr.get_workspace_in_bytes(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    } else {
        return proxy_opr.get_workspace_in_bytes(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    }
#endif
}

size_t MultiHeadAttnForwardImpl::get_mask_reservespace_in_bytes(
        MHA_FORWARD_LAYOUT_CONST_PARAM) {
    Param p = param();
#if CUDNN_VERSION < 8004
    return proxy_opr.get_mask_reservespace_in_bytes(
            this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
            bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
#else
    if (can_use_mha_cudnn(p)) {
        return cudnn_opr.get_mask_reservespace_in_bytes(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    } else {
        return proxy_opr.get_mask_reservespace_in_bytes(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    }
#endif
}

size_t MultiHeadAttnForwardImpl::get_othr_reservespace_in_bytes(
        MHA_FORWARD_LAYOUT_CONST_PARAM) {
    Param p = param();
#if CUDNN_VERSION < 8004
    return proxy_opr.get_othr_reservespace_in_bytes(
            this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
            bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
#else
    if (can_use_mha_cudnn(p)) {
        return cudnn_opr.get_othr_reservespace_in_bytes(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    } else {
        return proxy_opr.get_othr_reservespace_in_bytes(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace);
    }
#endif
}

void MultiHeadAttnForwardImpl::exec(MHA_FORWARD_EXEC_PARAM) {
    check_exec(
            queries.layout, keys.layout, values.layout, qkvo_weight_bias.layout,
            attn_mask.layout, bias_k.layout, bias_v.layout, out.layout,
            attn_weight.layout, mask_reservespace.layout, othr_reservespace.layout,
            workspace.size);
    Param p = param();
#if CUDNN_VERSION < 8004
    proxy_opr.exec(
            this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
            bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace,
            workspace);
#else
    if (can_use_mha_cudnn(p)) {
        cudnn_opr.exec(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace,
                workspace);
    } else {
        proxy_opr.exec(
                this->handle(), p, queries, keys, values, qkvo_weight_bias, attn_mask,
                bias_k, bias_v, out, attn_weight, mask_reservespace, othr_reservespace,
                workspace);
    }
#endif
}

void MultiHeadAttnBackwardImpl::exec(MHA_BACKWARD_EXEC_PARAM) {
    check_exec(
            diff.layout, queries.layout, keys.layout, values.layout,
            qkvo_weight_bias.layout, attn_mask.layout, attn_weight.layout,
            mask_reservespace.layout, othr_reservespace.layout, dqueries.layout,
            dkeys.layout, dvalues.layout, dqkvo_weight_bias.layout, dbias_k.layout,
            dbias_v.layout, workspace.size);
    Param p = param();
#if CUDNN_VERSION < 8004
    proxy_opr.exec(
            this->handle(), p, diff, queries, keys, values, qkvo_weight_bias, attn_mask,
            attn_weight, mask_reservespace, othr_reservespace, dqueries, dkeys, dvalues,
            dqkvo_weight_bias, dbias_k, dbias_v, workspace);
#else
    if (can_use_mha_cudnn(p)) {
        cudnn_opr.exec(
                this->handle(), p, diff, queries, keys, values, qkvo_weight_bias,
                attn_mask, attn_weight, mask_reservespace, othr_reservespace, dqueries,
                dkeys, dvalues, dqkvo_weight_bias, dbias_k, dbias_v, workspace);
    } else {
        proxy_opr.exec(
                this->handle(), p, diff, queries, keys, values, qkvo_weight_bias,
                attn_mask, attn_weight, mask_reservespace, othr_reservespace, dqueries,
                dkeys, dvalues, dqkvo_weight_bias, dbias_k, dbias_v, workspace);
    }
#endif
}

size_t MultiHeadAttnBackwardImpl::get_workspace_in_bytes(
        MHA_BACKWARD_LAYOUT_CONST_PARAM) {
    Param p = param();
    if (can_use_mha_cudnn(p)) {
        return 0;
    } else {
        return proxy_opr.get_workspace_in_bytes(
                this->handle(), p, diff, queries, keys, values, qkvo_weight_bias,
                attn_mask, attn_weight, mask_reservespace, othr_reservespace, dqueries,
                dkeys, dvalues, dqkvo_weight_bias, dbias_k, dbias_v);
    }
}
}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
