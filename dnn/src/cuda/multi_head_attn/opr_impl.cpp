#include "src/cuda/multi_head_attn/opr_impl.h"
#include "src/common/utils.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void MultiHeadAttnForwardImpl::deduce_layout(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& bias_k,
        const TensorLayout& bias_v, TensorLayout& out, TensorLayout& attn_weight,
        TensorLayout& mask_reservespace, TensorLayout& othr_reservespace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    return;
#else
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    auto p = param();
    megdnn_assert(
            queries.ndim == 3, "queries.ndim should be 3, but got %zu", queries.ndim);

    if (!desc_status.is_initialized(p, queries, keys, values))
        desc_status.set(cudnn_handle(this->handle()), p, queries, keys, values);

    auto input_type = p.tensor_combination_type;
    using INPUT_TYPE = Param::TENSOR_COMBINATION_TYPE;
    bool have_biaskv =
            input_type == INPUT_TYPE::ONLY_BIASKV or input_type == INPUT_TYPE::ALL;
    size_t attn_seqk_dim_add = (have_biaskv ? 1 : 0) + (p.add_zero_attn ? 1 : 0);
    attn_weight = TensorLayout(
            TensorShape{
                    queries.shape[0] * p.num_heads, queries.shape[1],
                    keys.shape[1] + attn_seqk_dim_add},
            queries.dtype);

    size_t osize = p.oproj_size != 0 ? p.oproj_size
                                     : (p.vproj_size != 0 ? p.vproj_size : p.v_size);
    out = TensorLayout(
            TensorShape{queries.shape[0], queries.shape[1], osize}, queries.dtype);
    mask_reservespace = TensorLayout(TensorShape{0}, dtype::Uint8());
    othr_reservespace =
            TensorLayout(TensorShape{desc_status.sizeReserve}, queries.dtype);

#endif
}

size_t MultiHeadAttnForwardImpl::get_workspace_in_bytes(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& bias_k,
        const TensorLayout& bias_v, const TensorLayout& out,
        const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
        const TensorLayout& othr_reservespace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    return 0;
#else
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);

    if (!desc_status.is_initialized(param(), queries, keys, values))
        desc_status.set(cudnn_handle(this->handle()), param(), queries, keys, values);

    return desc_status.sizeWkspace;
#endif
}

size_t MultiHeadAttnForwardImpl::get_mask_reservespace_in_bytes(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& bias_k,
        const TensorLayout& bias_v, const TensorLayout& out,
        const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
        const TensorLayout& othr_reservespace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    return 0;
#else
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    if (!desc_status.is_initialized(param(), queries, keys, values))
        desc_status.set(cudnn_handle(this->handle()), param(), queries, keys, values);
    return 0;
#endif
}
size_t MultiHeadAttnForwardImpl::get_othr_reservespace_in_bytes(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& bias_k,
        const TensorLayout& bias_v, const TensorLayout& out,
        const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
        const TensorLayout& othr_reservespace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    return 0;
#else
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    if (!desc_status.is_initialized(param(), queries, keys, values))
        desc_status.set(cudnn_handle(this->handle()), param(), queries, keys, values);
    return desc_status.sizeReserve;
#endif
}

void MultiHeadAttnForwardImpl::exec(
        _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
        _megdnn_tensor_in qkvo_weight_bias, _megdnn_tensor_in attn_mask,
        _megdnn_tensor_in bias_k, _megdnn_tensor_in bias_v, _megdnn_tensor_out out,
        _megdnn_tensor_out attn_weight, _megdnn_tensor_out mask_reservespace,
        _megdnn_tensor_out othr_reservespace, _megdnn_workspace workspace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    megdnn_throw(
            "The cudnn version is lower than 8.0.4. Please upgrade the cudnn version.");
#else
    check_exec(
            queries.layout, keys.layout, values.layout, qkvo_weight_bias.layout,
            attn_mask.layout, bias_k.layout, bias_v.layout, out.layout,
            attn_weight.layout, mask_reservespace.layout, othr_reservespace.layout,
            workspace.size);
    auto p = param();

    if (!desc_status.is_initialized(p, queries.layout, keys.layout, values.layout))
        desc_status.set(
                cudnn_handle(this->handle()), p, queries.layout, keys.layout,
                values.layout);

    size_t osize =
            desc_status.oProjSize != 0
                    ? desc_status.oProjSize
                    : (desc_status.vProjSize != 0 ? desc_status.vProjSize * p.num_heads
                                                  : desc_status.vSize);
    SeqTensorDesc q{queries.layout,      desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.qSize,
                    p.input_order,       desc_status.auxArray.seqQArray};
    SeqTensorDesc o{out.layout, desc_status.batchSize, desc_status.seqLenQ,
                    osize,      p.input_order,         desc_status.auxArray.seqQArray};
    SeqTensorDesc k{keys.layout,         desc_status.batchSize,
                    desc_status.seqLenK, desc_status.kSize,
                    p.input_order,       desc_status.auxArray.seqKArray};
    SeqTensorDesc v{values.layout,       desc_status.batchSize,
                    desc_status.seqLenK, desc_status.vSize,
                    p.input_order,       desc_status.auxArray.seqKArray};

    cudnn_check(cudnnMultiHeadAttnForward(
            cudnn_handle(this->handle()), desc_status.attn_desc, -1,
            desc_status.auxArray.loWinIdx, desc_status.auxArray.hiWinIdx,
            desc_status.auxArray.devSeqQArray, desc_status.auxArray.devSeqKArray,
            q.desc, queries.raw_ptr(), p.reslink ? queries.raw_ptr() : NULL, k.desc,
            keys.raw_ptr(), v.desc, values.raw_ptr(), o.desc, out.raw_ptr(),
            desc_status.sizeWeights,
            desc_status.sizeWeights > 0 ? qkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr,
            p.training ? desc_status.sizeReserve : 0,
            p.training ? othr_reservespace.raw_ptr() : NULL));
#endif
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
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004 and param().bias = true, we need to go to the proxy
    // cuda implementation.
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    MEGDNN_MARK_USED_VAR(dqueries);
    MEGDNN_MARK_USED_VAR(dkeys);
    MEGDNN_MARK_USED_VAR(dvalues);
    MEGDNN_MARK_USED_VAR(dqkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(dbias_k);
    MEGDNN_MARK_USED_VAR(dbias_v);
    megdnn_throw(
            "The cudnn version is lower than 8.0.4. Please upgrade the cudnn version.");
#else
#if CUDNN_VERSION < 8600
    megdnn_assert(
            !(param().qbias or param().kbias or param().vbias or param().obias),
            "If the cudnn version is lower than 8.6.0, param().bias must be false, "
            "but got true, because there is an error in the "
            "dbias result during the backward calculation.");
#endif
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(dbias_k);
    MEGDNN_MARK_USED_VAR(dbias_v);

    check_exec(
            diff.layout, queries.layout, keys.layout, values.layout,
            qkvo_weight_bias.layout, attn_mask.layout, attn_weight.layout,
            mask_reservespace.layout, othr_reservespace.layout, dqueries.layout,
            dkeys.layout, dvalues.layout, dqkvo_weight_bias.layout, dbias_k.layout,
            dbias_v.layout, workspace.size);
    auto p = param();

    if (!desc_status.is_initialized(p, queries.layout, keys.layout, values.layout))
        desc_status.set(
                cudnn_handle(this->handle()), p, queries.layout, keys.layout,
                values.layout);

    size_t osize =
            desc_status.oProjSize != 0
                    ? desc_status.oProjSize
                    : (desc_status.vProjSize != 0 ? desc_status.vProjSize * p.num_heads
                                                  : desc_status.vSize);
    SeqTensorDesc q{queries.layout,      desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.qSize,
                    p.input_order,       desc_status.auxArray.seqQArray};
    SeqTensorDesc d{diff.layout, desc_status.batchSize, desc_status.seqLenQ,
                    osize,       p.input_order,         desc_status.auxArray.seqQArray};
    SeqTensorDesc k{keys.layout,         desc_status.batchSize,
                    desc_status.seqLenK, desc_status.kSize,
                    p.input_order,       desc_status.auxArray.seqKArray};
    SeqTensorDesc v{values.layout,       desc_status.batchSize,
                    desc_status.seqLenK, desc_status.vSize,
                    p.input_order,       desc_status.auxArray.seqKArray};

    cudnn_check(cudnnMultiHeadAttnBackwardData(
            cudnn_handle(this->handle()), desc_status.attn_desc,
            desc_status.auxArray.loWinIdx, desc_status.auxArray.hiWinIdx,
            desc_status.auxArray.devSeqQArray, desc_status.auxArray.devSeqKArray,
            d.desc, diff.raw_ptr(), q.desc, dqueries.raw_ptr(), queries.raw_ptr(),
            k.desc, dkeys.raw_ptr(), keys.raw_ptr(), v.desc, dvalues.raw_ptr(),
            values.raw_ptr(), desc_status.sizeWeights,
            desc_status.sizeWeights > 0 ? qkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr, desc_status.sizeReserve,
            othr_reservespace.raw_ptr()));

    cuda_check(cudaMemset(dqkvo_weight_bias.raw_ptr(), 0, desc_status.sizeWeights));
#if CUDNN_VERSION < 8600
    cuda_check(cudaDeviceSynchronize());
#endif
    cudnn_check(cudnnMultiHeadAttnBackwardWeights(
            cudnn_handle(this->handle()), desc_status.attn_desc, CUDNN_WGRAD_MODE_ADD,
            q.desc, queries.raw_ptr(), k.desc, keys.raw_ptr(), v.desc, values.raw_ptr(),
            d.desc, diff.raw_ptr(), desc_status.sizeWeights,
            desc_status.sizeWeights > 0 ? qkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWeights > 0 ? dqkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr, desc_status.sizeReserve,
            othr_reservespace.raw_ptr()));
#endif
}
size_t MultiHeadAttnBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& diff, const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& attn_weight,
        const TensorLayout& mask_reservespace, const TensorLayout& othr_reservespace,
        const TensorLayout& dqueries, const TensorLayout& dkeys,
        const TensorLayout& dvalues, const TensorLayout& dqkvo_weight_bias,
        const TensorLayout& dbias_k, const TensorLayout& dbias_v) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    MEGDNN_MARK_USED_VAR(dqueries);
    MEGDNN_MARK_USED_VAR(dkeys);
    MEGDNN_MARK_USED_VAR(dvalues);
    MEGDNN_MARK_USED_VAR(dqkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(dbias_k);
    MEGDNN_MARK_USED_VAR(dbias_v);
    return 0;
}
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
