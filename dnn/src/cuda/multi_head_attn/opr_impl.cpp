#include "src/cuda/multi_head_attn/opr_impl.h"
#include "src/common/utils.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void MultiHeadAttnForwardImpl::deduce_layout(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv, TensorLayout& out,
        TensorLayout& reserveSpace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    return;
#else
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(wqkv);
    megdnn_assert(
            queries.ndim == 3,
            "queries.ndim should be 3[batch, sequence, embeding], but got %zu",
            queries.ndim);

    if (!desc_status.is_initialized(param(), queries, keys, values)) {
        desc_status.set(cudnn_handle(this->handle()), param(), queries, keys, values);

        out = TensorLayout(
                TensorShape{queries.shape[0], queries.shape[1], queries.shape[2]},
                queries.dtype);
        reserveSpace =
                TensorLayout(TensorShape{desc_status.sizeReserve}, queries.dtype);
    }
#endif
}

size_t MultiHeadAttnForwardImpl::get_workspace_in_bytes(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv, const TensorLayout& out,
        const TensorLayout& reserveSpace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    return 0;
#else
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);

    if (!desc_status.is_initialized(param(), queries, keys, values))
        desc_status.set(cudnn_handle(this->handle()), param(), queries, keys, values);

    return desc_status.sizeWkspace;
#endif
}

size_t MultiHeadAttnForwardImpl::get_reservespace_in_bytes(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv, const TensorLayout& out,
        const TensorLayout& reserveSpace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    return 0;
#else
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    if (!desc_status.is_initialized(param(), queries, keys, values))
        desc_status.set(cudnn_handle(this->handle()), param(), queries, keys, values);
    return desc_status.sizeReserve;
#endif
}
void MultiHeadAttnForwardImpl::exec(
        _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
        _megdnn_tensor_in wqkv, _megdnn_tensor_out out, _megdnn_tensor_out reserveSpace,
        _megdnn_workspace workspace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004,  we need to go to the proxy cuda implementation.
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    MEGDNN_MARK_USED_VAR(workspace);
    megdnn_throw(
            "The cudnn version is lower than 8.0.4. Please upgrade the cudnn version.");
#else
    check_exec(
            queries.layout, keys.layout, values.layout, wqkv.layout, out.layout,
            reserveSpace.layout, workspace.size);
    auto p = param();

    if (!desc_status.is_initialized(p, queries.layout, keys.layout, values.layout))
        desc_status.set(
                cudnn_handle(this->handle()), p, queries.layout, keys.layout,
                values.layout);

    SeqTensorDesc q{queries.layout,      desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.qSize,
                    p.input_order,       desc_status.auxArray.seqQArray};
    SeqTensorDesc o{out.layout,          desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.oProjSize,
                    p.input_order,       desc_status.auxArray.seqQArray};
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
            desc_status.sizeWeights > 0 ? wqkv.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr,
            p.training ? desc_status.sizeReserve : 0,
            p.training ? reserveSpace.raw_ptr() : NULL));
#endif
}

void MultiHeadAttnBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
        _megdnn_tensor_in values, _megdnn_tensor_in wqkv,
        _megdnn_tensor_in reserveSpace, _megdnn_tensor_out dqueries,
        _megdnn_tensor_out dkeys, _megdnn_tensor_out dvalues,
        _megdnn_tensor_out dweights, _megdnn_workspace workspace) {
#if CUDNN_VERSION < 8004
    // TODO: CUDNN_VERSION < 8004 and param().bias = true, we need to go to the proxy
    // cuda implementation.
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    MEGDNN_MARK_USED_VAR(dqueries);
    MEGDNN_MARK_USED_VAR(dkeys);
    MEGDNN_MARK_USED_VAR(dvalues);
    MEGDNN_MARK_USED_VAR(dweights);
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

    check_exec(
            diff.layout, queries.layout, keys.layout, values.layout, wqkv.layout,
            reserveSpace.layout, dqueries.layout, dkeys.layout, dvalues.layout,
            dweights.layout, workspace.size);
    auto p = param();

    if (!desc_status.is_initialized(p, queries.layout, keys.layout, values.layout))
        desc_status.set(
                cudnn_handle(this->handle()), p, queries.layout, keys.layout,
                values.layout);

    SeqTensorDesc q{queries.layout,      desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.qSize,
                    p.input_order,       desc_status.auxArray.seqQArray};
    SeqTensorDesc d{diff.layout,         desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.oProjSize,
                    p.input_order,       desc_status.auxArray.seqQArray};
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
            desc_status.sizeWeights > 0 ? wqkv.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr, desc_status.sizeReserve,
            reserveSpace.raw_ptr()));

    cuda_check(cudaMemset(dweights.raw_ptr(), 0, desc_status.sizeWeights));
#if CUDNN_VERSION < 8600
    cuda_check(cudaDeviceSynchronize());
#endif
    cudnn_check(cudnnMultiHeadAttnBackwardWeights(
            cudnn_handle(this->handle()), desc_status.attn_desc, CUDNN_WGRAD_MODE_ADD,
            q.desc, queries.raw_ptr(), k.desc, keys.raw_ptr(), v.desc, values.raw_ptr(),
            d.desc, diff.raw_ptr(), desc_status.sizeWeights,
            desc_status.sizeWeights > 0 ? wqkv.raw_ptr() : NULL,
            desc_status.sizeWeights > 0 ? dweights.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr, desc_status.sizeReserve,
            reserveSpace.raw_ptr()));
#endif
}
size_t MultiHeadAttnBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& diff, const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv,
        const TensorLayout& reserveSpace, const TensorLayout& dqueries,
        const TensorLayout& dkeys, const TensorLayout& dvalues,
        const TensorLayout& dweights) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    MEGDNN_MARK_USED_VAR(dqueries);
    MEGDNN_MARK_USED_VAR(dkeys);
    MEGDNN_MARK_USED_VAR(dvalues);
    MEGDNN_MARK_USED_VAR(dweights);
    return 0;
}
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
