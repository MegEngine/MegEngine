#include "src/cuda/multi_head_attn/cudnn_fwbw.h"
#include <vector>
#include "megdnn/handle.h"
#include "src/cuda/utils.h"
#if CUDNN_VERSION >= 8004
#include "megdnn/dtype.h"

namespace megdnn {
namespace cuda {

/***************************** AuxiliaryArray *****************************/
AuxiliaryArray::~AuxiliaryArray() {
    if (attnMaskType != MaskType::CUDNN_STYLE_MASK) {
        if (devSeqQArray) {
            cuda_check(cudaFree(devSeqQArray));
        }
        if (devSeqKArray) {
            cuda_check(cudaFree(devSeqKArray));
        }
    }
}

bool AuxiliaryArray::is_initialized(
        const size_t _batchSize, const size_t _seqLenQ, const size_t _seqLenK,
        MaskType _attnMaskType) {
    if (_batchSize != batchSize or _seqLenQ != seqLenQ or _seqLenK != seqLenK or
        _attnMaskType != attnMaskType or (seqQArray.size() != _batchSize) or
        (seqKArray.size() != _batchSize) or !devSeqQArray or !devSeqKArray or
        (loWinIdx.size() != _seqLenQ) or (hiWinIdx.size() != _seqLenQ)) {
        return false;
    }
    return true;
}

void AuxiliaryArray::set_cudnn_style_mask(Handle* handle, const TensorND& attn_mask) {
    megdnn_assert(attnMaskType == MaskType::CUDNN_STYLE_MASK);
    auto stream = cuda_stream(handle);
#define T DTypeTrait<::megdnn::dtype::Int32>::ctype
    devSeqQArray = attn_mask.ptr<T>() + 2 * seqLenQ;
    devSeqKArray = attn_mask.ptr<T>() + 2 * seqLenQ + batchSize;
    cuda_check(cudaMemcpyAsync(
            seqQArray.data(), devSeqQArray, batchSize * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(
            seqKArray.data(), devSeqKArray, batchSize * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(
            loWinIdx.data(), attn_mask.ptr<T>(), seqLenQ * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(
            hiWinIdx.data(), attn_mask.ptr<T>() + seqLenQ, seqLenQ * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
#undef T
}

void AuxiliaryArray::set(
        Handle* handle, const size_t _batchSize, const size_t _seqLenQ,
        const size_t _seqLenK, MaskType _attnMaskType) {
    if (_batchSize == batchSize && _seqLenQ == seqLenQ && _seqLenK == seqLenK &&
        _attnMaskType == attnMaskType) {
        return;
    } else {
        if (attnMaskType != MaskType::CUDNN_STYLE_MASK) {
            if (devSeqQArray) {
                cuda_check(cudaFree(devSeqQArray));
            }
            if (devSeqKArray) {
                cuda_check(cudaFree(devSeqKArray));
            }
        }
    };
    seqLenQ = _seqLenQ;
    seqLenK = _seqLenK;
    batchSize = _batchSize;
    attnMaskType = _attnMaskType;
    loWinIdx.resize(seqLenQ);
    hiWinIdx.resize(seqLenQ);
    size_t seqQArraySize = 1 * batchSize;
    size_t seqKArraySize = batchSize;
    seqQArray.resize(seqQArraySize);
    seqKArray.resize(seqKArraySize);
    if (attnMaskType == MaskType::CUDNN_STYLE_MASK) {
        return;
    }
    for (size_t i = 0; i < seqQArraySize; ++i) {
        seqQArray[i] = seqLenQ;
    }
    for (size_t i = 0; i < seqKArraySize; ++i) {
        seqKArray[i] = seqLenK;
    }

    cuda_check(cudaMalloc((void**)&devSeqQArray, seqQArraySize * sizeof(int)));
    cuda_check(cudaMalloc((void**)&devSeqKArray, seqKArraySize * sizeof(int)));

    auto stream = cuda_stream(handle);
    cuda_check(cudaMemcpyAsync(
            devSeqQArray, seqQArray.data(), seqQArraySize * sizeof(int),
            cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(
            devSeqKArray, seqKArray.data(), seqKArraySize * sizeof(int),
            cudaMemcpyHostToDevice, stream));

    for (size_t i = 0; i < seqLenQ; ++i) {
        loWinIdx[i] = 0;
        if (attnMaskType == MaskType::DEFAULT_MASK) {
            hiWinIdx[i] = i + 1;
        } else if (attnMaskType == MaskType::NO_MASK) {
            hiWinIdx[i] = seqLenK;
        }
    }
}

/***************************** MultiHeadAttnStatus *****************************/
void MultiHeadAttnStatus::set(
        Handle* handle, const Param& p, const TensorLayout& q, const TensorLayout& k,
        const TensorLayout& v) {
    // It is consistent with the conditions judged in is_initialized.
    // dropout
    float attn_prob = p.training ? p.attn_prob : 0.f;
    float out_prob = p.training ? p.out_prob : 0.f;
    if (!attn_dropout_status.initialized()) {
        attn_dropout_status.set(cudnn_handle(handle), p.seed, attn_prob);
    }
    if (!out_dropout_status.initialized()) {
        out_dropout_status.set(cudnn_handle(handle), p.seed, out_prob);
    }

    if (attn_dropout_status.drop_prob != attn_prob) {
        attn_dropout_status.drop_prob = attn_prob;
        attn_dropout_status.restore_desc(cudnn_handle(handle));
    }
    if (out_dropout_status.drop_prob != out_prob) {
        out_dropout_status.drop_prob = out_prob;
        out_dropout_status.restore_desc(cudnn_handle(handle));
    }
    // size
    batchSize = q.shape[0];
    seqLenQ = q.shape[1];
    seqLenK = k.shape[1];
    numHeads = p.num_heads;
    qSize = p.embeding_size;
    kSize = p.k_size;
    vSize = p.v_size;
    qProjSize = p.qproj_size / numHeads;
    kProjSize = p.kproj_size / numHeads;
    vProjSize = p.vproj_size / numHeads;
    oProjSize = p.oproj_size;
    attnMaskType = p.attn_mask_type;
    bias = p.qbias or p.kbias or p.vbias or p.obias;
    cudnnDataType_t cudnn_dtype = to_cudnn_dtype(q.dtype);
    auto flag = CUDNN_ATTN_QUERYMAP_ONE_TO_ONE;
    if (bias) {
        flag = flag | CUDNN_ATTN_ENABLE_PROJ_BIASES;
    }
#if CUDNN_VERSION < 8600
    cudnn_check(cudnnSetAttnDescriptor(
            attn_desc, flag, numHeads, p.sm_scaler, cudnn_dtype, cudnn_dtype,
            CUDNN_DEFAULT_MATH, attn_dropout_status.desc.desc, NULL, qSize, kSize,
            vSize, qProjSize, kProjSize, vProjSize, oProjSize, seqLenQ, seqLenK,
            batchSize, 1));
#else
    cudnn_check(cudnnSetAttnDescriptor(
            attn_desc, flag, numHeads, p.sm_scaler, cudnn_dtype, cudnn_dtype,
            CUDNN_DEFAULT_MATH, attn_dropout_status.desc.desc,
            out_dropout_status.desc.desc, qSize, kSize, vSize, qProjSize, kProjSize,
            vProjSize, oProjSize, seqLenQ, seqLenK, batchSize, 1));
#endif

    // misc
    auxArray.set(handle, batchSize, seqLenQ, seqLenK, p.attn_mask_type);

    if (p.training) {
        cudnnGetMultiHeadAttnBuffers(
                cudnn_handle(handle), attn_desc, &sizeWeights, &sizeWkspace,
                &sizeReserve);
    } else {
        cudnnGetMultiHeadAttnBuffers(
                cudnn_handle(handle), attn_desc, &sizeWeights, &sizeWkspace, NULL);
        sizeReserve = 0;
    }
}

void MultiHeadAttnStatus::set_cudnn_style_mask(
        Handle* handle, const TensorND& attn_mask) {
    auxArray.set_cudnn_style_mask(handle, attn_mask);
}

bool MultiHeadAttnStatus::is_initialized(
        const Param& p, const TensorLayout& q, const TensorLayout& k,
        const TensorLayout& v) {
    // By default, the info of q, k and v must be consistent with the corresponding
    // parameters in param respectively, otherwise an error will occur (so, check is not
    // done here, mainly by check_exec).
    // dropout
    float attn_prob = p.training ? p.attn_prob : 0.f;
    float out_prob = p.training ? p.out_prob : 0.f;
    if (!attn_dropout_status.initialized() or !out_dropout_status.initialized() or
        attn_dropout_status.drop_prob != attn_prob or
        out_dropout_status.drop_prob != out_prob) {
        return false;
    }
    // size
    if (q.shape[0] != batchSize or q.shape[1] != seqLenQ or k.shape[1] != seqLenK or
        q.shape[2] != qSize or k.shape[2] != kSize or v.shape[2] != vSize or
        attnMaskType != p.attn_mask_type or numHeads != p.num_heads) {
        return false;
    }
    bool pbias = p.qbias or p.kbias or p.vbias or p.obias;
    if (qSize != p.embeding_size or kSize != p.k_size or vSize != p.v_size) {
        return false;
    }
    if (bias != pbias) {
        return false;
    }
    if ((qProjSize != (p.qproj_size / p.num_heads)) or
        (kProjSize != (p.kproj_size / p.num_heads)) or
        (vProjSize != (p.vproj_size / p.num_heads)) or (oProjSize != p.oproj_size)) {
        return false;
    }
    // misc
    if (!auxArray.is_initialized(batchSize, seqLenQ, seqLenK, attnMaskType)) {
        return false;
    }
    if (p.training and sizeReserve == 0) {
        return false;
    }
    return true;
}

/***************************** MHA forward *****************************/
void MHAForwardCudnnOpr::deduce_layout(MHA_PROXY_FORWARD_LAYOUT_PARAM) {
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    megdnn_assert(
            queries.ndim == 3,
            "queries.ndim should be 3[batch, sequence, embeding], but got %zu",
            queries.ndim);
    if (!desc_status.is_initialized(param, queries, keys, values)) {
        desc_status.set(handle, param, queries, keys, values);
    }

    attn_weight = TensorLayout(
            TensorShape{
                    queries.shape[0] * param.num_heads, queries.shape[1],
                    keys.shape[1]},
            queries.dtype);
    size_t osize = param.oproj_size != 0
                         ? param.oproj_size
                         : (param.vproj_size != 0 ? param.vproj_size
                                                  : (param.v_size * param.num_heads));
    out = TensorLayout(
            TensorShape{queries.shape[0], queries.shape[1], osize}, queries.dtype);
    mask_reservespace = TensorLayout(TensorShape{0}, dtype::Uint8());
    othr_reservespace = TensorLayout(
            TensorShape{desc_status.sizeReserve / queries.dtype.size()}, queries.dtype);
}

void MHAForwardCudnnOpr::exec(MHA_PROXY_FORWARD_EXEC_PARAM) {
    if (!desc_status.is_initialized(
                param, queries.layout, keys.layout, values.layout)) {
        desc_status.set(handle, param, queries.layout, keys.layout, values.layout);
    }
    if (param.attn_mask_type == MaskType::CUDNN_STYLE_MASK) {
        desc_status.set_cudnn_style_mask(handle, attn_mask);
    }

    size_t osize = desc_status.oProjSize != 0
                         ? desc_status.oProjSize
                         : (desc_status.vProjSize != 0
                                    ? desc_status.vProjSize * param.num_heads
                                    : desc_status.vSize * param.num_heads);
    SeqTensorDesc q{queries.layout,      desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.qSize,
                    param.input_order,   desc_status.auxArray.seqQArray.data()};
    SeqTensorDesc o{out.layout,          desc_status.batchSize,
                    desc_status.seqLenQ, osize,
                    param.input_order,   desc_status.auxArray.seqQArray.data()};
    SeqTensorDesc k{keys.layout,         desc_status.batchSize,
                    desc_status.seqLenK, desc_status.kSize,
                    param.input_order,   desc_status.auxArray.seqKArray.data()};
    SeqTensorDesc v{values.layout,       desc_status.batchSize,
                    desc_status.seqLenK, desc_status.vSize,
                    param.input_order,   desc_status.auxArray.seqKArray.data()};

    cudnn_check(cudnnMultiHeadAttnForward(
            cudnn_handle(handle), desc_status.attn_desc, -1,
            desc_status.auxArray.loWinIdx.data(), desc_status.auxArray.hiWinIdx.data(),
            desc_status.auxArray.devSeqQArray, desc_status.auxArray.devSeqKArray,
            q.desc, queries.raw_ptr(), param.reslink ? queries.raw_ptr() : NULL, k.desc,
            keys.raw_ptr(), v.desc, values.raw_ptr(), o.desc, out.raw_ptr(),
            desc_status.sizeWeights,
            desc_status.sizeWeights > 0 ? qkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr,
            param.training ? desc_status.sizeReserve : 0,
            param.training ? othr_reservespace.raw_ptr() : NULL));
}

size_t MHAForwardCudnnOpr::get_workspace_in_bytes(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);

    if (!desc_status.is_initialized(param, queries, keys, values)) {
        desc_status.set(handle, param, queries, keys, values);
    }

    return desc_status.sizeWkspace;
}

size_t MHAForwardCudnnOpr::get_mask_reservespace_in_bytes(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    if (!desc_status.is_initialized(param, queries, keys, values)) {
        desc_status.set(handle, param, queries, keys, values);
    }
    return 0;
}

size_t MHAForwardCudnnOpr::get_othr_reservespace_in_bytes(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    if (!desc_status.is_initialized(param, queries, keys, values)) {
        desc_status.set(handle, param, queries, keys, values);
    }

    return desc_status.sizeReserve;
}

/***************************** MHA backward *****************************/
void MHABackwardCudnnOpr::exec(MHA_PROXY_BACKWARD_EXEC_PARAM) {
    if (!desc_status.is_initialized(
                param, queries.layout, keys.layout, values.layout)) {
        desc_status.set(handle, param, queries.layout, keys.layout, values.layout);
    }
    if (param.attn_mask_type == MaskType::CUDNN_STYLE_MASK) {
        desc_status.set_cudnn_style_mask(handle, attn_mask);
    }

    size_t osize = desc_status.oProjSize != 0
                         ? desc_status.oProjSize
                         : (desc_status.vProjSize != 0
                                    ? (desc_status.vProjSize * param.num_heads)
                                    : (desc_status.vSize * param.num_heads));
    SeqTensorDesc q{queries.layout,      desc_status.batchSize,
                    desc_status.seqLenQ, desc_status.qSize,
                    param.input_order,   desc_status.auxArray.seqQArray.data()};
    SeqTensorDesc d{diff.layout,         desc_status.batchSize,
                    desc_status.seqLenQ, osize,
                    param.input_order,   desc_status.auxArray.seqQArray.data()};
    SeqTensorDesc k{keys.layout,         desc_status.batchSize,
                    desc_status.seqLenK, desc_status.kSize,
                    param.input_order,   desc_status.auxArray.seqKArray.data()};
    SeqTensorDesc v{values.layout,       desc_status.batchSize,
                    desc_status.seqLenK, desc_status.vSize,
                    param.input_order,   desc_status.auxArray.seqKArray.data()};

    cudnn_check(cudnnMultiHeadAttnBackwardData(
            cudnn_handle(handle), desc_status.attn_desc,
            desc_status.auxArray.loWinIdx.data(), desc_status.auxArray.hiWinIdx.data(),
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
            cudnn_handle(handle), desc_status.attn_desc, CUDNN_WGRAD_MODE_ADD, q.desc,
            queries.raw_ptr(), k.desc, keys.raw_ptr(), v.desc, values.raw_ptr(), d.desc,
            diff.raw_ptr(), desc_status.sizeWeights,
            desc_status.sizeWeights > 0 ? qkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWeights > 0 ? dqkvo_weight_bias.raw_ptr() : NULL,
            desc_status.sizeWkspace, workspace.raw_ptr, desc_status.sizeReserve,
            othr_reservespace.raw_ptr()));
}
}  // namespace cuda
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
