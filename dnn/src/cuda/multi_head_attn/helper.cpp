#include "src/cuda/multi_head_attn/helper.h"
#if CUDNN_VERSION >= 8004

namespace megdnn {
namespace cuda {

AuxiliaryArray::~AuxiliaryArray() {
    if (loWinIdx)
        free(loWinIdx);
    if (hiWinIdx)
        free(hiWinIdx);
    if (seqQArray)
        free(seqQArray);
    if (seqKArray)
        free(seqKArray);
    if (devSeqQArray)
        cuda_check(cudaFree(devSeqQArray));
    if (devSeqKArray)
        cuda_check(cudaFree(devSeqKArray));
}

bool AuxiliaryArray::is_initialized(
        const size_t _batchSize, const size_t _seqLenQ, const size_t _seqLenK,
        bool _attnMask) {
    if (_batchSize != batchSize or _seqLenQ != seqLenQ or _seqLenK != seqLenK or
        _attnMask != attnMask or !seqQArray or !seqKArray or !devSeqQArray or
        !devSeqKArray or !loWinIdx or !hiWinIdx)
        return false;
    return true;
}

void AuxiliaryArray::set(
        const size_t _batchSize, const size_t _seqLenQ, const size_t _seqLenK,
        bool _attnMask) {
    if (_batchSize == batchSize && _seqLenQ == seqLenQ && _seqLenK == seqLenK &&
        _attnMask == attnMask)
        return;
    else {
        if (loWinIdx)
            free(loWinIdx);
        if (hiWinIdx)
            free(hiWinIdx);
        if (seqQArray)
            free(seqQArray);
        if (seqKArray)
            free(seqKArray);
        if (devSeqQArray)
            cuda_check(cudaFree(devSeqQArray));
        if (devSeqKArray)
            cuda_check(cudaFree(devSeqKArray));
    };

    seqLenQ = _seqLenQ;
    seqLenK = _seqLenK;
    batchSize = _batchSize;
    attnMask = _attnMask;
    size_t seqQArraySize = 1 * batchSize;
    size_t seqKArraySize = batchSize;
    seqQArray = (int*)calloc(seqQArraySize, sizeof(int));
    seqKArray = (int*)calloc(seqKArraySize, sizeof(int));
    for (size_t i = 0; i < seqQArraySize; ++i)
        seqQArray[i] = seqLenQ;
    for (size_t i = 0; i < seqKArraySize; ++i)
        seqKArray[i] = seqLenK;

    cuda_check(cudaMalloc((void**)&devSeqQArray, seqQArraySize * sizeof(int)));
    cuda_check(cudaMalloc((void**)&devSeqKArray, seqKArraySize * sizeof(int)));

    cuda_check(cudaMemcpy(
            devSeqQArray, seqQArray, seqQArraySize * sizeof(int),
            cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(
            devSeqKArray, seqKArray, seqKArraySize * sizeof(int),
            cudaMemcpyHostToDevice));

    loWinIdx = (int*)calloc(seqLenQ, sizeof(int));
    hiWinIdx = (int*)calloc(seqLenQ, sizeof(int));
    for (size_t i = 0; i < seqLenQ; ++i) {
        loWinIdx[i] = 0;
        if (attnMask)
            hiWinIdx[i] = i + 1;
        else
            hiWinIdx[i] = seqLenK;
    }
}

void MultiHeadAttnStatus::set(
        cudnnHandle_t handle, const Param& p, const TensorLayout& q,
        const TensorLayout& k, const TensorLayout& v) {
    float attn_prob = p.training ? p.attn_prob : 0.f;
    float out_prob = p.training ? p.out_prob : 0.f;
    if (!attn_dropout_status.initialized())
        attn_dropout_status.set(handle, p.seed, attn_prob);
    if (!out_dropout_status.initialized())
        out_dropout_status.set(handle, p.seed, out_prob);

    if (attn_dropout_status.drop_prob != attn_prob) {
        attn_dropout_status.drop_prob = attn_prob;
        attn_dropout_status.restore_desc(handle);
    }
    if (out_dropout_status.drop_prob != out_prob) {
        out_dropout_status.drop_prob = out_prob;
        out_dropout_status.restore_desc(handle);
    }
    batchSize = q.shape[0];
    seqLenQ = q.shape[1];
    seqLenK = k.shape[1];
    qSize = q.shape[2];
    kSize = k.shape[2];
    vSize = v.shape[2];
    numHeads = p.num_heads;
    qProjSize = p.qproj_size ? qSize / numHeads : 0;
    kProjSize = p.kproj_size ? kSize / numHeads : 0;
    vProjSize = p.vproj_size ? vSize / numHeads : 0;
    oProjSize = p.oproj_size ? qSize : 0;
    attnMask = p.attn_mask_type >= param::MultiHeadAttn::ATTN_MASK_TYPE::DEFAULT_MASK;
    cudnnDataType_t cudnn_dtype = to_cudnn_dtype(q.dtype);
    auto flag = CUDNN_ATTN_QUERYMAP_ONE_TO_ONE;
    if (p.qbias or p.kbias or p.vbias or p.obias)
        flag = flag | CUDNN_ATTN_ENABLE_PROJ_BIASES;
#if CUDNN_VERSION < 8600
    // TODO: CUDNN_VERSION < 8600 and out dropout > 0.0, we need to go to the proxy cuda
    // implementation.
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

    auxArray.set(
            batchSize, seqLenQ, seqLenK,
            p.attn_mask_type >= param::MultiHeadAttn::ATTN_MASK_TYPE::DEFAULT_MASK);

    if (p.training)
        cudnnGetMultiHeadAttnBuffers(
                handle, attn_desc, &sizeWeights, &sizeWkspace, &sizeReserve);
    else {
        cudnnGetMultiHeadAttnBuffers(
                handle, attn_desc, &sizeWeights, &sizeWkspace, NULL);
        sizeReserve = 0;
    }
}

bool MultiHeadAttnStatus::is_initialized(
        const Param& p, const TensorLayout& q, const TensorLayout& k,
        const TensorLayout& v) {
    float attn_prob = p.training ? p.attn_prob : 0.f;
    float out_prob = p.training ? p.out_prob : 0.f;
    if (!attn_dropout_status.initialized() or !out_dropout_status.initialized() or
        attn_dropout_status.drop_prob != attn_prob or
        out_dropout_status.drop_prob != out_prob)
        return false;
    if (q.shape[0] != batchSize or q.shape[1] != seqLenQ or k.shape[1] != seqLenK or
        q.shape[2] != qSize or k.shape[2] != kSize or v.shape[2] != vSize or
        attnMask != (p.attn_mask_type >=
                     param::MultiHeadAttn::ATTN_MASK_TYPE::DEFAULT_MASK) or
        numHeads != p.num_heads) {
        return false;
    }
    if ((p.qproj_size && (qProjSize == 0 or qProjSize != qSize / p.num_heads)) or
        (p.kproj_size && (kProjSize == 0 or kProjSize != kSize / p.num_heads)) or
        (p.vproj_size && (vProjSize == 0 or vProjSize != vSize / p.num_heads)) or
        (p.oproj_size && (oProjSize == 0 or oProjSize != q.shape[2])))
        return false;
    if ((!p.qproj_size && qProjSize != 0) or (!p.kproj_size && kProjSize != 0) or
        (!p.vproj_size && vProjSize != 0) or (!p.oproj_size && oProjSize != 0))
        return false;
    if (!auxArray.is_initialized(batchSize, seqLenQ, seqLenK, attnMask))
        return false;
    if (p.training and sizeReserve == 0)
        return false;
    return true;
}

}  // namespace cuda
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
