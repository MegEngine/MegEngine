#pragma once
#include "src/cuda/cudnn_wrapper.h"
#if CUDNN_VERSION >= 8004
#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"
#include "src/common/algo_chooser.h"
#include "src/common/utils.h"
#include "src/cuda/dropout/opr_impl.h"
#include "src/cuda/handle.h"

namespace megdnn {
namespace cuda {

struct AuxiliaryArray {
public:
    int* seqQArray = nullptr;
    int* seqKArray = nullptr;
    int* devSeqQArray = nullptr;
    int* devSeqKArray = nullptr;
    int* loWinIdx = nullptr;
    int* hiWinIdx = nullptr;
    size_t seqLenQ = 0;
    size_t seqLenK = 0;
    size_t batchSize = 0;
    bool attnMask = 0;
    ~AuxiliaryArray();
    void set(
            const size_t _batchSize, const size_t _seqLenQ, const size_t _seqLenK,
            bool _attnMask);
    bool is_initialized(
            const size_t _batchSize, const size_t _seqLenQ, const size_t _seqLenK,
            bool _attnMask);
};

using Param = megdnn::MultiHeadAttn::Param;

class MultiHeadAttnStatus {
    DropoutStatus attn_dropout_status;
    DropoutStatus out_dropout_status;

    cudnnAttnDescriptor_t attn_desc;

    AuxiliaryArray auxArray;

    size_t numHeads = 0;
    size_t batchSize = 0;
    size_t seqLenQ = 0;
    size_t seqLenK = 0;
    size_t qSize = 0;
    size_t kSize = 0;
    size_t vSize = 0;
    size_t qProjSize = 0;
    size_t kProjSize = 0;
    size_t vProjSize = 0;
    size_t oProjSize = 0;
    bool attnMask = 0;

    size_t sizeWeights = 0;
    size_t sizeWkspace = 0;
    size_t sizeReserve = 0;

public:
    MultiHeadAttnStatus() { cudnn_check(cudnnCreateAttnDescriptor(&attn_desc)); }
    ~MultiHeadAttnStatus() { cudnn_check(cudnnDestroyAttnDescriptor(attn_desc)); }

private:
    void set(
            cudnnHandle_t handle, const Param& p, const TensorLayout& q,
            const TensorLayout& k, const TensorLayout& v);
    bool is_initialized(
            const Param& p, const TensorLayout& q, const TensorLayout& k,
            const TensorLayout& v);
    friend class MultiHeadAttnBase;
    friend class MultiHeadAttnForwardImpl;
    friend class MultiHeadAttnBackwardImpl;
};
}  // namespace cuda
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
