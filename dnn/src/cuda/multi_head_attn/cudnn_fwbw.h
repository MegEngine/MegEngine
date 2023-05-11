#pragma once
#include <vector>
#include "megdnn/handle.h"
#include "megdnn/thin/small_vector.h"
#include "src/cuda/cudnn_wrapper.h"
#if CUDNN_VERSION >= 8004
#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"
#include "src/common/algo_chooser.h"
#include "src/common/multi_head_attn/helper.h"
#include "src/common/utils.h"
#include "src/cuda/dropout/opr_impl.h"
#include "src/cuda/handle.h"

using Param = megdnn::MultiHeadAttn::Param;
using MaskType = Param::AttnMaskType;
using InputType = Param::TensorCombinationType;

namespace megdnn {
namespace cuda {

struct AuxiliaryArray {
public:
    SmallVector<int> seqQArray;
    SmallVector<int> seqKArray;
    int* devSeqQArray = nullptr;
    int* devSeqKArray = nullptr;
    SmallVector<int> loWinIdx;
    SmallVector<int> hiWinIdx;
    size_t seqLenQ = 0;
    size_t seqLenK = 0;
    size_t batchSize = 0;
    MaskType attnMaskType = MaskType::NO_MASK;
    ~AuxiliaryArray();
    void set(
            Handle* handle, const size_t _batchSize, const size_t _seqLenQ,
            const size_t _seqLenK, MaskType _attnMaskType);
    void set_cudnn_style_mask(Handle* handle, const TensorND& attn_mask);
    bool is_initialized(
            const size_t _batchSize, const size_t _seqLenQ, const size_t _seqLenK,
            MaskType _attnMaskType);
};

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
    MaskType attnMaskType = MaskType::NO_MASK;
    bool bias = false;

    size_t sizeWeights = 0;
    size_t sizeWkspace = 0;
    size_t sizeReserve = 0;

public:
    MultiHeadAttnStatus() { cudnn_check(cudnnCreateAttnDescriptor(&attn_desc)); }
    ~MultiHeadAttnStatus() { cudnn_check(cudnnDestroyAttnDescriptor(attn_desc)); }

private:
    void set(
            Handle* handle, const Param& p, const TensorLayout& q,
            const TensorLayout& k, const TensorLayout& v);
    void set_cudnn_style_mask(Handle* handle, const TensorND& attn_mask);
    bool is_initialized(
            const Param& p, const TensorLayout& q, const TensorLayout& k,
            const TensorLayout& v);
    friend class MultiHeadAttnBase;
    friend class MultiHeadAttnForwardImpl;
    friend class MultiHeadAttnBackwardImpl;
    friend class MHAForwardCudnnOpr;
    friend class MHABackwardCudnnOpr;
};

class MHAForwardCudnnOpr {
public:
    MHAForwardCudnnOpr(){};

    void exec(MHA_PROXY_FORWARD_EXEC_PARAM);
    void deduce_layout(MHA_PROXY_FORWARD_LAYOUT_PARAM);
    size_t get_workspace_in_bytes(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);
    size_t get_mask_reservespace_in_bytes(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);
    size_t get_othr_reservespace_in_bytes(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM);

private:
    MultiHeadAttnStatus desc_status;
};

class MHABackwardCudnnOpr {
public:
    MHABackwardCudnnOpr(){};

    void exec(MHA_PROXY_BACKWARD_EXEC_PARAM);

private:
    MultiHeadAttnStatus desc_status;
};

}  // namespace cuda
}  // namespace megdnn
#endif
   // vim: syntax=cpp.doxygen
