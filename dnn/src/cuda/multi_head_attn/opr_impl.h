#pragma once
#include "megdnn/handle.h"
#include "megdnn/oprs.h"
#include "src/common/reduce_helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/handle.h"
#include "src/cuda/multi_head_attn/cudnn_fwbw.h"
#include "src/cuda/multi_head_attn/proxy_bw.h"
#include "src/cuda/multi_head_attn/proxy_fw.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

using Param = megdnn::MultiHeadAttn::Param;
using MaskType = Param::AttnMaskType;
using InputType = Param::TensorCombinationType;

bool can_use_mha_cudnn(const Param& param);

class MultiHeadAttnForwardImpl final : public MultiHeadAttnForward {
public:
    using MultiHeadAttnForward::MultiHeadAttnForward;
#if CUDNN_VERSION >= 8004
    MHAForwardCudnnOpr cudnn_opr;
#endif
    MHAForwardProxyOpr proxy_opr;

    void exec(MHA_FORWARD_EXEC_PARAM) override;
    void deduce_layout(MHA_FORWARD_LAYOUT_PARAM) override;
    size_t get_workspace_in_bytes(MHA_FORWARD_LAYOUT_CONST_PARAM) override;
    size_t get_mask_reservespace_in_bytes(MHA_FORWARD_LAYOUT_CONST_PARAM) override;
    size_t get_othr_reservespace_in_bytes(MHA_FORWARD_LAYOUT_CONST_PARAM) override;
};

class MultiHeadAttnBackwardImpl final : public MultiHeadAttnBackward {
public:
    using MultiHeadAttnBackward::MultiHeadAttnBackward;
#if CUDNN_VERSION >= 8004
    MHABackwardCudnnOpr cudnn_opr;
#endif
    MHABackwardProxyOpr proxy_opr;

    void exec(MHA_BACKWARD_EXEC_PARAM) override;
    size_t get_workspace_in_bytes(MHA_BACKWARD_LAYOUT_CONST_PARAM) override;
};
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
