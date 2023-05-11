#pragma once
#include <memory>
#include "megdnn/oprs.h"
#include "megdnn/oprs/cv.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "src/common/multi_head_attn/proxy_forward_base.h"
#include "src/naive/multi_head_attn/proxy_fwbw.h"

namespace megdnn {
namespace naive {

class MultiHeadAttnForwardImpl final : public MultiHeadAttnForward {
public:
    using MultiHeadAttnForward::MultiHeadAttnForward;
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

    MHABackwardProxyOpr proxy_opr;

    void exec(MHA_BACKWARD_EXEC_PARAM) override;
    size_t get_workspace_in_bytes(MHA_BACKWARD_LAYOUT_CONST_PARAM) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
