#pragma once
#include "megdnn/handle.h"
#include "megdnn/oprs.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"
#include "src/common/multi_head_attn/proxy_backward_base.h"
#include "src/common/reduce_helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

using Param = megdnn::MultiHeadAttn::Param;
using MaskType = Param::AttnMaskType;
using InputType = Param::TensorCombinationType;
using multi_head_attn::matmul_deduce_layout;
using multi_head_attn::matmul_exec;

class MHABackwardProxyOpr final : public multi_head_attn::MHABackwardProxyBase {
public:
    MHABackwardProxyOpr() : MHABackwardProxyBase() {}

#define cb(DType)               \
    void move_scaler_to_device( \
            Handle*, DTypeTrait<DType>::ctype*, DTypeTrait<DType>::ctype*) override;
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
};
}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
