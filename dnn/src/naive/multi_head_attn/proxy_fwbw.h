#pragma once
#include <memory>
#include "megdnn/oprs.h"
#include "megdnn/oprs/cv.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "src/common/multi_head_attn/proxy_backward_base.h"
#include "src/common/multi_head_attn/proxy_forward_base.h"

namespace megdnn {
namespace naive {

using Param = megdnn::MultiHeadAttn::Param;
using MaskType = Param::AttnMaskType;
using InputType = Param::TensorCombinationType;
using multi_head_attn::matmul_deduce_layout;
using multi_head_attn::matmul_exec;

class MHAForwardProxyOpr final : public multi_head_attn::MHAForwardProxyBase {
public:
#define cb(DType)               \
    void move_scaler_to_device( \
            Handle*, DTypeTrait<DType>::ctype*, DTypeTrait<DType>::ctype*) override;
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
};

class MHABackwardProxyOpr final : public multi_head_attn::MHABackwardProxyBase {
public:
#define cb(DType)               \
    void move_scaler_to_device( \
            Handle*, DTypeTrait<DType>::ctype*, DTypeTrait<DType>::ctype*) override;
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
