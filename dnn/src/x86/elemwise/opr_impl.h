#pragma once
#include "src/fallback/elemwise/opr_impl.h"

namespace megdnn {
namespace x86 {

class ElemwiseImpl final : public fallback::ElemwiseImpl {
    bool exec_unary();
    bool exec_binary();
    bool exec_ternary_fma3();

public:
    using fallback::ElemwiseImpl::ElemwiseImpl;
    void exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) override;
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
