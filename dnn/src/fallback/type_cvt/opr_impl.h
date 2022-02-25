#pragma once
#include "src/naive/type_cvt/opr_impl.h"

namespace megdnn {
namespace fallback {

class TypeCvtImpl : public naive::TypeCvtImpl {
    bool exec_optimized(_megdnn_tensor_in src, _megdnn_tensor_out dst);

public:
    using naive::TypeCvtImpl::TypeCvtImpl;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
    bool is_thread_safe() const override { return true; }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
