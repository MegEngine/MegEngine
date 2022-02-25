#pragma once
#include "src/fallback/type_cvt/opr_impl.h"
#include "src/x86/handle.h"

namespace megdnn {
namespace x86 {

class TypeCvtImpl : public fallback::TypeCvtImpl {
public:
    using fallback::TypeCvtImpl::TypeCvtImpl;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
    bool is_thread_safe() const override { return true; }
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
