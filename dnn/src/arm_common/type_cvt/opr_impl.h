#pragma once

#include "src/fallback/type_cvt/opr_impl.h"

namespace megdnn {
namespace arm_common {

class TypeCvtImpl : public fallback::TypeCvtImpl {
public:
    using fallback::TypeCvtImpl::TypeCvtImpl;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
