#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace rocm {

class TypeCvtImpl final : public TypeCvt {
public:
    using TypeCvt::TypeCvt;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
