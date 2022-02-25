#pragma once

#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {
namespace rocm {

class ElemwiseForwardImpl final : public ElemwiseForwardImplHelper {
#include "src/common/elemwise/opr_impl_class_def.inl"
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
