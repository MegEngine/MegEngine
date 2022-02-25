#pragma once

#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {
namespace cuda {

class ElemwiseForwardImpl final : public ElemwiseForwardImplHelper {
#include "src/common/elemwise/opr_impl_class_def.inl"
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
