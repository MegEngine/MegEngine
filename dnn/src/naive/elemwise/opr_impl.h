#pragma once

#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {
namespace naive {

class ElemwiseForwardImpl : public ElemwiseForwardImplHelper {
#include "src/common/elemwise/opr_impl_class_def.inl"
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
