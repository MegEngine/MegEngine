#pragma once

#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/common/opr_trait.h"

namespace megdnn {
namespace test {

void run_powc_test(Handle* handle, DType dtype, bool test_non_continuity = true);

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
