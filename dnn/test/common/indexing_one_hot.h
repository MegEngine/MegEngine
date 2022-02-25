#pragma once

#include "megdnn/handle.h"

namespace megdnn {
namespace test {

void run_indexing_one_hot_test(
        Handle* handle, const thin_function<void()>& fail_test = {});
void run_indexing_set_one_hot_test(Handle* handle);

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
