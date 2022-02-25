#pragma once

#include "megdnn/oprs.h"
#include "src/common/add_update_helper.h"
#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

class AddUpdateForwardImpl final : public AddUpdateForwardHelper {
    void exec_noncontig(_megdnn_tensor_inout dest, _megdnn_tensor_in delta);

public:
    using AddUpdateForwardHelper::AddUpdateForwardHelper;

    void exec(_megdnn_tensor_inout dest, _megdnn_tensor_in delta) override;

    bool is_thread_safe() const override { return true; }
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
