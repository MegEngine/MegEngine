#pragma once
#include "megdnn/oprs.h"
#include "src/common/add_update_helper.h"

namespace megdnn {
namespace naive {

class AddUpdateForwardImpl : public AddUpdateForwardHelper {
public:
    using AddUpdateForwardHelper::AddUpdateForwardHelper;
    void exec(_megdnn_tensor_inout dest, _megdnn_tensor_in delta) override;

    bool is_thread_safe() const override { return true; }
};

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
