#pragma once
#include "megdnn/oprs.h"
#include "src/naive/add_update/opr_impl.h"

namespace megdnn {
namespace fallback {

class AddUpdateImpl : public naive::AddUpdateForwardImpl {
public:
    using naive::AddUpdateForwardImpl::AddUpdateForwardImpl;
    void exec(_megdnn_tensor_inout dest, _megdnn_tensor_in delta) override;

    bool is_thread_safe() const override { return true; }
};

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
