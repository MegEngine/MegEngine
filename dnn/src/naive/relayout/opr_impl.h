#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class RelayoutForwardImpl : public RelayoutForward {
protected:
    //! check that src_handle is on CPU
    void check_cpu_handle(Handle* src_handle);

    void do_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst);

public:
    using RelayoutForward::RelayoutForward;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
