#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace atlas {

class TopKImpl : public TopK {
protected:
    void do_exec(
            int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t* indices,
            _megdnn_workspace workspace) override;
    void do_find_kth_value(
            int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t* indices,
            _megdnn_workspace workspace);

public:
    using TopK::TopK;

    size_t get_workspace_in_bytes(
            int, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
