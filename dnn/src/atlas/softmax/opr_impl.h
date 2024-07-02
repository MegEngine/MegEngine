#pragma once

#include "megdnn/oprs.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"

namespace megdnn {

namespace atlas {

class SoftmaxForwardImpl final : public SoftmaxForward {
public:
    using SoftmaxForward::SoftmaxForward;

    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, /* src */
            const TensorLayout& /* dst */) override {
        return 0;
    }
};

}  // namespace atlas

}  // namespace megdnn