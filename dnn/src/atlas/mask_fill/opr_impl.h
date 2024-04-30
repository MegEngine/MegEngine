#pragma once
#include "megdnn/oprs.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"

namespace megdnn {
namespace atlas {
class MaskedFillImpl : public MaskedFill {
public:
    using MaskedFill::MaskedFill;
    void exec(_megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dst)
            override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};
}  // namespace atlas
}  // namespace megdnn
