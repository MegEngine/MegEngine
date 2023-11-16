#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cambricon {
class MaskedFillImpl : public MaskedFill {
public:
    using MaskedFill::MaskedFill;
    void exec(_megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dst)
            override;
    void exec(
            _megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& origin, const TensorLayout& index,
            const TensorLayout& dest) override;
};
}  // namespace cambricon
}  // namespace megdnn
