#pragma once
#include "megdnn/oprs.h"
namespace megdnn {
namespace naive {

class ParamPackConcatImpl final : public ParamPackConcat {
public:
    using ParamPackConcat::ParamPackConcat;
    void exec(
            _megdnn_tensor_in srcs, _megdnn_tensor_in offsets, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorShape&, const TensorShape&, const TensorShape&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn
