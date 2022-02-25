#pragma once

#include "megdnn/oprs.h"
#include "src/common/utils.cuh"

namespace megdnn {
namespace naive {

class MaskConvForwardImpl : public MaskConvForward {
private:
    std::unique_ptr<Convolution> m_conv_opr;

public:
    MaskConvForwardImpl(Handle* handle);
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in mask,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& mask, const TensorLayout& dst) override;
};

class MaskPropagateImpl : public MaskPropagate {
public:
    MaskPropagateImpl(Handle* handle) : MaskPropagate(handle) {}

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace worksapce) override final;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&) override final {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn
