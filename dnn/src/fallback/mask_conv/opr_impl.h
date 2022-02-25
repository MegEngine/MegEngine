#pragma once

#include "megdnn/oprs.h"
#include "src/fallback/handle.h"

namespace megdnn {
namespace fallback {

class MaskConvForwardImpl : public MaskConvForward {
    std::unique_ptr<MatrixMul> m_matmul_opr;
    WorkspaceBundle get_wbundle(
            const size_t OC, const size_t OH, const size_t OW, const size_t IC,
            const size_t IH, const size_t IW, const size_t FH, const size_t FW,
            const size_t PH, const size_t PW);

public:
    MaskConvForwardImpl(Handle* handle);

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in mask,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& mask, const TensorLayout& dst) override;
};

}  // namespace fallback
}  // namespace megdnn
