#pragma once
#include "megdnn/oprs/linalg.h"

namespace megdnn {
namespace cuda {

class MatrixInverseImpl : public MatrixInverse {
public:
    using MatrixInverse::MatrixInverse;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override { m_error_tracker = tracker; }

protected:
    void* m_error_tracker = nullptr;
    size_t get_workspace_in_bytes(size_t batch, size_t n, size_t dtype_size) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
