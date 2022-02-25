#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace cuda {

class TopKImpl : public TopK {
protected:
    template <typename ctype>
    void dispatch_with_ctype(
            int k, size_t m, size_t n, ptrdiff_t lda, const ctype* data, ctype* values,
            int* indices, void* workspace);

    void do_exec(
            int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t* indices,
            _megdnn_workspace workspace) override;

public:
    using TopK::TopK;

    size_t get_workspace_in_bytes(
            int k, const TensorLayout& data, const TensorLayout& values,
            const TensorLayout& indices) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
