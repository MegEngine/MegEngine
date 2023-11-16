#pragma once

#include "megdnn/oprs.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

class FillImpl final : public Fill {
public:
    using Fill::Fill;

    void exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&) { return 0; }

private:
    template <typename ctype>
    void exec_internal(ctype* dst, size_t size, CnnlTensorDescriptor& out_des);
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
