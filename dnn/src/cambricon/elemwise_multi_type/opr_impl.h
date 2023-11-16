#pragma once

#include <cnnl.h>
#include <cnrt.h>
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn_int.h"

namespace megdnn {
namespace cambricon {

class ElemwiseMultiTypeImpl final : public ElemwiseMultiType {
public:
    using ElemwiseMultiType::ElemwiseMultiType;
    void exec(
            _megdnn_in const TensorNDArray& src, _megdnn_tensor_out dst) override final;
    size_t get_workspace_in_bytes(const TensorNDArray& src, const TensorND& dst);
    void dest_type_bool_mode(
            const TensorNDArray& src, const TensorND& dst_tensor,
            megdnn::param::ElemwiseMultiType::Mode mode, Workspace* workspace);
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
