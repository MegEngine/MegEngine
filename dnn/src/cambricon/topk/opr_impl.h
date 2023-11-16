#pragma once
#include "megdnn/oprs.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

struct TopKCnnlDescs {
    CnnlTensorDescriptor data_desc, out_value_desc, out_indices_desc;
    int sort_dim = 0;
    bool sorted = false;
    bool lower_index_first = true;
    using Mode = param::TopK::Mode;
    TopKCnnlDescs(
            const TensorLayout& data, const TensorLayout& values, const Mode mode);
};

class TopKImpl : public TopK {
protected:
    template <typename ctype>
    void dispatch_with_ctype(
            int k, const ctype* data, ctype* values, int* indices, void* workspace,
            size_t workspace_size, TopKCnnlDescs& descs);

    void do_exec(
            int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t* indices,
            _megdnn_workspace workspace) override;
    WorkspaceBundle make_bundle(
            int k, const TensorLayout& data, const TensorLayout& values,
            const TensorLayout& indices);

public:
    using TopK::TopK;

    size_t get_workspace_in_bytes(
            int k, const TensorLayout& data, const TensorLayout& values,
            const TensorLayout& indices) override;
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
