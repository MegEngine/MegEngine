#include "megdnn/oprs.h"
#include "src/common/utils.h"

using namespace megdnn;

size_t CondTake::check_exec_get_size(
        const TensorLayout& data, const TensorLayout& mask, size_t workspace_in_bytes) {
    megdnn_assert(
            data.eq_shape(mask), "CondTake shape differs: data=%s mask=%s",
            data.TensorShape::to_string().c_str(),
            mask.TensorShape::to_string().c_str());
    megdnn_assert(data.is_physical_contiguous() && mask.is_physical_contiguous());
    megdnn_assert(m_param.eps > 0, "eps must be non-negative; got: %g", m_param.eps);
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(data));
    return data.total_nr_elems();
}

CondTake::OutputDType CondTake::infer_dtype(DType data, DType /*mask*/) {
    return {{data, dtype::Int32()}};
}

// vim: syntax=cpp.doxygen
