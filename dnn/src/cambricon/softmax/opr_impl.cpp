#include "src/cambricon/softmax/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"

using namespace megdnn;
using namespace cambricon;

namespace {

int CanonicalAxis(const int axis, const int rank) {
    if (axis < 0) {
        return axis + rank;
    }
    return axis;
}

size_t SizeToAxis(const int axis, const size_t* dims) {
    size_t size = 1;
    for (int i = 0; i < axis; i++) {
        size *= dims[i];
    }
    return size;
}

size_t SizeOutAxis(const int axis, const size_t* dims, const int ndim) {
    size_t size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        size *= dims[i];
    }
    return size;
}

SmallVector<size_t> init_shape(_megdnn_tensor_in src, const int param_axis) {
    auto dims = src.layout.shape;
    const int rank = src.layout.ndim;
    const int axis = CanonicalAxis(param_axis, rank);
    const size_t dim = dims[axis];
    const size_t N = SizeToAxis(axis, dims);
    const size_t D = SizeOutAxis(axis, dims, rank);

    return {N, dim, D};
}
}  // anonymous namespace

void SoftmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto dtype_dest = dst.layout.dtype.enumv();
    megdnn_assert(
            check_dtype_float_ieee(dtype_dest),
            "Cambricon unsupport SoftmaxForward with dtype:%d",
            static_cast<int>(dtype_dest));
    auto cnnl_handler = cnnl_handle(this->handle());
    TensorLayout reshape_layout{
            TensorShape{init_shape(src, param().axis)}, dst.layout.dtype};
    reshape_layout.init_contiguous_stride();
    CnnlTensorDescriptor src_desc, dst_desc;
    src_desc.set(reshape_layout);
    dst_desc.set(reshape_layout);

    cnnl_check(cnnlSoftmaxForward_v2(
            cnnl_handler, cnnlSoftmaxAlgorithm_t::CNNL_SOFTMAX_ACCURATE,
            cnnlSoftmaxMode_t::CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION,
            cnnlComputationPreference_t::CNNL_COMPUTATION_HIGH_PRECISION, nullptr,
            src_desc.desc(), src.raw_ptr(), nullptr, dst_desc.desc(), dst.raw_ptr()));
}

//================================Softmax Backward============================

void SoftmaxBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    auto dtype_dest = grad.layout.dtype.enumv();
    megdnn_assert(
            check_dtype_float_ieee(dtype_dest),
            "Cambricon unsupport SoftmaxBackward with dtype:%d",
            static_cast<int>(dtype_dest));
    auto cnnl_handler = cnnl_handle(this->handle());
    TensorLayout reshape_layout{
            TensorShape{init_shape(src, param().axis)}, grad.layout.dtype};
    reshape_layout.init_contiguous_stride();
    CnnlTensorDescriptor src_desc, diff_desc, grad_desc;
    src_desc.set(reshape_layout);
    diff_desc.set(reshape_layout);
    grad_desc.set(reshape_layout);

    cnnl_check(cnnlSoftmaxBackward(
            cnnl_handler, cnnlSoftmaxAlgorithm_t::CNNL_SOFTMAX_ACCURATE,
            cnnlSoftmaxMode_t::CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION, nullptr,
            src_desc.desc(), src.raw_ptr(), diff_desc.desc(), diff.raw_ptr(), nullptr,
            grad_desc.desc(), grad.raw_ptr()));
}

// vim: syntax=cpp.doxygen