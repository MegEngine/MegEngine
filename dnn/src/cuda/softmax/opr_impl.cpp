#include "src/cuda/softmax/opr_impl.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

int CanonicalAxis(const int axis, const int rank) {
    if (axis < 0) {
        return axis + rank;
    }
    return axis;
}

int SizeToAxis(const int axis, const size_t* dims) {
    int size = 1;
    for (int i = 0; i < axis; i++) {
        size *= dims[i];
    }
    return size;
}

int SizeOutAxis(const int axis, const size_t* dims, const int ndim) {
    int size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        size *= dims[i];
    }
    return size;
}

std::vector<int> SoftmaxForwardImpl::init_mode(
        _megdnn_tensor_in src, cudnnSoftmaxMode_t& mode) const {
    auto dims = src.layout.shape;
    const int rank = src.layout.ndim;
    const int axis = CanonicalAxis(param().axis, rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims, rank);

    mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL;

    return {N, dim, D, 1};
}

int sc(const size_t x) {
    return static_cast<int>(x);
}

cudnnDataType_t to_cudnn_dtype(
        DType type, const param::Convolution::Format format = {}) {
    switch (type.enumv()) {
        case DTypeEnum::Float32:
            return CUDNN_DATA_FLOAT;
        case DTypeEnum::Float16:
            return CUDNN_DATA_HALF;
#if CUDNN_MAJOR >= 7
        case DTypeEnum::Int32:
        case DTypeEnum::QuantizedS32:
            return CUDNN_DATA_INT32;
#endif
#if CUDNN_MAJOR >= 6
        case DTypeEnum::QuantizedS8: {
            if (format == param::Convolution::Format::NCHW4)
                return CUDNN_DATA_INT8x4;
#if CUDNN_VERSION >= 7500
            else if (format == param::Convolution::Format::NCHW32)
                return CUDNN_DATA_INT8x32;
#endif
            else
                return CUDNN_DATA_INT8;
        }

        case DTypeEnum::Int8: {
            if (format == param::Convolution::Format::NCHW4)
                return CUDNN_DATA_INT8x4;
#if CUDNN_VERSION >= 7500
            else if (format == param::Convolution::Format::NCHW32)
                return CUDNN_DATA_INT8x32;
#endif
            else
                return CUDNN_DATA_INT8;
        }
#endif
        default:
#if CUDNN_MAJOR >= 6
            megdnn_throw("dtype must be float16/float32/int8/int32");
#else
            megdnn_throw("dtype must be float16/float32");
#endif
    }
}

void SoftmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    dt_float32 alpha = 1.0f, beta = 0.0f;
    TensorDesc src_desc, dst_desc;

    cudnnSoftmaxMode_t mode;
    std::vector<int> tensor_dims = init_mode(src, mode);
    const int dimA[] = {
            sc(tensor_dims[0]), sc(tensor_dims[1]), sc(tensor_dims[2]),
            sc(tensor_dims[3])};
    const int strideA[] = {
            sc(tensor_dims[1] * tensor_dims[2] * tensor_dims[3]),
            sc(tensor_dims[2] * tensor_dims[3]), sc(tensor_dims[3]), 1};

    cudnn_check(cudnnSetTensorNdDescriptor(
            src_desc.desc, to_cudnn_dtype(src.layout.dtype), 4, dimA, strideA));
    cudnn_check(cudnnSetTensorNdDescriptor(
            dst_desc.desc, to_cudnn_dtype(dst.layout.dtype), 4, dimA, strideA));

    cudnn_check(cudnnSoftmaxForward(
            cudnn_handle(this->handle()), CUDNN_SOFTMAX_ACCURATE, mode, &alpha,
            src_desc.desc, src.raw_ptr(), &beta, dst_desc.desc, dst.raw_ptr()));
}

//================================Softmax Backward============================

std::vector<int> SoftmaxBackwardImpl::init_mode(
        _megdnn_tensor_in src, cudnnSoftmaxMode_t& mode) const {
    auto dims = src.layout.shape;
    const int rank = src.layout.ndim;
    const int axis = CanonicalAxis(param().axis, rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims, rank);

    mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL;

    return {N, dim, D, 1};
}

void SoftmaxBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    {
        dt_float32 alpha = 1.0f, beta = 0.0f;
        TensorDesc src_desc, diff_desc, grad_desc;
        cudnnSoftmaxMode_t mode;
        std::vector<int> tensor_dims = init_mode(src, mode);

        const int dimA[] = {
                sc(tensor_dims[0]), sc(tensor_dims[1]), sc(tensor_dims[2]),
                sc(tensor_dims[3])};
        const int strideA[] = {
                sc(tensor_dims[1] * tensor_dims[2] * tensor_dims[3]),
                sc(tensor_dims[2] * tensor_dims[3]), sc(tensor_dims[3]), 1};

        cudnn_check(cudnnSetTensorNdDescriptor(
                src_desc.desc, to_cudnn_dtype(src.layout.dtype), 4, dimA, strideA));
        cudnn_check(cudnnSetTensorNdDescriptor(
                diff_desc.desc, to_cudnn_dtype(diff.layout.dtype), 4, dimA, strideA));
        cudnn_check(cudnnSetTensorNdDescriptor(
                grad_desc.desc, to_cudnn_dtype(grad.layout.dtype), 4, dimA, strideA));

        cudnn_check(cudnnSoftmaxBackward(
                cudnn_handle(this->handle()), CUDNN_SOFTMAX_ACCURATE, mode, &alpha,
                src_desc.desc, src.raw_ptr(), diff_desc.desc, diff.raw_ptr(), &beta,
                grad_desc.desc, grad.raw_ptr()));
    }
}

// vim: syntax=cpp.doxygen