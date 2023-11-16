#pragma once

#include "src/cambricon/cnnl_wrapper/cnnl_common_descriptors.h"
#include "src/cambricon/cnnl_wrapper/cnnl_types.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlTensorDescriptors.h

namespace megdnn {
namespace cambricon {
class CnnlTensorDescriptor : public CnnlDescriptor<
                                     cnnlTensorStruct, &cnnlCreateTensorDescriptor,
                                     &cnnlDestroyTensorDescriptor> {
public:
    // Init create Tensor descriptor
    CnnlTensorDescriptor() = default;
    CnnlTensorDescriptor(const TensorND* t) { set(t); }  // NOLINT
    CnnlTensorDescriptor(
            const TensorND* t, cnnlTensorLayout_t layout,
            cnnlDataType_t data_type = CNNL_DTYPE_INVALID) {
        set(t, layout, data_type);
    }
    CnnlTensorDescriptor(const TensorND* t, cnnlDataType_t dtype) { set(t, dtype); }
    // set descriptor from tensor
    void set(const TensorND* t);
    void set(const TensorND* t, cnnlDataType_t dtype);
    void set(
            const TensorND* t, cnnlTensorLayout_t layout,
            cnnlDataType_t dtype = CNNL_DTYPE_INVALID);

    void set(const TensorLayout& t);
    void set(const TensorLayout& t, cnnlDataType_t dtype);
    void set(
            const TensorLayout& t, cnnlTensorLayout_t layout,
            cnnlDataType_t dtype = CNNL_DTYPE_INVALID);

    template <typename T>
    void set(
            int ndim, const std::vector<T>& shape, cnnlDataType_t data_type,
            cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
        set(ndim, shape.data(), data_type, layout);
    }

    template <typename T>
    void set(
            int ndim, T shape[], cnnlDataType_t data_type,
            cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
        if (!ndim) {
            ndim = 1;
            std::vector<int> shape_info(1, 1);
            cnnl_check(cnnlSetTensorDescriptorEx(
                    this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, ndim,
                    shape_info.data(), shape_info.data()));
            return;
        }
        std::vector<int> shape_info(ndim, 1);
        std::vector<int> stride_info(ndim, 1);
        int value = 1;
        for (size_t i = ndim - 1; i > 0; --i) {
            shape_info[i] = static_cast<int>(shape[i]);
            stride_info[i] = value;
            value *= shape_info[i];
        }
        shape_info[0] = static_cast<int>(shape[0]);
        stride_info[0] = value;
        cnnl_check(cnnlSetTensorDescriptorEx(
                this->mut_desc(), layout, data_type, ndim, shape_info.data(),
                stride_info.data()));
    }

    template <typename T>
    void set(
            int ndim, const std::vector<T>& shape, const std::vector<T>& stride,
            cnnlDataType_t data_type, cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
        if (!ndim) {
            ndim = 1;
            std::vector<int> shape_info(1, 1);
            cnnl_check(cnnlSetTensorDescriptorEx(
                    this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, ndim,
                    shape_info.data(), shape_info.data()));
            return;
        }
        std::vector<int> shape_info(ndim, 1);
        std::vector<int> stride_info(ndim, 1);
        for (size_t i = ndim - 1; i > 0; --i) {
            shape_info[i] = static_cast<int>(shape[i]);
            stride_info[i] = static_cast<int>(stride[i]);
        }
        shape_info[0] = static_cast<int>(shape[0]);
        stride_info[0] = static_cast<int>(stride[0]);
        cnnl_check(cnnlSetTensorDescriptorEx(
                this->mut_desc(), layout, data_type, ndim, shape_info.data(),
                stride_info.data()));
    }
};
}  // namespace cambricon
}  // namespace megdnn
