#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.mlu.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlTensorDescriptors.cpp

namespace megdnn {
namespace cambricon {

void CnnlTensorDescriptor::set(const TensorND* t) {
    set(t->layout);
}

void CnnlTensorDescriptor::set(const TensorND* t, cnnlDataType_t data_type) {
    set(t->layout, data_type);
}

void CnnlTensorDescriptor::set(
        const TensorND* t, cnnlTensorLayout_t layout, cnnlDataType_t data_type) {
    set(t->layout, layout, data_type);
}

void CnnlTensorDescriptor::set(const TensorLayout& t) {
    auto data_type = convert_to_cnnl_datatype(t.dtype.enumv());
    set(t, data_type);
}

void CnnlTensorDescriptor::set(const TensorLayout& t, cnnlDataType_t data_type) {
    int t_dim = t.ndim;
    if (!t_dim) {
        t_dim = 1;
        std::vector<int> dim_array(1, 1);
        cnnl_check(cnnlSetTensorDescriptorEx(
                this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim, dim_array.data(),
                dim_array.data()));
        return;
    }
    std::vector<int> shape_info(t_dim);
    std::vector<int> stride_info(t_dim);
    for (size_t i = 0; i < t_dim; ++i) {
        shape_info[i] = static_cast<int>(t.shape[i]);
        stride_info[i] = static_cast<int>(t.stride[i]);
    }
    cnnl_check(cnnlSetTensorDescriptorEx(
            this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim, shape_info.data(),
            stride_info.data()));
}

void CnnlTensorDescriptor::set(
        const TensorLayout& t, cnnlTensorLayout_t layout, cnnlDataType_t data_type) {
    int t_dim = t.ndim;
    if (data_type == CNNL_DTYPE_INVALID) {
        data_type = convert_to_cnnl_datatype(t.dtype.enumv());
    }
    if (!t_dim) {
        t_dim = 1;
        std::vector<int> dim_array(1, 1);
        cnnl_check(cnnlSetTensorDescriptorEx(
                this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim, dim_array.data(),
                dim_array.data()));
        return;
    }
    std::vector<int> shape_info(t_dim);
    std::vector<int> stride_info(t_dim);
    for (size_t i = 0; i < t_dim; ++i) {
        shape_info[i] = static_cast<int>(t.shape[i]);
        stride_info[i] = static_cast<int>(t.stride[i]);
    }
    cnnl_check(cnnlSetTensorDescriptorEx(
            this->mut_desc(), layout, data_type, t_dim, shape_info.data(),
            stride_info.data()));
}

}  // namespace cambricon
}  // namespace megdnn
