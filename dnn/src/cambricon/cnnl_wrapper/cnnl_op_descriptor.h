#pragma once

#include "src/cambricon/cnnl_wrapper/cnnl_common_descriptors.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlOpDescriptors.h

namespace megdnn {
namespace cambricon {

class CnnlConvolutionDescriptor
        : public CnnlDescriptor<
                  cnnlConvolutionStruct, &cnnlCreateConvolutionDescriptor,
                  &cnnlDestroyConvolutionDescriptor> {
public:
    CnnlConvolutionDescriptor() {}

    void set(
            int stride_h, int stride_w, int pad_h, int pad_w, int dilate_h,
            int dilate_w, int groups, cnnlDataType_t dtype);
};

class CnnlActivationDescriptor
        : public CnnlDescriptor<
                  cnnlActivationStruct, &cnnlCreateActivationDescriptor,
                  &cnnlDestroyActivationDescriptor> {
public:
    CnnlActivationDescriptor() {}

    void set(
            cnnlActivationMode_t mode, cnnlActivationPreference_t prefer,
            cnnlNanPropagation_t nanProp, float ceof, int sliced_dim = 0,
            float gamma = 0.f, float scale = 0.f, bool is_result = false,
            bool approximate = true);
};

class CnnlMatmulDescriptor
        : public CnnlDescriptor<
                  cnnlMatMulStruct, &cnnlMatMulDescCreate, &cnnlMatMulDescDestroy> {
public:
    CnnlMatmulDescriptor() {}
    void set_attr(
            cnnlMatMulDescAttribute_t attr, const void* buf, size_t size_in_bytes);
};
class CnnlPoolingDescriptor : public CnnlDescriptor<
                                      cnnlPoolingStruct, &cnnlCreatePoolingDescriptor,
                                      &cnnlDestroyPoolingDescriptor> {
public:
    CnnlPoolingDescriptor() {}

    void set2D(
            cnnlPoolingMode_t mode, cnnlNanPropagation_t nan_propagation,
            int window_height, int window_width, int top_padding, int bottom_padding,
            int left_padding, int right_padding, int vertical_stride,
            int horizon_stride);
};

class CnnlReduceDescriptor : public CnnlDescriptor<
                                     cnnlReduceStruct, &cnnlCreateReduceDescriptor,
                                     &cnnlDestroyReduceDescriptor> {
public:
    CnnlReduceDescriptor() {}

    void set(
            int axis[], int axis_num, cnnlReduceOp_t reduce_op,
            cnnlDataType_t tensor_type, cnnlNanPropagation_t nan_propagation,
            cnnlReduceIndices_t tensor_indices, cnnlIndicesType_t indices_type);
};
class CnnlTransposeDescriptor
        : public CnnlDescriptor<
                  cnnlTransposeStruct, &cnnlCreateTransposeDescriptor,
                  &cnnlDestroyTransposeDescriptor> {
public:
    CnnlTransposeDescriptor() {}

    void set(int dims, int permute[]);
    void set(int dims, size_t permute[]);
};

class CnnlOpTensorDescriptor
        : public CnnlDescriptor<
                  cnnlOpTensorStruct, &cnnlCreateOpTensorDescriptor,
                  &cnnlDestroyOpTensorDescriptor> {
public:
    CnnlOpTensorDescriptor() {}

    void set(
            cnnlOpTensorDesc_t mode, cnnlDataType_t data_type,
            cnnlNanPropagation_t nan_propagation);
};

}  // namespace cambricon
}  // namespace megdnn