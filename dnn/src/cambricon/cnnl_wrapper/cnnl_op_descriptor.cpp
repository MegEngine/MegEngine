#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/utils.mlu.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

void CnnlConvolutionDescriptor::set(
        int stride_h, int stride_w, int pad_h, int pad_w, int dilate_h, int dilate_w,
        int groups, cnnlDataType_t dtype) {
    int dim = 4;
    int spatial_size = 2;
    std::vector<int> padding_t(2 * spatial_size);
    std::vector<int> stride_t(spatial_size);
    std::vector<int> dilation_t(spatial_size);
    int groups_t = groups;
    padding_t[0] = pad_h;
    padding_t[1] = pad_h;
    padding_t[2] = pad_w;
    padding_t[3] = pad_w;
    stride_t[0] = stride_h;
    stride_t[1] = stride_w;
    dilation_t[0] = dilate_h;
    dilation_t[1] = dilate_w;
    cnnl_check(cnnlSetConvolutionDescriptor(
            this->mut_desc(), dim, padding_t.data(), stride_t.data(), dilation_t.data(),
            groups_t, dtype));
}

void CnnlActivationDescriptor::set(
        cnnlActivationMode_t mode, cnnlActivationPreference_t prefer,
        cnnlNanPropagation_t nanProp, float ceof, int sliced_dim, float gamma,
        float scale, bool is_result, bool approximate) {
    cnnl_check(cnnlSetActivationDescriptor_v6(
            this->mut_desc(), mode, prefer, nanProp, ceof, sliced_dim, gamma, scale,
            is_result, approximate));
}

void CnnlMatmulDescriptor::set_attr(
        cnnlMatMulDescAttribute_t attr, const void* buf, size_t size_in_bytes) {
    cnnl_check(cnnlSetMatMulDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
}

void CnnlPoolingDescriptor::set2D(
        cnnlPoolingMode_t mode, cnnlNanPropagation_t nan_propagation, int window_height,
        int window_width, int top_padding, int bottom_padding, int left_padding,
        int right_padding, int vertical_stride, int horizon_stride) {
    cnnl_check(cnnlSetPooling2dDescriptor(
            this->mut_desc(), mode, nan_propagation, window_height, window_width,
            top_padding, bottom_padding, left_padding, right_padding, vertical_stride,
            horizon_stride));
}

void CnnlReduceDescriptor::set(
        int axis[], int axis_num, cnnlReduceOp_t reduce_op, cnnlDataType_t tensor_type,
        cnnlNanPropagation_t nan_propagation, cnnlReduceIndices_t tensor_indices,
        cnnlIndicesType_t indices_type) {
    cnnl_check(cnnlSetReduceDescriptor(
            this->mut_desc(), axis, axis_num, reduce_op, tensor_type, nan_propagation,
            tensor_indices, indices_type));
}
void CnnlTransposeDescriptor::set(int dims, int permute[]) {
    cnnl_check(cnnlSetTransposeDescriptor(this->mut_desc(), dims, permute));
}
void CnnlTransposeDescriptor::set(int dims, size_t permute[]) {
    std::vector<int> _permute(dims);
    for (int i = 0; i < dims; ++i) {
        _permute[i] = static_cast<int>(permute[i]);
    }
    cnnl_check(cnnlSetTransposeDescriptor(this->mut_desc(), dims, _permute.data()));
}

void CnnlOpTensorDescriptor::set(
        cnnlOpTensorDesc_t mode, cnnlDataType_t data_type,
        cnnlNanPropagation_t nan_propagation) {
    cnnl_check(cnnlSetOpTensorDescriptor(
            this->mut_desc(), mode, data_type, nan_propagation));
}

}  // namespace cambricon
}  // namespace megdnn