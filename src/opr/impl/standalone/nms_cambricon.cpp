#include "nms_cambricon.h"
#include <memory>

#if MGB_CAMBRICON

namespace {

template <typename T, cnnlStatus_t (*dtor)(T*)>
struct CnnlDescriptorDeleter {
    void operator()(T* ptr) {
        if (ptr != nullptr) {
            MGB_CNNL_CHECK(dtor(ptr));
        }
    }
};

template <typename T, cnnlStatus_t (*ctor)(T**), cnnlStatus_t (*dtor)(T*)>
class CnnlDescriptor {
public:
    CnnlDescriptor() = default;
    T* desc() const { return desc_.get(); }
    T* desc() { return desc_.get(); }
    T* mut_desc() {
        init();
        return desc_.get();
    }

protected:
    void init() {
        if (desc_ == nullptr) {
            T* ptr;
            MGB_CNNL_CHECK(ctor(&ptr));
            desc_.reset(ptr);
        }
    }

private:
    std::unique_ptr<T, CnnlDescriptorDeleter<T, dtor>> desc_;
};

class CnnlTensorDescriptor : public CnnlDescriptor<
                                     cnnlTensorStruct, &cnnlCreateTensorDescriptor,
                                     &cnnlDestroyTensorDescriptor> {
public:
    CnnlTensorDescriptor() = default;
    void set(
            int ndim, std::initializer_list<size_t> shape, cnnlDataType_t data_type,
            cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
        set(ndim, std::vector<size_t>{shape}, data_type, layout);
    }
    void set(
            int ndim, std::vector<size_t> shape, cnnlDataType_t data_type,
            cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY) {
        if (!ndim) {
            ndim = 1;
            std::vector<int> shape_info(1, 1);
            MGB_CNNL_CHECK(cnnlSetTensorDescriptorEx(
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
        MGB_CNNL_CHECK(cnnlSetTensorDescriptorEx(
                this->mut_desc(), layout, data_type, ndim, shape_info.data(),
                stride_info.data()));
    };
};

class CnnlNmsDescriptor
        : public CnnlDescriptor<
                  cnnlNmsStruct, &cnnlCreateNmsDescriptor, &cnnlDestroyNmsDescriptor> {
public:
    CnnlNmsDescriptor() {}
    void set(
            const cnnlNmsOutputMode_t mode, const float iou_threshold,
            const int max_output_size, const float confidence_threshold,
            const int input_layout) {
        MGB_CNNL_CHECK(cnnlSetNmsDescriptor_v2(
                this->mut_desc(), mode, iou_threshold, max_output_size,
                confidence_threshold, input_layout));
    }
};
}  // namespace

size_t mgb::opr::standalone::nms::cambricon_kern_workspace(
        cnnlHandle_t handle, size_t nr_boxes) {
    CnnlTensorDescriptor dets_desc, confidence_desc;
    dets_desc.set(2, {nr_boxes, 4}, CNNL_DTYPE_FLOAT);
    confidence_desc.set(1, {nr_boxes}, CNNL_DTYPE_FLOAT);
    size_t space_size = 0;
    MGB_CNNL_CHECK(cnnlGetNmsWorkspaceSize_v3(
            handle, dets_desc.desc(), confidence_desc.desc(), &space_size));
    return space_size + nr_boxes * sizeof(float);
};

// #if MGB_CAMBRICON
void mgb::opr::standalone::nms::cambricon_kern(
        size_t nr_boxes, size_t max_output, float overlap_thresh, const void* boxes,
        void* out_idx, void* out_size, void* workspace, cnnlHandle_t handle) {
    CnnlNmsDescriptor nms_desc;
    nms_desc.set(CNNL_NMS_OUTPUT_TARGET_INDICES, overlap_thresh, max_output, 0.f, 0);

    CnnlTensorDescriptor dets_desc, confidence_desc, output_desc;
    dets_desc.set(2, {nr_boxes, 4}, CNNL_DTYPE_FLOAT);
    confidence_desc.set(1, {nr_boxes}, CNNL_DTYPE_FLOAT);
    output_desc.set(1, {max_output}, CNNL_DTYPE_UINT32);

    size_t space_size = 0;
    MGB_CNNL_CHECK(cnnlGetNmsWorkspaceSize_v3(
            handle, dets_desc.desc(), confidence_desc.desc(), &space_size));
    float value = 1.0f;
    void* confidence = static_cast<char*>(workspace) + space_size;
    MGB_CNNL_CHECK(cnnlFill_v3(
            handle, CNNL_POINTER_MODE_HOST, &value, confidence_desc.desc(),
            confidence));
    MGB_CNNL_CHECK(cnnlNms_v2(
            handle, nms_desc.desc(), dets_desc.desc(), boxes, confidence_desc.desc(),
            confidence, workspace, space_size, output_desc.desc(), out_idx, out_size));
};

#endif