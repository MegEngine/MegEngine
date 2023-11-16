#pragma once

#include <memory>
#include <vector>
#include "cnnl.h"
#include "src/cambricon/utils.mlu.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlCommonDescriptors.h

namespace megdnn {
namespace cambricon {

template <typename T, cnnlStatus_t (*dtor)(T*)>
struct CnnlDescriptorDeleter {
    void operator()(T* ptr) {
        if (ptr != nullptr) {
            cnnl_check(dtor(ptr));
        }
    }
};

template <typename T, cnnlStatus_t (*ctor)(T**), cnnlStatus_t (*dtor)(T*)>
class CnnlDescriptor {
public:
    CnnlDescriptor() = default;

    // Use desc() to access the underlying descriptor pointer in
    // a read-only fashion.  Most client code should use this.
    // If the descriptor was never initialized, this will return
    // nullptr.
    T* desc() const { return desc_.get(); }
    T* desc() { return desc_.get(); }

    // Use CnnlDescriptor() to access the underlying desciptor pointer
    // if you intend to modify what it points to This will ensure
    // that the descriptor is initialized.
    // Code in this file will use this function.
    T* mut_desc() {
        init();
        return desc_.get();
    }

protected:
    void init() {
        if (desc_ == nullptr) {
            T* ptr;
            cnnl_check(ctor(&ptr));
            desc_.reset(ptr);
        }
    }

private:
    std::unique_ptr<T, CnnlDescriptorDeleter<T, dtor>> desc_;
};

// void convertShapeAndStride(std::vector<int>& shape_info, std::vector<int>&
// stride_info);

}  // namespace cambricon
}  // namespace megdnn
