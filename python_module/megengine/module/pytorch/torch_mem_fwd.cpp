/**
 * \file python_module/megengine/module/pytorch/torch_mem_fwd.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "torch/extension.h"
#include "megbrain_pubapi.h"

using MGBTensor = mgb::pubapi::DeviceTensor;

torch::Tensor mgb_to_torch(const MGBTensor *src) {

    mgb::pubapi::CallbackOnce deleter;
    void* tensor_raw_ptr;
    src->forward_to(&tensor_raw_ptr, &deleter);
    auto deleter_wrap = [deleter](void*) mutable {
        deleter.consume();
    };

    // TODO: support non-contiguous layout
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < src->desc.ndim; ++ i) {
        sizes.push_back(src->desc.shape[i]);
    }

    torch::TensorOptions options;
    switch (src->desc.dtype) {
#define map_dtype(mgb_dtype, torch_dtype) \
    case MGBTensor::DataType::mgb_dtype: \
        options = options.dtype(caffe2::TypeMeta::Make<torch_dtype>()); \
        break;
        map_dtype(FLOAT32, float);
        map_dtype(FLOAT16, torch::Half);
        map_dtype(INT32, int);
        map_dtype(INT16, int16_t);
        map_dtype(INT8, int8_t);
        map_dtype(UINT8, uint8_t);
#undef map_dtype
        default:
            throw std::runtime_error("bad case for data type.");
    }

    // TODO: Maybe we should impl copy on different devices?
    switch (src->desc.type) {
        case MGBTensor::Type::CUDA: {
            int device_id = src->desc.cuda_ctx.device;
            if (device_id >= 0) {
                options = options.device(torch::DeviceType::CUDA, device_id);
            } else {
                throw std::runtime_error("bad case for device(cuda) id.");
            }
            // TODO: consider cuda synchronization here
            // Maybe all tasks issued on cuda_ctx(device, stream) should be done?
            break;
        }
        case MGBTensor::Type::CPU:
            options = options.device(torch::DeviceType::CPU);
            // Torch's API are all synchronous.
            src->sync();
            break;
        default:
            throw std::runtime_error("bad case for device type.");
    }

    auto tensor = torch::from_blob(tensor_raw_ptr, sizes, deleter_wrap, options);
    return tensor;
}

void torch_to_mgb(MGBTensor* dst, torch::Tensor src) {
    MGBTensor::Desc desc;

    desc.dev_ptr = src.data_ptr();

    // src is contiguous torch tensor here, so no strides needed
    std::vector<size_t> shape;
    // desc.shape is the pointer to a size array used to construct
    // an inner-mgb tensor, which should be valid until calling of
    // forward_other_memory return
    for (auto &&i : src.sizes()) {
        shape.push_back(i);
    }
    desc.shape = shape.data();
    desc.ndim = shape.size();

    switch (src.scalar_type()) {
#define map_dtype(mgb_dtype, torch_dtype) \
    case torch::ScalarType::torch_dtype: \
        desc.dtype = MGBTensor::DataType::mgb_dtype; \
        break;
        map_dtype(FLOAT32, Float);
        map_dtype(FLOAT16, Half);
        map_dtype(INT32, Int);
        map_dtype(INT16, Short);
        map_dtype(INT8, Char);
        map_dtype(UINT8, Byte);
#undef map_dtype
        default:
            throw std::runtime_error("bad case for data type.");
    }

    // TODO: cuda setting and synchronization like mgb_to_torch
    if (src.device().type() == torch::DeviceType::CUDA) {
        desc.type = MGBTensor::Type::CUDA;
        desc.cuda_ctx.device = src.get_device();
        desc.cuda_ctx.stream = nullptr;
    } else {
        assert(src.device().type() == torch::DeviceType::CPU);
        desc.type = MGBTensor::Type::CUDA;
    }

    mgb::pubapi::CallbackOnce deleter;
    deleter.user_data = new torch::Tensor(src);
    deleter.fptr = [](void* ptr) {
        delete static_cast<torch::Tensor*>(ptr);
    };
    dst->forward_other_memory(desc, deleter);
}

torch::Tensor inp_mem_fwd(uintptr_t dv_ptr) {
    // construct torch Tensor from mgb DeviceTensor stored in dv_ptr.
    return mgb_to_torch(reinterpret_cast<MGBTensor*>(dv_ptr));
}

void oup_mem_fwd(uintptr_t dv_ptr, torch::Tensor src,
                 bool keep_data_ptr=false) {
    // forward storage in torch Tensor to mgb DeviceTensor
    // keep_data_ptr: set to True to ensure forwarding data_ptr under \p src
    // to megbrain, or it maybe copy src to a new contiguous tensor storage.

    // which would return src itself if tensor is contiguous
    auto src_contig = src.contiguous();

    if (keep_data_ptr && src_contig.data_ptr() != src.data_ptr()) {
        throw std::runtime_error("should keep tensor data ptr, but it changed");
    }
    torch_to_mgb(reinterpret_cast<MGBTensor*>(dv_ptr), src_contig);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inp_mem_fwd", &inp_mem_fwd, "Forward mgb DeviceTensor ptr into torch Tensor as network input.");
    m.def("oup_mem_fwd", &oup_mem_fwd, "Forward torch network Tensor to corresponding mgb VarNode.",
        py::arg("dv_ptr"), py::arg("src"), py::arg("keep_data_ptr") = false);
}
