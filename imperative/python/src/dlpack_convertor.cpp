#include "./dlpack_convertor.h"
#include <limits>
#include "./helper.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/tensor.h"

using namespace mgb::imperative;
using namespace mgb;

DLDataType mgb::imperative::get_dl_datatype(const DeviceTensorND& dv) {
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = dv.dtype().size() * 8;
    switch (dv.dtype().enumv()) {
        case DTypeEnum::Byte:
            dtype.code = DLDataTypeCode::kDLUInt;
            break;
        case DTypeEnum::Uint8:
            dtype.code = DLDataTypeCode::kDLUInt;
            break;
        case DTypeEnum::Uint16:
            dtype.code = DLDataTypeCode::kDLUInt;
            break;
        case DTypeEnum::Int32:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        case DTypeEnum::Int16:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        case DTypeEnum::Int8:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        case DTypeEnum::Float32:
            dtype.code = DLDataTypeCode::kDLFloat;
            break;
        case DTypeEnum::Float16:
            dtype.code = DLDataTypeCode::kDLFloat;
            break;
        case DTypeEnum::Bool:
            mgb_throw(MegBrainError, "Bool type is not supported by dlpack");
            break;
        case DTypeEnum::BFloat16:
            dtype.code = DLDataTypeCode::kDLBfloat;
            break;
        case DTypeEnum::Complex64:
            dtype.code = DLDataTypeCode::kDLComplex;
            break;
        default:
            mgb_throw(MegBrainError, "type is not supported by dlpack");
    }
    return dtype;
}

DLDevice mgb::imperative::get_dl_device(const DeviceTensorND& dv) {
    auto cn = dv.comp_node();
    DLDevice ctx;
    switch (cn.device_type()) {
        case CompNode::DeviceType::CPU: {
            ctx.device_id = 0;
            ctx.device_type = DLDeviceType::kDLCPU;
            break;
        }
        case CompNode::DeviceType::CUDA: {
#if MGB_CUDA
            auto&& env = CompNodeEnv::from_comp_node(cn).cuda_env();
            ctx.device_id = env.device;
            ctx.device_type = DLDeviceType::kDLCUDA;
#else
            mgb_throw(MegBrainError, "CUDA device is not available");
#endif
            break;
        }
        default:
            mgb_throw(
                    MegBrainError, "Cannot pack tensors on %s", cn.to_string().c_str());
    }
    return ctx;
}

CompNode as_comp_node(const std::string& name) {
    thread_local struct {
        std::string name;
        CompNode cn;
    } dlpack_cncached;
    if (dlpack_cncached.name != name) {
        dlpack_cncached.name = name;
        dlpack_cncached.cn = CompNode::load(name);
    }
    return dlpack_cncached.cn;
}

CompNode mgb::imperative::get_tensor_device(const DLDevice& ctx, int stream) {
    int id = ctx.device_id;
    switch (ctx.device_type) {
        case DLDeviceType::kDLCPU: {
            auto device = "cpu" + std::to_string(id);
            return as_comp_node(device);
        }
        case DLDeviceType::kDLCUDA: {
            auto device = "gpu" + std::to_string(id) + ":" + std::to_string(stream);
            return as_comp_node(device);
        }
        default:
            mgb_throw(MegBrainError, "Unsupported device_type");
    }
}

DType mgb::imperative::get_tensor_type(const DLDataType& dtype) {
    DType tensortype;
    switch (dtype.code) {
        case DLDataTypeCode::kDLUInt:
            switch (dtype.bits) {
                case 8:
                    tensortype = DType::from_enum(DTypeEnum::Uint8);
                    break;
                case 16:
                    tensortype = DType::from_enum(DTypeEnum::Uint16);
                    break;
                default:
                    mgb_throw(
                            MegBrainError, "Unsupported kUInt bits: %s",
                            std::to_string(dtype.bits).c_str());
            }
            break;

        case DLDataTypeCode::kDLInt:
            switch (dtype.bits) {
                case 8:
                    tensortype = DType::from_enum(DTypeEnum::Int8);
                    break;
                case 16:
                    tensortype = DType::from_enum(DTypeEnum::Int16);
                    break;
                case 32:
                    tensortype = DType::from_enum(DTypeEnum::Int32);
                    break;
                default:
                    mgb_throw(
                            MegBrainError, "Unsupported kInt bits: %s",
                            std::to_string(dtype.bits).c_str());
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (dtype.bits) {
                case 16:
                    tensortype = DType::from_enum(DTypeEnum::Float16);
                    break;
                case 32:
                    tensortype = DType::from_enum(DTypeEnum::Float32);
                    break;
                default:
                    mgb_throw(
                            MegBrainError, "Unsupported kFloat bits: %s",
                            std::to_string(dtype.bits).c_str());
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (dtype.bits) {
                case 16:
                    tensortype = DType::from_enum(DTypeEnum::BFloat16);
                    break;
                default:
                    mgb_throw(
                            MegBrainError, "Unsupported kBFloat bits: %s",
                            std::to_string(dtype.bits).c_str());
            }
            break;
        case DLDataTypeCode::kDLComplex:
            switch (dtype.bits) {
                case 64:
                    tensortype = DType::from_enum(DTypeEnum::Complex64);
                default:
                    mgb_throw(
                            MegBrainError, "Unsupported Complex bits: %s",
                            std::to_string(dtype.bits).c_str());
            }
            break;
    }
    return tensortype;
}

struct DLMTensor {
    DeviceTensorND value;
    DLManagedTensor tensor;
    int64_t shape[MEGDNN_MAX_NDIM];
    int64_t stride[MEGDNN_MAX_NDIM];
};

void deleter(DLManagedTensor* arg) {
    delete static_cast<DLMTensor*>(arg->manager_ctx);
}

DLManagedTensor* mgb::imperative::to_dlpack(const ValueRef src) {
    DeviceTensorND dv = src.dev_tensor()->as_nd(true);
    DLMTensor* TensorHandler(new DLMTensor);
    size_t ndim = dv.shape().ndim;
    TensorHandler->value = dv;
    TensorHandler->tensor.manager_ctx = TensorHandler;
    TensorHandler->tensor.deleter = &deleter;
    TensorHandler->tensor.dl_tensor.data = TensorHandler->value.raw_ptr();
    TensorHandler->tensor.dl_tensor.device = get_dl_device(dv);
    TensorHandler->tensor.dl_tensor.ndim = ndim;
    TensorHandler->tensor.dl_tensor.dtype = get_dl_datatype(dv);

    auto src_shape = TensorHandler->value.layout().shape;
    auto src_stride = TensorHandler->value.layout().stride;
    for (size_t i = 0; i < ndim; i++) {
        if (src_shape[i] > std::numeric_limits<int64_t>::max()) {
            mgb_throw(
                    MegBrainError, "unsupported input shape: %s",
                    TensorHandler->value.layout().to_string().c_str());
        }
        TensorHandler->shape[i] = static_cast<int64_t>(src_shape[i]);
        TensorHandler->stride[i] = static_cast<int64_t>(src_stride[i]);
    }
    TensorHandler->tensor.dl_tensor.shape = TensorHandler->shape;
    TensorHandler->tensor.dl_tensor.strides = TensorHandler->stride;
    TensorHandler->tensor.dl_tensor.byte_offset = 0;
    return &(TensorHandler->tensor);
}

TensorShape ptr2shape(const int64_t* ptr, size_t ndim) {
    TensorShape shape;
    mgb_assert(
            ndim <= TensorShape::MAX_NDIM, "dim too large: %zd (max %zd)", ndim,
            TensorShape::MAX_NDIM);
    shape.ndim = ndim;
    for (size_t i = 0; i < ndim; i++) {
        if (ptr[i] < 0 || ptr[i] > std::numeric_limits<size_t>::max()) {
            std::string error_msg = "";
            for (size_t idx = 0; idx < ndim; idx++) {
                auto shape_i = " " + std::to_string(ptr[i]);
                error_msg += shape_i;
            }
            mgb_throw(
                    MegBrainError, "unsupported dlpack input shape: %s",
                    error_msg.c_str());
        }
        shape[i] = ptr[i];
    }
    return shape;
}

ValueRef mgb::imperative::from_dlpack(DLManagedTensor* dlMTensor, int stream = 0) {
    std::function<void(void*)> deleter_dispatch = [dlMTensor](void*) {
        if (dlMTensor->deleter) {
            dlMTensor->deleter(dlMTensor);
        }
    };

    DType tensor_type = get_tensor_type(dlMTensor->dl_tensor.dtype);
    CompNode tensor_device = get_tensor_device(dlMTensor->dl_tensor.device, stream);
    DeviceTensorStorage storage;
    size_t dtype_size = tensor_type.size();
    size_t ndim = dlMTensor->dl_tensor.ndim;
    TensorShape tensor_shape = ptr2shape(dlMTensor->dl_tensor.shape, ndim);

    storage.reset(
            tensor_device, tensor_shape.total_nr_elems() * dtype_size,
            {static_cast<dt_byte*>(dlMTensor->dl_tensor.data), deleter_dispatch});

    ValueShape shapevalue = ValueShape::from(tensor_shape);
    ValueRef val = imperative::apply(
            CreateTensor(
                    CreateTensor::Common, tensor_device, tensor_type, shapevalue, {}),
            DeviceStorage::make(storage))[0];
    return val;
};