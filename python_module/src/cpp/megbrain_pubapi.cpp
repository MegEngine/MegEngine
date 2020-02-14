/**
 * \file python_module/src/cpp/megbrain_pubapi.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./megbrain_pubapi.h"
#include "./megbrain_pubapi_internal.h"

#include "megbrain/tensor.h"
#include "megbrain/graph/var_node.h"
#include "megbrain/comp_node_env.h"

namespace {

class DeleteDispatcher final : public mgb::CompNodeDepedentObject {
    mgb::thin_function<void()> m_deleter;
    mgb::CompNode m_comp_node;
    std::atomic<bool> done;

    std::shared_ptr<void> on_comp_node_finalize() override {
        bool _ = false;
        if (done.compare_exchange_strong(_, true)) {
            m_deleter();
        }
        return {};
    }
public:
    explicit DeleteDispatcher(mgb::thin_function<void()>&& deleter,
                              mgb::CompNode cn)
            : m_deleter(std::move(deleter)), m_comp_node(cn) {
        done.store(false);
    }

    void trigger() {
        bool _ = false;
        if (done.compare_exchange_strong(_, true)) {
            if (!is_finalized()) {
                m_comp_node.add_callback(std::move(m_deleter));
            } else {
                m_deleter();
            }
        }
    }
};

}  // namespace

using namespace mgb;

pubapi::DeviceTensor::DataType mgb::dtype_mgb2pubapi(DType dtype) {
    using DevDType = pubapi::DeviceTensor::DataType;
    switch (dtype.enumv()) {
#define o(s, t)        \
    case DTypeEnum::s: \
        return DevDType::t
        o(Float32, FLOAT32);
        o(Float16, FLOAT16);
        o(Int32, INT32);
        o(Int16, INT16);
        o(Int8, INT8);
        o(Uint8, UINT8);
#undef o
        default:
            mgb_throw(MegBrainError, "dtype %s not implemented for pubapi",
                      dtype.name());
    }
}

struct pubapi::DeviceTensor::_Impl {

static TensorShape desc_shape_to_tensor_shape(const DeviceTensor::Desc &desc) {
    TensorShape shape;
    mgb_assert(desc.ndim && desc.ndim <= TensorShape::MAX_NDIM,
            "invalid ndim: %zu", desc.ndim);
    shape.ndim = desc.ndim;
    for (size_t i = 0; i < desc.ndim; ++ i) {
        shape[i] = desc.shape[i];
    }
    return shape;
}

#if MGB_CUDA
class CudaCurrentDeviceRestore {
    int m_orig_dev = -1;

    public:

        CudaCurrentDeviceRestore(CompNode cn) {
            if (cn.device_type() == CompNode::DeviceType::CUDA) {
                MGB_CUDA_CHECK(cudaGetDevice(&m_orig_dev));
            }
        }

        ~CudaCurrentDeviceRestore() {
            if (m_orig_dev != -1) {
                cudaSetDevice(m_orig_dev);
            }
        }
};
#else
class CudaCurrentDeviceRestore {
    public:
        CudaCurrentDeviceRestore(CompNode) {
        }
};
#endif

static void sync(const DeviceTensor *self, bool strong) {
    CompNode cn;
    if (self->m_dev_nd) {
        cn = static_cast<DeviceTensorND*>(self->m_dev_nd)->comp_node();
    } else {
        mgb_assert(self->m_varptr);
        cn = static_cast<cg::VarNode*>(self->m_varptr)->comp_node();
    }
    CudaCurrentDeviceRestore cuda_dev_restore{cn};
    cn.sync();
#if MGB_CUDA
    if (strong && cn.device_type() == CompNode::DeviceType::CUDA) {
        cn.activate();
        MGB_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

static const char* dtype_name(DataType dtype) {
    switch (dtype) {
#define on(c)         \
    case DataType::c: \
        return #c
        on(FLOAT32);
        on(FLOAT16);
        on(INT32);
        on(INT16);
        on(INT8);
        on(UINT8);
#undef on
        default:
            mgb_throw(MegBrainError, "invalid pubapi dtype enum: %d",
                      static_cast<int>(dtype));
    }
}

static void copy(
        DeviceTensor *self, const Desc &other, CopyDirection direction) {
    mgb_assert(self->desc.dtype == other.dtype, "dtype mismatch: %s vs %s",
               self->dtype_name(), dtype_name(other.dtype));
    mgb_assert(self->m_varptr || self->m_dev_nd);
    const DeviceTensorND *dv;
    if (direction == CopyDirection::OTHER_TO_SELF) {
        mgb_assert(!self->m_readonly, "can not copy into readonly tensor");
        auto shape = desc_shape_to_tensor_shape(other);
        if (self->m_varptr) {
            auto var = static_cast<cg::VarNode*>(self->m_varptr);
            dv = &var->shape_alloc(shape).dev_tensor();
        } else {
            dv = static_cast<DeviceTensorND*>(self->m_dev_nd);
            mgb_assert(dv->shape().eq_shape(shape),
                    "copy dest tensor shape is %s, but source shape is %s",
                    dv->shape().to_string().c_str(), shape.to_string().c_str());
        }
        mgb_assert(self->desc.dtype == dtype_mgb2pubapi(dv->dtype()));
        self->desc.dev_ptr = dv->raw_ptr();
        self->desc.ndim = dv->shape().ndim;
        self->desc.shape = dv->shape().shape;
        if (!other.dev_ptr) {
            // used in resize()
            return;
        }
    } else {
        mgb_assert(direction == CopyDirection::SELF_TO_OTHER);
        if (self->m_varptr) {
            dv = &static_cast<cg::VarNode*>(self->m_varptr)->dev_tensor();
        } else {
            dv = static_cast<DeviceTensorND*>(self->m_dev_nd);
        }
    }

    mgb_assert(dv->layout().is_contiguous());
    auto size = dv->layout().span().dist_byte();
    auto cn = dv->comp_node();
    CudaCurrentDeviceRestore cuda_dev_restore{cn};

    void *dst = dv->raw_ptr(), *src = other.dev_ptr;
    if (direction == CopyDirection::SELF_TO_OTHER) {
        std::swap(dst, src);
    }

#if !MGB_CUDA
    mgb_assert(other.type != Type::CUDA, "cuda disabled at compile time");
#endif

    auto &&desc = self->desc;
    if (other.type == desc.type) {
#if MGB_CUDA
        if (desc.type == Type::CUDA) {
            int dev = desc.cuda_ctx.device;
            if (dev == -1) {
                MGB_CUDA_CHECK(cudaGetDevice(&dev));
            }
            mgb_assert(dev == other.cuda_ctx.device,
                    "DeviceTensor copy must be on the same device; "
                    "got %d vs %d", dev, other.cuda_ctx.device);
        }
#endif
        cn.peer_copy_to(cn, dst, src, size);
    } else {
        if ((desc.type == Type::CPU && other.type == Type::CUDA &&
                    direction == CopyDirection::SELF_TO_OTHER) ||
                (other.type == Type::CPU && desc.type == Type::CUDA &&
                 direction == CopyDirection::OTHER_TO_SELF)) {
            cn.copy_to_device(dst, src, size);
        } else {
            mgb_assert((desc.type == Type::CUDA && other.type == Type::CPU &&
                        direction == CopyDirection::SELF_TO_OTHER) ||
                    (other.type == Type::CUDA && desc.type == Type::CPU &&
                     direction == CopyDirection::OTHER_TO_SELF));
            cn.copy_to_host(dst, src, size);
        }
    }
}

static void forward_other_memory(
        const DeviceTensor *self,
        const Desc &other, CallbackOnce deleter) {
    mgb_assert(self->desc.dtype == other.dtype, "dtype mismatch: %s vs %s",
               self->dtype_name(), dtype_name(other.dtype));
    auto deleter_wrap = [deleter]() mutable { deleter.consume(); };
    thin_function<void(void*)> deleter_dispatch;
    if (self->desc.type == Type::CPU) {
        CompNode cn{};
        if (self->m_varptr) {
            cn = static_cast<cg::VarNode*>(self->m_varptr)->comp_node();
        } else {
            cn = static_cast<DeviceTensorND*>(self->m_dev_nd)->comp_node();
        }
        deleter_dispatch = [d = new DeleteDispatcher(deleter_wrap, cn)](void*) {
            d->trigger();
            delete d;
        };
    } else {
        deleter_dispatch = [deleter_wrap](void*) mutable { deleter_wrap(); };
    }
    auto shape = desc_shape_to_tensor_shape(other);
    if (self->m_varptr) {
        auto var = static_cast<cg::VarNode*>(self->m_varptr);
        DeviceTensorStorage storage;
        storage.reset(var->comp_node(),
                shape.total_nr_elems() * var->dtype().size(),
                {static_cast<dt_byte*>(other.dev_ptr), deleter_dispatch});
        DeviceTensorND tensor;
        tensor.reset(storage, {shape, var->dtype()});
        var->reset_dev_tensor_from_tensor(tensor);
    } else {
        DeviceTensorND& tensor = *static_cast<DeviceTensorND*>(self->m_dev_nd);
        DeviceTensorStorage storage;
        size_t dtype_size = tensor.layout().dtype.size();
        storage.reset(tensor.comp_node(),
                shape.total_nr_elems() * dtype_size,
                {static_cast<dt_byte*>(other.dev_ptr), deleter_dispatch});
        tensor.reset(storage, {shape, tensor.layout().dtype});
    }
}

static void forward_to(
        const DeviceTensor *self,
        void **dest, CallbackOnce* deleter) {
    auto orig_dv_ptr = static_cast<DeviceTensorStorage*>(self->m_dev_nd);
    *dest = orig_dv_ptr->ptr();
    mgb_assert(*dest == self->desc.dev_ptr);
    deleter->user_data = new DeviceTensorStorage(*orig_dv_ptr);
    deleter->fptr = [](void* ptr) {
        delete reinterpret_cast<DeviceTensorStorage*>(ptr);
    };
}

static void init_tensor(pubapi::DeviceTensor& dest, DeviceTensorND* tensor,
                        VarNode* var, bool readonly) {
    memset(&dest, 0, sizeof(pubapi::DeviceTensor));
    {
        static FuncTable functable{&sync, &copy, &forward_other_memory,
                                   &dtype_name, &forward_to};
        dest.m_functable = &functable;
    }
    dest._version0 = dest._version1 = CURRENT_VERSION;

    mgb_assert((!!tensor) ^ (!!var));
    auto cn = tensor ? tensor->comp_node() : var->comp_node();
    using Type = pubapi::DeviceTensor::Type;
    switch (cn.device_type()) {
        case CompNode::DeviceType::CPU:
            dest.desc.type = Type::CPU;
            break;
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            dest.desc.type = Type::CUDA;
            break;
#endif
        default:
            mgb_throw(MegBrainError, "bad comp node type: %d",
                      static_cast<int>(cn.device_type()));
    }
    dest.desc.dtype = dtype_mgb2pubapi(tensor ? tensor->dtype() : var->dtype());
    if (tensor) {
        dest.desc.dev_ptr = tensor->raw_ptr();
        dest.desc.shape = tensor->shape().shape;
        dest.desc.ndim = tensor->shape().ndim;
        dest.size_bytes = tensor->layout().span().dist_byte();
    }
#if MGB_CUDA
    if (dest.desc.type == Type::CUDA) {
        auto&& env = CompNodeEnv::from_comp_node(cn).cuda_env();
        dest.desc.cuda_ctx.device = env.device;
        dest.desc.cuda_ctx.stream = env.stream;
    }
#endif
    dest.m_readonly = readonly;
    dest.m_dev_nd = tensor;
    dest.m_varptr = var;
}

};  // pubapi::DeviceTensor::Impl

void mgb::init_pubapi_dev_tensor(pubapi::DeviceTensor& dest,
                                 DeviceTensorND* tensor, VarNode* var,
                                 bool readonly) {
    pubapi::DeviceTensor::_Impl::init_tensor(dest, tensor, var, readonly);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
