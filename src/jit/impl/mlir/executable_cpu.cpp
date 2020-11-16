/**
 * \file src/jit/impl/mlir/executable_cpu.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "./executable_cpu.h"
#include "./ir/types.h"

#include "megbrain/jit/mlir/ir/utils.h"

#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>

using namespace mgb;
using namespace jit;

namespace {

template <typename T, int N>
StridedMemRefType<T, N>* get_strided_memref_type(
        const megdnn::TensorND& tensor) {
    using DescType = StridedMemRefType<T, N>;
    DescType* desc = static_cast<DescType*>(malloc(sizeof(DescType)));
    desc->basePtr = tensor.ptr<T>();
    desc->data = tensor.ptr<T>();
    desc->offset = 0;
    for (size_t i = 0; i < tensor.layout.ndim; i++) {
        desc->sizes[i] = tensor.layout.shape[i];
        desc->strides[i] = tensor.layout.stride[i];
    }
    return desc;
}

template <int N>
void* tensor2memref_dim(const megdnn::TensorND& tensor) {
    switch (tensor.layout.dtype.enumv()) {
#define cb(_dtype, _type)           \
    case megdnn::DTypeEnum::_dtype: \
        return get_strided_memref_type<_type, N>(tensor);
        FOR_EACH_DNN_DTYPE(cb)
#undef cb
        default:
            mgb_throw(InternalError, "Unsupported dtype: %s",
                      tensor.layout.dtype.name());
    }
    return nullptr;
}

void* tensor2memref(const megdnn::TensorND& tensor) {
    switch (tensor.layout.ndim) {
#define cb(i) \
    case i:   \
        return tensor2memref_dim<i>(tensor)

        cb(1);
        cb(2);
        cb(3);
        cb(4);
        cb(5);
        default:
            mgb_throw(InternalError, "Unsupported ndim, got %zu",
                      tensor.layout.ndim);
#undef cb
    }
}

}  // namespace
MLIRCPUExecutable::MLIRCPUExecutable(mlir::OwningModuleRef& module,
                                     const std::string& kernel_name)
        : m_kernel_name{kernel_name} {
    auto opt_pipeline = mlir::makeOptimizingTransformer(3, 3, 0);
    std::vector<std::string> libs;
    auto&& engine = mlir::ExecutionEngine::create(
            *module, nullptr, opt_pipeline, llvm::None,
            std::vector<llvm::StringRef>(libs.begin(), libs.end()), true,
            false);
    mgb_assert(engine);
    m_engine = std::move(*engine);
}

void MLIRCPUExecutable::execute(JITExecutor* fusion_opr) {
    auto&& args = fusion_opr->args();
    std::vector<void*> args_array(args.inputs.size() + args.outputs.size());
    std::vector<void*> args_array_pointer(args.inputs.size() +
                                          args.outputs.size());
    size_t idx = 0;
    for (size_t i = 0; i < args.inputs.size(); i++) {
        args_array[idx] =
                tensor2memref({args.inputs[i].from->dev_tensor().raw_ptr(),
                               args.inputs[i].layout});
        args_array_pointer[idx] = &args_array[idx];
        idx++;
    }
    int64_t nr_elements = 0;
    for (size_t i = 0; i < args.outputs.size(); i++) {
        if (nr_elements == 0) {
            nr_elements = args.outputs[i].layout.total_nr_elems();
        } else {
            mgb_assert(static_cast<size_t>(nr_elements) ==
                               args.outputs[i].layout.total_nr_elems(),
                       "The number of elements of outputs mismatch, expected: "
                       "%zu got: %zu(%s)",
                       static_cast<size_t>(nr_elements),
                       args.outputs[i].layout.total_nr_elems(),
                       args.outputs[i].layout.to_string().c_str());
        }
        args_array[idx] =
                tensor2memref({args.outputs[i].from->dev_tensor().raw_ptr(),
                               args.outputs[i].layout});
        args_array_pointer[idx] = &args_array[idx];
        idx++;
    }

    args_array_pointer.push_back(&nr_elements);
    std::string adapter_name = std::string("_mlir_ciface_") + m_kernel_name;
    auto err = m_engine->invoke(
            adapter_name, llvm::MutableArrayRef<void*>(args_array_pointer));
    if (err) {
        mgb_throw(InternalError, "failed to run MLIR kernel %s\n",
                  m_kernel_name.c_str());
    }

    for (size_t i = 0; i < args_array.size(); i++) {
        free(args_array[i]);
    }
}

MLIRCPUExecutable::~MLIRCPUExecutable() {}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
