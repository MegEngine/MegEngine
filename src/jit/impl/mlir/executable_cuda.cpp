/**
 * \file src/jit/impl/mlir/executable_cuda.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <vector>
#include "megbrain_build_config.h"
#include "megdnn/dtype.h"
#if MGB_JIT && MGB_JIT_MLIR

#if MGB_CUDA
#include "./executable_cuda.h"
#include "./utils.h"
#include "megbrain/utils/timer.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/comp_node_env.h"

#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/IR/OpDefinition.h>

using namespace mgb;
using namespace jit;

namespace {
template <int out_dim, typename ctype>
void setup_and_launch(const JITExecutor* fusion_opr, CUfunction func,
                      int block_size) {
    auto&& args = fusion_opr->args();
    std::vector<StridedMemRefType<ctype, out_dim>> param_holders;
    std::vector<void*> params;

    auto set_params = [&param_holders, &params](
                              void* ptr, const megdnn::TensorLayout& layout) {
        param_holders.push_back(StridedMemRefType<ctype, out_dim>{});
        StridedMemRefType<ctype, out_dim>& desc = param_holders.back();
        desc.basePtr = static_cast<ctype*>(ptr);
        params.push_back(&(desc.basePtr));
        desc.data = static_cast<ctype*>(ptr);
        params.push_back(&(desc.data));
        desc.offset = 0;
        params.push_back(&(desc.offset));
        for (size_t i = 0; i < layout.ndim; i++) {
            desc.sizes[i] = layout.shape[i];
            params.push_back(&(desc.sizes[i]));
            desc.strides[i] = layout.stride[i];
            params.push_back(&(desc.strides[i]));
        }
    };
    for (const auto& arg : args.inputs) {
        set_params(arg.from->dev_tensor().raw_ptr(), arg.layout);
    }
    int64_t nr_elements = 0;
    for (const auto& arg : args.outputs) {
        if (nr_elements == 0) {
            nr_elements = arg.layout.total_nr_elems();
        } else {
            mgb_assert(static_cast<size_t>(nr_elements) ==
                               arg.layout.total_nr_elems(),
                       "The number of elements of outputs mismatch, expected: "
                       "%zu got: %zu(%s)",
                       static_cast<size_t>(nr_elements),
                       arg.layout.total_nr_elems(),
                       arg.layout.to_string().c_str());
        }

        set_params(arg.from->dev_tensor().raw_ptr(), arg.layout);
    }
    const CompNodeEnv& env =
            CompNodeEnv::from_comp_node(fusion_opr->comp_node());

    int64_t num_block = (nr_elements - 1) / block_size + 1;
    params.insert(params.begin(), &nr_elements);
    MGB_CUDA_CU_CHECK(cuLaunchKernel(func, num_block, 1, 1, block_size, 1, 1, 0,
                                     env.cuda_env().stream, params.data(), 0));
}
}  // namespace

const std::string MLIRCUDAExecutable::sm_blob_annotation = "nvvm.cubin";
MLIRCUDAExecutable::MLIRCUDAExecutable(mlir::OwningModuleRef& module,
                                       const std::string& kernel_name) {
    m_kernel_name = kernel_name + "_kernel";
    auto kernel_module =
            module->lookupSymbol<mlir::gpu::GPUModuleOp>(m_kernel_name);
    mgb_assert(kernel_module, "Expected gpu kernel module");

    auto binary_attr = kernel_module.getAttrOfType<mlir::StringAttr>(
            llvm::StringRef(sm_blob_annotation));
    mgb_assert(binary_attr, "Missing %s attribute in gpu kernel module",
               sm_blob_annotation.c_str());
    m_kernel_data = binary_attr.getValue().str();
}

void MLIRCUDAExecutable::execute(JITExecutor* fusion_opr) {
    FuncCache* func;
    auto cn = fusion_opr->comp_node();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    func = &m_func_cache[{prop.major, prop.minor}];
    func->kernel_data = m_kernel_data;
    func->exec(fusion_opr, this);
}

MLIRCUDAExecutable::~MLIRCUDAExecutable() {}

void MLIRCUDAExecutable::FuncCache::exec(const JITExecutor* fusion_opr,
                                         const MLIRCUDAExecutable* cuda_exe) {
    Func* func;
    {
        MGB_LOCK_GUARD(mtx);
        auto ins = cn2func.insert({fusion_opr->comp_node(), {}});
        func = &ins.first->second;
        if (ins.second) {
            MGB_CUDA_CU_CHECK(
                    cuModuleLoadData(&func->module, kernel_data.data()));
            MGB_CUDA_CU_CHECK(
                    cuModuleGetFunction(&func->func, func->module,
                                        cuda_exe->m_kernel_name.c_str()));
            int min_grid_size = 0;
            MGB_CUDA_CU_CHECK(cuOccupancyMaxPotentialBlockSize(
                    &min_grid_size, &func->block_size, func->func, nullptr, 0,
                    0));
        }
    }

    mgb_assert(fusion_opr->args().outputs.size() == 1,
               "Currently only support 1 outputs, got %zu",
               fusion_opr->args().outputs.size());
    int out_dim = fusion_opr->args().outputs[0].layout.ndim;
    DType dtype = fusion_opr->args().outputs[0].layout.dtype;
#define cb_outdim(_ndim, _dtype)                                \
    if (_ndim == out_dim) {                                     \
        setup_and_launch<_ndim, _dtype>(fusion_opr, func->func, \
                                        func->block_size);      \
        return;                                                 \
    }

#define cb(_dtype)                                      \
    cb_outdim(1, float);                                \
    cb_outdim(2, float);                                \
    cb_outdim(3, float);                                \
    cb_outdim(4, float);                                \
    mgb_throw(InternalError, "unsupported out_dim=%zu", \
              static_cast<size_t>(out_dim));            \
    return;

    switch (dtype.enumv()) {
        case DTypeEnum::Float32:
            cb(float);
        default:
            mgb_throw(InternalError, "unsupport dtype: %s", dtype.name());
    }
#undef cb
#undef cb_outdim
}

#endif  // MGB_CUDA
#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
