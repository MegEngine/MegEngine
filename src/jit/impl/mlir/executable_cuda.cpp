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

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR
#if MGB_CUDA

#include "./executable_cuda.h"
#include "./ir/types.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/jit/mlir/ir/utils.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/utils/timer.h"
#include "megdnn/dtype.h"

#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/OpDefinition.h>

using namespace mgb;
using namespace jit;

namespace {

int64_t get_grid_size(int64_t nr_elements, int64_t block_size) {
    // unroll three times in the kernel
    int64_t a = nr_elements / (block_size * 2);
    int64_t b = (nr_elements - 1) / (block_size * 3) + 1;
    return std::max(a, b);
}

template <int out_dim, typename ctype>
void setup_and_launch(const JITExecutor* fusion_opr, CUfunction func,
                      int block_size) {
    auto&& args = fusion_opr->args();
    size_t num_memrefs = args.inputs.size() + args.outputs.size();
    std::vector<StridedMemRefType<ctype, out_dim>> param_holders(num_memrefs);
    std::vector<void*> params;

    auto set_params = [&param_holders, &params](
                              size_t idx, void* ptr,
                              const megdnn::TensorLayout& layout) {
        auto& desc = param_holders[idx];
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

    size_t idx = 0;
    for (const auto& arg : args.inputs) {
        set_params(idx++, arg.from->dev_tensor().raw_ptr(), arg.from->layout());
    }

    int64_t nr_elements = 0;
    for (const auto& arg : args.outputs) {
        if (nr_elements == 0) {
            nr_elements = arg.from->layout().total_nr_elems();
        } else {
            mgb_assert(static_cast<size_t>(nr_elements) ==
                               arg.layout.total_nr_elems(),
                       "The number of elements of outputs mismatch, expected: "
                       "%zu got: %zu(%s)",
                       static_cast<size_t>(nr_elements),
                       arg.from->layout().total_nr_elems(),
                       arg.from->layout().to_string().c_str());
        }

        set_params(idx++, arg.from->dev_tensor().raw_ptr(), arg.from->layout());
    }

    mgb_assert(param_holders.size() == num_memrefs,
               "calling push_back method of param_holders is unsafe as it "
               "might cause reallocation of std::vector");

    const CompNodeEnv& env =
            CompNodeEnv::from_comp_node(fusion_opr->comp_node());

    int64_t grid_size;
    if (nr_elements <= block_size) {
        block_size = nr_elements;
        grid_size = 1;
    } else {
        grid_size = get_grid_size(nr_elements, block_size);
    }
    int64_t nr_threads = grid_size * block_size;
    params.push_back(&nr_elements);
    params.push_back(&nr_threads);

    MGB_CUDA_CU_CHECK(cuLaunchKernel(func, grid_size, 1, 1, block_size, 1, 1, 0,
                                     env.cuda_env().stream, params.data(), 0));
}

template <int out_dim>
void setup_and_launch_dim(const megdnn::DType dtype,
                          const JITExecutor* fusion_opr, CUfunction func,
                          int block_size) {
    switch (dtype.enumv()) {
#define cb(_dtype, _type)                                               \
    case megdnn::DTypeEnum::_dtype:                                     \
        setup_and_launch<out_dim, _type>(fusion_opr, func, block_size); \
        return;
        FOR_EACH_DNN_DTYPE(cb)
#undef cb
        default:
            mgb_throw(InternalError, "Unsupported dtype: %s", dtype.name());
    }
    return;
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
    int out_dim = fusion_opr->args().outputs[0].from->layout().ndim;
    DType dtype = fusion_opr->args().outputs[0].from->layout().dtype;

    switch (out_dim) {
#define cb(_ndim)                                                  \
    case _ndim:                                                    \
        setup_and_launch_dim<_ndim>(dtype, fusion_opr, func->func, \
                                    func->block_size);             \
        break;
        cb(1);
        cb(2);
        cb(3);
        cb(4);
#undef cb
    }
}

#endif  // MGB_CUDA
#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
