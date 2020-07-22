/**
 * \file src/jit/impl/mlir/executable_cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#if MGB_CUDA
#include "megbrain/jit/compiler.h"

#include <mlir/IR/Module.h>

#include <cuda.h>

namespace mgb {
namespace jit {

/*!
 * \brief Executable class for MLIR
 */
class MLIRCUDAExecutable final : public Executable {
public:
    MLIRCUDAExecutable(mlir::OwningModuleRef& module,
                       const std::string& kernel_name);
    ~MLIRCUDAExecutable();

    /*!
     * \brief execute
     * A executable instance can be executed by one or more fusion_opr
     */
    void execute(JITExecutor* fusion_opr) override final;

    const static std::string sm_blob_annotation;
private:
    //! cache for a func on a specific device
    struct FuncCache {
        struct Func {
            int block_size{-1};
            CUmodule module{nullptr};
            CUfunction func{nullptr};
        };

        std::mutex mtx;
        std::string kernel_data;
        CompNode::UnorderedMap<Func> cn2func;

        void exec(const JITExecutor* fusion_opr,
                  const MLIRCUDAExecutable* cuda_exe);
    };

    std::string m_kernel_name;
    std::string m_kernel_data;

    //! (cuda_major, cuda_minor) => func
    ThinHashMap<std::pair<uint32_t, uint32_t>, FuncCache> m_func_cache;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_CUDA
#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
