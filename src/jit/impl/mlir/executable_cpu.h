/**
 * \file src/jit/impl/mlir/executable_cpu.h
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

#include "megbrain/jit/compiler.h"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Module.h>

namespace mgb {
namespace jit {

/*!
 * \brief Executable class for MLIR
 */
class MLIRCPUExecutable final : public Executable {
public:
    MLIRCPUExecutable(mlir::OwningModuleRef& module,
                      const std::string& kernel_name);
    ~MLIRCPUExecutable();

    /*!
     * \brief execute
     * A executable instance can be executed by one or more fusion_opr
     */
    void execute(JITExecutor* fusion_opr) override final;

private:
    std::unique_ptr<mlir::ExecutionEngine> m_engine;
    std::string m_kernel_name;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
