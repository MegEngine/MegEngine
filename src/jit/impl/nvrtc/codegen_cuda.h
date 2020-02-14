/**
 * \file src/jit/impl/nvrtc/codegen_cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#if MGB_JIT && MGB_CUDA

#include "megbrain/jit/executor_opr.h"

namespace mgb {
namespace jit {
/*!
 * \brief generate cuda kernel source code
 * \return (kernel name, kernel source)
 */
std::pair<std::string, std::string> codegen_cuda(
        const InternalGraph& internal_graph, const JITExecutor::Args& args,
        bool copy_param_to_dev);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_CUDA

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
