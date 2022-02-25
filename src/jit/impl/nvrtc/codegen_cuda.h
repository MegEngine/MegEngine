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
