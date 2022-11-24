#pragma once

#include "megbrain_build_config.h"

#if MGB_JIT && MGB_OPENCL

#include "megbrain/jit/executor_opr.h"

namespace mgb {
namespace jit {
/*!
 * \brief generate opencl kernel source code
 * \return (kernel name, kernel source)
 */
std::pair<std::string, std::string> codegen_opencl(
        const InternalGraph& internal_graph, const JITExecutor::Args& args);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_OPENCL
