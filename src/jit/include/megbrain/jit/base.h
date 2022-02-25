#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "megbrain/gopt/gtrans.h"
#include "megbrain/jit/internal_graph.h"

#if MGB_JIT

namespace mgb {
namespace jit {

using InternalNode = VarNode;

using Kernel = std::pair<std::string, std::string>;

using InternalNodePtr = std::shared_ptr<InternalNode>;

using NodePtr = std::shared_ptr<VarNode>;

using InternalGraphPtr = std::shared_ptr<InternalGraph>;

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
