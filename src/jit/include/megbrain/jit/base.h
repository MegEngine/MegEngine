/**
 * \file src/jit/include/megbrain/jit/base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

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
