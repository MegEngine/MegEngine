/**
 * \file src/core/include/megbrain/graph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/cg.h"
#include "megbrain/graph/helper.h"

namespace mgb {

using cg::ComputingGraph;
using cg::GraphError;
using cg::OperatorNodeConfig;
using cg::SymbolVar;
using cg::SymbolVarArray;
using cg::SymbolVarArrayView;
using cg::VarNode;
using cg::VarNodeArray;
using cg::VarNodeArrayView;

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
