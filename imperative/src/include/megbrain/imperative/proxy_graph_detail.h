/**
 * \file imperative/src/include/megbrain/imperative/proxy_graph_detail.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {
namespace proxy_graph_detail {

SmallVector<TensorPtr>
apply_on_physical_tensor(const OpDef& def,
        SmallVector<TensorPtr> inputs);

void execute(const OpDef& def,
        SmallVector<TensorPtr> inputs,
        SmallVector<TensorPtr> outputs,
        SmallVector<TensorPtr> workspace);

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs);

std::tuple<SmallVector<MemoryDesc>, SmallVector<MemoryDesc>> infer_output_mem_desc(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs_tensors,
        const SmallVector<MemoryDesc>& inputs_mems);

void exec(const OpDef& def,
        const SmallVector<TensorPtr>& inputs,
        const SmallVector<TensorPtr>& outputs);

EncodedSubgraph
make_backward_graph(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

} // namespace proxy_graph_detail
} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
