/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {
namespace proxy_graph_detail {

void exec(const OpDef& def,
        const SmallVector<TensorPtr>& inputs_,
        const SmallVector<TensorPtr>& outputs_);

SmallVector<LogicalTensorDesc> infer_output_attrs(const OpDef& def,
        const SmallVector<TensorPtr>& inputs);

SmallVector<LogicalTensorDesc>
infer_output_attrs_fallible(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs);

BackwardGraphResult
make_backward_graph(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

} // namespace proxy_graph_detail
} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}