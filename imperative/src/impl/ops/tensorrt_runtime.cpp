/**
 * \file imperative/src/impl/ops/tensorrt_runtime.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_runtime_opr.h"
namespace mgb::imperative {

namespace { namespace tensorrt_runtime {
    auto apply_on_var_node(
            const OpDef& def,
            const VarNodeArray& inputs) {
        auto&& op = static_cast<const TensorRTRuntime&>(def);
        OperatorNodeConfig config{op.make_name()};
        SymbolVarArray sinputs(inputs.begin(), inputs.end());
        return opr::TensorRTRuntimeOpr::make(op.buf.c_str(), op.buf_size, sinputs, config);
    }
OP_TRAIT_REG(TensorRTRuntime, TensorRTRuntime)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // tensorrt_runtime

} // namespace mgb::imperative
#endif  // MGB_ENABLE_TENSOR_RT
