/**
 * \file imperative/src/impl/ops/extern_opr.cpp
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

#include "megbrain/serialization/extern_c_opr_io.h"

namespace mgb::imperative {

namespace {
namespace externopr {

TensorShapeArray get_shapes(const std::vector<std::vector<size_t>>& shapes) {
    TensorShapeArray ret;
    for (auto&& i : shapes) {
        SmallVector<size_t> shape(i.begin(), i.end());
        TensorShape shp(shape);
        ret.push_back(shp);
    }
    return ret;
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const ExternOpr&>(def);
    SymbolVarArray symbol_var_inputs(inputs.begin(), inputs.end());

    SmallVector<DType> output_dtypes(op.output_dtypes.begin(), op.output_dtypes.end());
    auto&& output_shapes = get_shapes(op.output_shapes);

    cg::OperatorNodeBase* opr = opr::ExternCOprRunner::make_placeholder(
            symbol_var_inputs, output_shapes, op.name.c_str(), op.data.c_str(),
            op.data_len, {}, output_dtypes);
    return opr;
}

OP_TRAIT_REG(ExternOpr, ExternOpr, opr::ExternCOprRunner)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace externopr
}  // namespace

}  // namespace mgb::imperative
