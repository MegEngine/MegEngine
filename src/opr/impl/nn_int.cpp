/**
 * \file src/opr/impl/nn_int.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/opr/nn_int.h"
#include "./internal/megdnn_opr_wrapper.inl"

#include "megdnn/oprs/general.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AffineInt);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ElemwiseMultiType);

ElemwiseMultiType::ElemwiseMultiType(const VarNodeArrayView& inputs,
                                     Param param,
                                     const OperatorNodeConfig& config)
        : Super{inputs.at(0)->owner_graph(), config,
                ModeTrait::from_mode(param.mode).name, inputs} {
    Super::init_megdnn_opr(*this, param);
    for (auto i : inputs) {
        add_input({i});
    }
}

SymbolVar ElemwiseMultiType::make(const VarNodeArrayView& inputs, Param param,
                                  const OperatorNodeConfig& config) {
    mgb_assert(!inputs.empty());
    return SymbolVar{inputs[0]}.insert_single_output_opr<ElemwiseMultiType>(
            inputs, param, config);
}

void ElemwiseMultiType::init_output_dtype() {
    auto trait = ModeTrait::from_mode(param().mode);
    mgb_throw_if(trait.arity != input().size(), MegBrainError,
                 "%s requires %u inputs, but %zu are given", trait.name,
                 trait.arity, input().size());
    for (size_t i = 0; i < trait.arity; ++i) {
        auto dtype = input()[i]->dtype();
        trait.check_inp[i](dtype);
    }
    if (trait.need_specify_out_dtype) {
        auto dtype = config().output_dtype();
        mgb_assert(dtype.valid());
        output(0)->dtype(dtype);
        trait.check_out(dtype, true);
    } else {
        DType dtype;
        trait.check_out(dtype, false);
        output(0)->dtype(dtype);
    }
}

void ElemwiseMultiType::scn_do_execute() {
    megdnn::TensorNDArray inp_arr(input().size());
    for (size_t i = 0; i < input().size(); ++i) {
        inp_arr[i] = input()[i]->dev_tensor().as_megdnn();
    }
    megdnn_opr()->exec(inp_arr, output(0)->dev_tensor().as_megdnn());
}

void ElemwiseMultiType::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    mgb_assert(out_shape.size() == 1);
    megdnn::Elemwise::deduce_shape(inp_shape, out_shape[0]);
}

void ElemwiseMultiType::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
