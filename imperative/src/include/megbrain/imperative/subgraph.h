/**
 * \file imperative/src/include/megbrain/imperative/subgraph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <list>

#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/imperative/utils/to_string.h"
#include "megbrain/utils/small_vector.h"

namespace mgb {
namespace imperative {

class OpDef;

template <typename T>
struct Expr {
    std::shared_ptr<OpDef> op;
    SmallVector<T> inputs;
    SmallVector<T> outputs;
};

template <typename T>
struct ToStringTrait<Expr<T>> {
    std::string operator()(const Expr<T>& expr) {
        return ssprintf("%s = %s %s\n", to_string(expr.inputs).c_str(), to_string(expr.op.get()).c_str(), to_string(expr.outputs).c_str());
    }
};

struct Subgraph {
    
    template <typename TDesc>
    class Builder;

    using var_t = size_t;
    using vars_t = SmallVector<size_t>;
    using op_t = std::shared_ptr<OpDef>;
    using expr_t = Expr<var_t>;

    template <typename TDesc>
    using builder_t = Builder<TDesc>;
    
    SmallVector<var_t> inputs;
    SmallVector<std::pair<var_t, TensorPtr>> constants;
    SmallVector<var_t> outputs;
    SmallVector<expr_t> exprs;

    template <typename T, typename F, typename C>
    SmallVector<T> apply(SmallVector<T> input_vars, F&& f, C&& c) const {
        std::unordered_map<size_t, T> idx2var;
        mgb_assert(inputs.size() == input_vars.size(), "input size mismatch");
        for (size_t i = 0; i < inputs.size(); ++i) {
            idx2var[inputs[i]] = input_vars[i];
        }
        for (auto&& [idx, val] : constants) {
            idx2var[idx] = c(val);
        }
        for (auto& expr : exprs) {
            SmallVector<T> expr_inputs;
            for (auto idx : expr.inputs) {
                expr_inputs.push_back(idx2var[idx]);
            }
            SmallVector<T> expr_outputs =
                    f(expr.op, std::move(expr_inputs), expr.outputs.size());
            mgb_assert(expr_outputs.size() == expr.outputs.size(),
                       "output size mismatch");
            for (size_t i = 0; i < expr_outputs.size(); ++i) {
                idx2var[expr.outputs[i]] = expr_outputs[i];
            }
        }
        SmallVector<T> output_vars;
        for (auto idx : outputs) {
            output_vars.push_back(idx2var[idx]);
        }
        return output_vars;
    }

    void remove_unused_exprs();
    SmallVector<bool> gen_input_mask();
    SmallVector<bool> gen_output_mask();
    bool empty() const { return outputs.size() == 0; }
    void replace_vars(const std::unordered_map<size_t, size_t>& replace_map);
    std::string repr() const;
    bool is_single() const;
    std::shared_ptr<OpDef> as_single() const;
    bool operator==(const Subgraph& rhs) const;
};

}  // namespace imperative
}  // namespace mgb