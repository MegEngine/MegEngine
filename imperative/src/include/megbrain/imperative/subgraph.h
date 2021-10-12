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
        return ssprintf(
                "%s = %s %s\n", to_string(expr.outputs).c_str(),
                to_string(expr.op.get()).c_str(), to_string(expr.inputs).c_str());
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
            mgb_assert(
                    expr_outputs.size() == expr.outputs.size(), "output size mismatch");
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

struct EncodedSubgraph {
    Subgraph graph;
    SmallVector<bool> input_mask;
    SmallVector<bool> output_mask;

    template <typename TContainer>
    TContainer encode_inputs(TContainer inputs) const {
        TContainer encoded_inputs;
        size_t index = 0;
        for (auto&& input : inputs) {
            mgb_assert(index < input_mask.size(), "index out of range");
            if (input_mask[index++]) {
                encoded_inputs.push_back(input);
            }
        }
        mgb_assert(index == input_mask.size(), "mask size mismatch");
        return encoded_inputs;
    }

    template <typename TContainer>
    TContainer encode_outputs(TContainer outputs) const {
        TContainer encoded_outputs;
        size_t index = 0;
        for (auto&& output : outputs) {
            mgb_assert(index < output_mask.size(), "index out of range");
            if (output_mask[index++]) {
                encoded_outputs.push_back(output);
            }
        }
        mgb_assert(index == output_mask.size(), "mask size mismatch");
        return encoded_outputs;
    }

    template <typename TContainer>
    TContainer decode_outputs(TContainer outputs) const {
        TContainer decoded_outputs;
        size_t index = 0;
        for (size_t i = 0; i < output_mask.size(); i++) {
            mgb_assert(index < output_mask.size(), "index out of range");
            if (output_mask[i]) {
                decoded_outputs.push_back(outputs[index++]);
            } else {
                decoded_outputs.emplace_back();
            }
        }
        mgb_assert(decoded_outputs.size() == output_mask.size(), "mask size mismatch");
        return decoded_outputs;
    }

    static EncodedSubgraph make(Subgraph graph) {
        EncodedSubgraph result;
        result.input_mask = graph.gen_input_mask();
        result.output_mask = graph.gen_output_mask();
        graph.inputs = result.encode_inputs(graph.inputs);
        graph.outputs = result.encode_outputs(graph.outputs);
        result.graph = graph;
        return result;
    }

    static EncodedSubgraph make_single(
            std::shared_ptr<OpDef> op, SmallVector<bool> input_mask,
            SmallVector<bool> output_mask) {
        EncodedSubgraph result;
        result.input_mask = input_mask;
        result.output_mask = output_mask;
        Subgraph::var_t last_var = 0;
        for (auto&& mask : input_mask) {
            if (mask) {
                result.graph.inputs.push_back(++last_var);
            }
        }
        for (auto&& mask : output_mask) {
            if (mask) {
                result.graph.outputs.push_back(++last_var);
            }
        }
        result.graph.exprs = {
                Subgraph::expr_t{op, result.graph.inputs, result.graph.outputs}};
        return result;
    }

    template <typename T, typename F, typename C>
    SmallVector<T> apply(SmallVector<T> input_vars, F&& f, C&& c) const {
        auto encoded_inputs = encode_inputs(input_vars);
        auto encoded_outputs =
                graph.apply(encoded_inputs, std::forward<F>(f), std::forward<C>(c));
        return decode_outputs(encoded_outputs);
    }

    std::string repr() const;
    size_t hash() const;
};

template <typename T>
class GradContext {
public:
    using var_t = T;
    using vars_t = SmallVector<var_t>;
    using expr_t = Expr<T>;

private:
    std::unordered_map<var_t, var_t> m_grads;
    std::unordered_set<var_t> m_vars_require_grad;
    std::function<var_t(var_t, var_t)> m_accumulator;
    std::vector<expr_t> m_exprs;

public:
    GradContext(std::function<var_t(var_t, var_t)> accumulator)
            : m_accumulator{std::move(accumulator)} {}
    SmallVector<bool> get_require_grads(vars_t dests) {
        SmallVector<bool> mask;
        for (auto&& dest : dests) {
            mask.push_back(bool(m_vars_require_grad.count(dest)));
        }
        return mask;
    }
    SmallVector<bool> get_has_grads(vars_t dests) {
        SmallVector<bool> mask;
        for (auto&& dest : dests) {
            mask.push_back(bool(m_grads.count(dest)));
        }
        return mask;
    }
    void mark_require_grads(vars_t dests) {
        for (auto&& dest : dests) {
            m_vars_require_grad.insert(dest);
        }
    }
    var_t accumulate_grad(var_t dest, var_t grad) {
        if (!m_grads.count(dest)) {
            return m_grads[dest] = grad;
        } else {
            return m_grads[dest] = m_accumulator(m_grads[dest], grad);
        }
    }
    void record_expr(std::shared_ptr<OpDef> op, vars_t inputs, vars_t outputs) {
        bool require_grad = false;
        for (auto&& input : inputs) {
            if (m_vars_require_grad.count(input)) {
                require_grad = true;
                break;
            }
        }
        if (require_grad) {
            m_exprs.push_back({op, inputs, outputs});
            mark_require_grads(outputs);
        }
    }
    template <typename TFunctor>
    void backward(vars_t outputs, vars_t output_grads, TFunctor functor) {
        size_t nr_outputs = outputs.size();
        for (size_t i = 0; i < nr_outputs; ++i) {
            m_grads[outputs[i]] = output_grads[i];
        }
        auto exprs = m_exprs;
        std::reverse(exprs.begin(), exprs.end());
        for (const expr_t& expr : exprs) {
            size_t nr_inputs = expr.inputs.size();
            vars_t input_grads = functor(expr, get_grads(expr.outputs));
            mgb_assert(input_grads.size() == nr_inputs, "input size mismatch");
            for (size_t i = 0; i < nr_inputs; ++i) {
                if (input_grads[i] && m_vars_require_grad.count(expr.inputs[i])) {
                    accumulate_grad(expr.inputs[i], input_grads[i]);
                }
            }
        }
    }
    var_t get_grad(var_t dest) {
        if (m_grads.count(dest)) {
            return m_grads.at(dest);
        }
        return 0;
    }
    vars_t get_grads(vars_t dests) {
        vars_t grads;
        for (auto&& dest : dests) {
            grads.push_back(get_grad(dest));
        }
        return grads;
    }
};

}  // namespace imperative
}  // namespace mgb