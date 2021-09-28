/**
 * \file imperative/src/include/megbrain/imperative/graph_builder.h
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

#include "megbrain/imperative/subgraph.h"

namespace mgb {
namespace imperative {

template <typename TDesc>
class Subgraph::Builder {
    using graph_t = Subgraph;
    using var_t = graph_t::var_t;
    using vars_t = graph_t::vars_t;
    using op_t = graph_t::op_t;
    using expr_t = graph_t::expr_t;
    using exprs_t = std::list<expr_t>;
    using expr_iter_t = std::list<expr_t>::iterator;
    using desc_t = TDesc;
    using descs_t = SmallVector<TDesc>;
    using infer_fn_t = std::function<descs_t(op_t, descs_t, size_t)>;
    using encoded_graph_t = EncodedSubgraph;
    using var_map_t = std::unordered_map<var_t, var_t>;
    vars_t m_inputs;
    SmallVector<std::pair<var_t, TensorPtr>> m_constants;
    vars_t m_outputs;
    exprs_t m_exprs;
    var_t m_last_var = 0;
    std::unordered_map<var_t, TDesc> m_var2desc;
    infer_fn_t m_infer_fn;
    var_map_t m_var_replace_map;

private:
    var_t next_var() { return ++m_last_var; }

public:
    explicit Builder(std::function<descs_t(op_t, descs_t, size_t)> infer_function)
            : m_infer_fn{infer_function} {}
    vars_t write_expr(op_t op, vars_t inputs, size_t nr_outputs) {
        return write_expr_before(m_exprs.end(), std::move(op),
                                 std::move(inputs), std::move(nr_outputs));
    }
    vars_t write_expr_before(expr_iter_t iter, op_t op, vars_t inputs,
                             size_t nr_outputs) {
        vars_t outputs;
        for (size_t i = 0; i < nr_outputs; ++i) {
            outputs.push_back(next_var());
        }
        m_exprs.insert(iter, {op, inputs, outputs});
        descs_t input_descs = get_descs(inputs);
        descs_t output_descs = m_infer_fn(op, input_descs, nr_outputs);
        mgb_assert(output_descs.size() == nr_outputs,
                   "bad infer_function: output descs size mismatch");
        for (size_t i = 0; i < nr_outputs; ++i) {
            m_var2desc[outputs[i]] = output_descs[i];
        }
        return outputs;
    }
    var_t write_constant(TensorPtr constant, desc_t desc) {
        var_t constant_var = next_var();
        m_constants.emplace_back(constant_var, constant);
        m_var2desc[constant_var] = std::move(desc);
        return constant_var;
    }
    var_t write_input(desc_t input_desc) {
        var_t input = next_var();
        m_var2desc[input] = input_desc;
        m_inputs.push_back(input);
        return input;
    }
    vars_t write_inputs(descs_t input_descs) {
        vars_t inputs;
        for (auto&& input_desc: input_descs) {
            inputs.push_back(write_input(input_desc));
        }
        return inputs;
    }
    void add_output(var_t var) { m_outputs.push_back(var); }
    void add_outputs(vars_t vars) {
        m_outputs.insert(m_outputs.begin(), vars.begin(), vars.end());
    }
    desc_t get_desc(var_t var) { return m_var2desc.at(var); }
    descs_t get_descs(vars_t vars) {
        descs_t descs;
        for (auto&& var : vars) {
            descs.push_back(get_desc(var));
        }
        return descs;
    }
    encoded_graph_t encode() const {
        graph_t graph{m_inputs,
                      m_constants,
                      m_outputs,
                      {m_exprs.begin(), m_exprs.end()}};
        graph.replace_vars(m_var_replace_map);
        graph.remove_unused_exprs();
        return encoded_graph_t::make(std::move(graph));
    }
    void replace_var(var_t old_var, var_t new_var) {
        mgb_assert(!m_var_replace_map.count(old_var),
                   "var cannot be replaced twice");
        m_var_replace_map[old_var] = new_var;
    }
    template <typename TFunctor>
    void iterate(TFunctor&& functor) {
        for (expr_iter_t iter = m_exprs.begin(); iter != m_exprs.end();
             ++iter) {
            functor(iter);
        }
    }
    template <typename TFunctor>
    void reverse_iterate(TFunctor&& functor) {
        for (expr_iter_t iter = --m_exprs.end();; --iter) {
            functor(iter);
            if (iter == m_exprs.begin()) {
                break;
            }
        }
    }
    expr_iter_t begin() { return m_exprs.begin(); }
    expr_iter_t end() { return m_exprs.end(); }
};
}  // namespace imperative
}  // namespace mgb