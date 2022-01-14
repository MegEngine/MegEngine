/**
 * \file imperative/src/include/megbrain/imperative/lazy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <future>
#include <variant>

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/opr/io.h"

namespace mgb::imperative {

class LazyEvalInfo {
private:
    VarNode* m_node = nullptr;
    ValueRef m_bound_data;
    std::string m_name;

public:
    LazyEvalInfo() = default;
    LazyEvalInfo(VarNode* node, ValueRef bound_data, std::string name)
            : m_node(node), m_bound_data(bound_data), m_name(name) {}
    VarNode* node() const { return m_node; }

    ValueRef bound_data() const { return m_bound_data; }

    std::string name() const { return m_name; }
};

class LazyEvalValue final : public MixinValueImpl<LazyEvalValue, LazyEvalInfo> {
public:
    using MixinValueImpl::MixinValueImpl;

    std::string to_string() const override {
        return ssprintf(
                "LazyEvalValue{node=%p, name=%s}", node(), node()->name().c_str());
    }
};

/**
 * \brief lazy evaluate on megbrain graph
 *
 * 1. Make a varnode for each external value (HoostToDeviceCopy/ImmutableTensor);
 * 2. Invoke apply_on_var_node when handling ApplyOp, return LazyEvalValue(VarNode) as
 * stub;
 * 3. Try infer value/shape when handling GetAttr;
 * 4. Compile and execute graph, get values and replace LazyEvalValues by concrete
 * values.
 */
class LazyEvalTransformation final : public Transformation {
private:
    bool m_no_exec;
    std::shared_ptr<ComputingGraph> m_graph;
    std::vector<LazyEvalValue::weak_ref_t> m_weak_vars;
    SymbolVar m_io_link = nullptr;
    std::exception_ptr m_graph_exc;

public:
    LazyEvalTransformation(bool no_exec) : m_no_exec(no_exec) {
        m_graph = ComputingGraph::make();
    }

    LazyEvalValue::ref_t record_var(
            VarNode* node, ValueRef bound_data = {}, std::string name = {}) {
        auto lazy_eval_val = LazyEvalValue::make(node, bound_data, name);
        m_weak_vars.push_back(lazy_eval_val);
        return lazy_eval_val;
    }

    ComputingGraph::Options& options() { return m_graph->options(); }

    std::vector<ValueRef> apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override {
        mgb_assert(!value.is<LazyEvalValue>());
        return value;
    }

    std::string name() const override { return "LazyEvalTransformation"; }

    void on_unregister() noexcept override;

    void check_exception();
};

}  // namespace mgb::imperative
