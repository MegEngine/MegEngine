/**
 * \file imperative/src/include/megbrain/imperative/op_def.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/imperative/utils/to_string.h"

namespace mgb {
namespace imperative {

class OpDef;
struct OpTrait;

enum DispatchMode {
    DEFAULT_CPU = 0,
    KERNEL = 1
};

using SharedOp = std::shared_ptr<OpDef>;

template <typename T>
struct Expr {
    std::shared_ptr<OpDef> op;
    SmallVector<T> inputs;
    SmallVector<T> outputs;
};

struct Subgraph {
    SmallVector<size_t> inputs;
    SmallVector<std::pair<size_t, TensorPtr>> constants;
    SmallVector<size_t> outputs;
    SmallVector<Expr<size_t>> exprs;

    template <typename T, typename F, typename C>
    SmallVector<T> apply(SmallVector<T> input_vars, F&& f, C&& c) const {
        std::unordered_map<size_t, T> idx2var;
        mgb_assert(inputs.size() == input_vars.size(), "input size mismatch");
        for (size_t i = 0; i < inputs.size(); ++i) {
            idx2var[inputs[i]] = input_vars[i];
        }
        for (auto&& [idx, val]: constants) {
            idx2var[idx] = c(val);
        }
        for (auto& expr: exprs) {
            SmallVector<T> expr_inputs;
            for (auto idx: expr.inputs) {
                expr_inputs.push_back(idx2var[idx]);
            }
            SmallVector<T> expr_outputs = f(expr.op, std::move(expr_inputs));
            mgb_assert(expr_outputs.size() == expr.outputs.size(), "output size mismatch");
            for (size_t i = 0; i < expr_outputs.size(); ++i) {
                idx2var[expr.outputs[i]] = expr_outputs[i];
            }
        }
        SmallVector<T> output_vars;
        for (auto idx: outputs) {
            output_vars.push_back(idx2var[idx]);
        }
        return output_vars;
    }

    bool empty() const {
        return outputs.size() == 0;
    }

    std::string repr() const;
};

struct BackwardGraphResult {
    Subgraph backward;
    SmallVector<bool> save_for_backward;
    SmallVector<bool> input_has_grad;
};

class OpDef : public Hashable,
              public NonCopyableObj,
              public std::enable_shared_from_this<OpDef> {
    mutable const OpTrait* m_trait = nullptr;
    std::string m_scope;
public:
    virtual ~OpDef() = default;

    static std::shared_ptr<OpDef> make_from_op_node(
        cg::OperatorNodeBase* node);

    /*!
     * \brief Decide which dispatch method to be used according to the inputs'
     * host value and size.
     *
     * \param def Specific :c:expr:`OpDef` to be executed.
     * \param inputs Input tensor descriptions.
     * \return Which DispatchMode to be used, such as `CUDA` or `DEFAULT_CPU`.
     */
    static DispatchMode decide_dispatch_mode(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs);

    static SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def,
        SmallVector<TensorPtr> inputs);

    /*!
     * \brief Call the corresponding dnn op to calculate results. Output
     * tensors' device memory should be allocated outside.
     */
    static void apply_on_device_tensornd(
        const OpDef& def,
        const SmallVector<DeviceTensorND>& inputs,
        SmallVector<DeviceTensorND>* outputs);

    static cg::VarNodeArray apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs);

    static std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs);

    static BackwardGraphResult make_backward_graph(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

    static std::vector<std::pair<const char*, std::string>> props(
        const OpDef& def);

    const OpTrait* trait() const;

    std::string to_string() const;

    const std::string scope() const;

    const std::string make_name() const;

    void set_scope(const std::string& scope);

    virtual size_t hash() const;

    virtual bool is_same_st(const Hashable&) const;
};

template<typename T>
class OpDefImplBase : public OpDef {
public:
    template<typename ...Args>
    static std::shared_ptr<T> make(Args&& ...args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
};

template <>
struct ToStringTrait<OpDef*>{
    std::string operator()(OpDef* op) const {
        if (op == nullptr) {
            return "nullptr";
        }
        return op->to_string();
    }
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
