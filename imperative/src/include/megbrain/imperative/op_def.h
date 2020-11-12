/**
 * \file imperative/src/include/megbrain/imperative/op_def.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/imperative/physical_tensor.h"

namespace mgb {
namespace imperative {

class OpDef;
struct OpTrait;

struct BackwardGraphResult {
    std::shared_ptr<OpDef> backward;
    std::vector<bool> save_for_backward;
    std::vector<bool> input_has_grad;
};

class OpDef : public Hashable {
    mutable const OpTrait* m_trait = nullptr;
public:
    virtual ~OpDef() = default;

    virtual std::shared_ptr<OpDef> copy() const = 0;

    static std::shared_ptr<OpDef> make_from_op_node(
        cg::OperatorNodeBase* node);

    static SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs);

    static cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs);

    static SmallVector<LogicalTensorDesc> infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs);

    static BackwardGraphResult make_backward_graph(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

    const OpTrait* trait() const;

    virtual size_t hash() const {
        mgb_throw(MegBrainError, "not implemented");
    }

    virtual bool is_same_st(const Hashable&) const {
        mgb_throw(MegBrainError, "not implemented");
    }
};

template<typename T>
class OpDefImplBase : public OpDef {
public:
    virtual std::shared_ptr<OpDef> copy() const override {
        return std::shared_ptr<OpDef>(new T(this->cast_final_safe<T>()));
    }

    template<typename ...Args>
    static std::shared_ptr<OpDef> make(const Args& ...args) {
        return std::shared_ptr<OpDef>(new T(args...));
    }
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
