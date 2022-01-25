/**
 * \file imperative/src/impl/op_trait.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/graph_cache.h"
#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {
namespace detail {
template <typename Tag, typename Signature>
struct OpMeth;

template <typename T>
struct ToVarNodeArray : std::false_type {};
template <>
struct ToVarNodeArray<SymbolVar> : std::true_type {
    VarNodeArray operator()(const SymbolVar& inp) { return {inp.node()}; }
};
template <>
struct ToVarNodeArray<SymbolVarArray> : std::true_type {
    VarNodeArray operator()(const SymbolVarArray& inputs) {
        return cg::to_var_node_array(inputs);
    }
};
template <size_t N>
struct ToVarNodeArray<std::array<SymbolVar, N>> : std::true_type {
    VarNodeArray operator()(const std::array<SymbolVar, N>& inp) {
        return cg::to_var_node_array({inp.begin(), inp.end()});
    }
};
template <>
struct ToVarNodeArray<cg::OperatorNodeBase*> : std::true_type {
    VarNodeArray operator()(const cg::OperatorNodeBase* opr) {
        return opr->usable_output();
    }
};
}  // namespace detail

// clang-format off
#define OpMethType(TYPE, SIG)                     \
    namespace detail::op_meth_tag {               \
        struct TYPE {                             \
            constexpr static char name[] = #TYPE; \
        };                                        \
    }                                             \
    using TYPE = detail::OpMeth<detail::op_meth_tag::TYPE, SIG>

OpMethType(OpDefMaker,
           decltype(OpDef::make_from_op_node));

OpMethType(DecideDispatchMode,
           decltype(OpDef::decide_dispatch_mode));

OpMethType(ApplyOnPhysicalTensor,
           decltype(OpDef::apply_on_physical_tensor));

OpMethType(ApplyOnDeviceTensorND,
           decltype(OpDef::apply_on_device_tensornd));

OpMethType(ApplyOnVarNode,
           decltype(OpDef::apply_on_var_node));

OpMethType(InferOutputAttrsFallible,
           decltype(OpDef::infer_output_attrs_fallible));

OpMethType(GradMaker,
           decltype(OpDef::make_backward_graph));

OpMethType(Props,
           decltype(OpDef::props));

OpMethType(HashFunc,
           size_t(const OpDef&));

OpMethType(IsSame,
           bool(const OpDef&, const OpDef&));

OpMethType(MakeNameFunc,
           std::string(const OpDef&));

OpMethType(GraphMaker,
           decltype(OpDef::make_forward_graph));
// clang-format on

namespace detail {

struct OpMethImplBase {
    template <typename Tag, typename RType, typename... Args>
    static void impl(thin_function<RType(Args...)>& func, Tag) {}
};

struct OpMethNotImpl {
    template <typename Tag, typename RType, typename... Args>
    static void impl(thin_function<RType(Args...)>& func, Tag) {
        func = [](Args... args) -> RType {
            mgb_throw(MegBrainError, "%s was not implemented yet", Tag::name);
        };
    }
};

struct OpMethFallback : OpMethImplBase {
    using OpMethImplBase::impl;
    static void impl(DecideDispatchMode& func, op_meth_tag::DecideDispatchMode);
    static void impl(MakeNameFunc& func, op_meth_tag::MakeNameFunc);
};

struct OpMethFallbackByProxyGraph : OpMethImplBase {
    using OpMethImplBase::impl;
    static void impl(ApplyOnPhysicalTensor& func, op_meth_tag::ApplyOnPhysicalTensor);
    static void impl(
            InferOutputAttrsFallible& func, op_meth_tag::InferOutputAttrsFallible);
    static void impl(GradMaker& func, op_meth_tag::GradMaker);
};

struct OpMethFallbackFromSubgraph : OpMethImplBase {
    using OpMethImplBase::impl;
    static void impl(ApplyOnPhysicalTensor& func, op_meth_tag::ApplyOnPhysicalTensor);
    static void impl(ApplyOnVarNode& func, op_meth_tag::ApplyOnVarNode);
    static void impl(
            InferOutputAttrsFallible& func, op_meth_tag::InferOutputAttrsFallible);
    static void impl(GradMaker& func, op_meth_tag::GradMaker);
};

struct OpMethFallbackMode {
    static constexpr uint64_t None = 0;
    static constexpr uint64_t Default = 1;
    static constexpr uint64_t ByProxyGraph = 2;
    static constexpr uint64_t FromSubgraph = 4;
};

template <typename Tag, typename RType, typename... Args>
struct OpMeth<Tag, RType(Args...)> : public thin_function<RType(Args...)> {
    using Base = thin_function<RType(Args...)>;
    OpMeth() : Base{} {};
    explicit OpMeth(const Base& base) { this->Base::operator=(base); }
    using Base::operator bool;
    RType operator()(Args... args) const {
        uint64_t mode_mask = ~uint64_t(0);
        auto match_mode = [&](uint64_t mode) {
            if ((fallback_mode & mode_mask) & mode) {
                mode_mask &= ~mode;
                return true;
            }
            return false;
        };
        while (!this->Base::operator bool()) {
            using Mode = OpMethFallbackMode;
            if (match_mode(Mode::FromSubgraph)) {
                OpMethFallbackFromSubgraph::impl(*const_cast<OpMeth*>(this), Tag{});
            } else if (match_mode(Mode::ByProxyGraph)) {
                OpMethFallbackByProxyGraph::impl(*const_cast<OpMeth*>(this), Tag{});
            } else if (match_mode(Mode::Default)) {
                OpMethFallback::impl(*const_cast<OpMeth*>(this), Tag{});
            } else {
                OpMethNotImpl::impl(*const_cast<OpMeth*>(this), Tag{});
            }
        }
        return this->Base::operator()(std::forward<Args>(args)...);
    }
    uint64_t fallback_mode = OpMethFallbackMode::None;
};
}  // namespace detail

struct OpTrait {
    const char* name;
    OpDefMaker make_from_op_node;
    DecideDispatchMode decide_dispatch_mode;
    ApplyOnPhysicalTensor apply_on_physical_tensor;
    ApplyOnDeviceTensorND apply_on_device_tensornd;
    ApplyOnVarNode apply_on_var_node;
    InferOutputAttrsFallible infer_output_attrs_fallible;
    GradMaker make_backward_graph;
    Props props;
    HashFunc hash;
    IsSame is_same_st;
    MakeNameFunc make_name;
    GraphMaker make_forward_graph;
    OpTrait(const char* name);
    static OpTrait* find_by_name(const char* name);
    static OpTrait* find_by_typeinfo(Typeinfo* type);
    static void for_each_trait(thin_function<void(OpTrait&)> visitor);
};

// clang-format off
#define FOR_EACH_OP_METH(cb)        \
    cb(make_from_op_node)           \
    cb(decide_dispatch_mode)        \
    cb(apply_on_physical_tensor)    \
    cb(apply_on_device_tensornd)    \
    cb(apply_on_var_node)           \
    cb(infer_output_attrs_fallible) \
    cb(make_backward_graph)         \
    cb(props)                       \
    cb(hash)                        \
    cb(is_same_st)                  \
    cb(make_name)                   \
    cb(make_forward_graph)

// clang-format on

struct OpTraitRegistry {
    OpTrait* trait;
#define DECL(meth)                                                                     \
    OpTraitRegistry& meth(decltype(OpTrait::meth)::Base f) {                           \
        mgb_assert(!trait->meth, "op %s has duplicate method %s", trait->name, #meth); \
        trait->meth.Base::operator=(f);                                                \
        return *this;                                                                  \
    }
    FOR_EACH_OP_METH(DECL)
#undef DECL

    OpTraitRegistry& fallback();

    template <typename T>
    void insert() {
        do_insert(T::typeinfo());
    }

    template <typename T0, typename T1, typename... Ts>
    void insert() {
        insert<T0>();
        insert<T1, Ts...>();
    }

    template <typename... Args>
    static OpTraitRegistry insert(const char* name) {
        auto&& ret = do_insert(name);
        ret.insert<Args...>();
        return ret;
    }

    void do_insert(Typeinfo* type);

    static OpTraitRegistry do_insert(const char* name);

    template <
            typename T, typename To = detail::ToVarNodeArray<T>,
            typename = std::enable_if_t<To::value>>
    OpTraitRegistry& apply_on_var_node(T (*f)(const OpDef&, const VarNodeArray&)) {
        return apply_on_var_node([=](const OpDef& opdef, const VarNodeArray& inputs) {
            return To()(f(opdef, inputs));
        });
    }
};

}  // namespace imperative
}  // namespace mgb

#define OP_TRAIT_REG(name, ...)                           \
    static OpTraitRegistry __##name##_global_registry__ = \
            OpTraitRegistry::insert<__VA_ARGS__>(#name)

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
