/**
 * \file imperative/src/impl/op_trait.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <exception>
#include <sstream>
#include <stdexcept>

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/imperative/subgraph_detail.h"
#include "megbrain/tensor.h"

#include "./op_trait.h"

namespace mgb {
namespace imperative {

namespace detail {

struct StaticData {
    std::list<OpTrait> registries;
    std::unordered_map<std::string, OpTrait*> name2reg;
    std::unordered_map<Typeinfo*, OpTrait*> type2reg;
};

// use "Construct On First Use" to prevent "static initialization order fiasco"
// (i.e., ensure global registry was initialized before calling opr registration)
StaticData& static_data() {
    static StaticData data;
    return data;
}

void OpMethFallbackByProxyGraph::impl(
        ApplyOnPhysicalTensor& func, op_meth_tag::ApplyOnPhysicalTensor) {
    func.Base::operator=(proxy_graph_detail::apply_on_physical_tensor);
}
void OpMethFallbackByProxyGraph::impl(
        InferOutputAttrsFallible& func, op_meth_tag::InferOutputAttrsFallible) {
    func.Base::operator=(proxy_graph_detail::infer_output_attrs_fallible);
}
void OpMethFallbackByProxyGraph::impl(GradMaker& func, op_meth_tag::GradMaker) {
    func.Base::operator=(proxy_graph_detail::make_backward_graph);
}

void OpMethFallbackFromSubgraph::impl(
        ApplyOnPhysicalTensor& func, op_meth_tag::ApplyOnPhysicalTensor) {
    func.Base::operator=(subgraph_detail::apply_on_physical_tensor);
}
void OpMethFallbackFromSubgraph::impl(
        ApplyOnVarNode& func, op_meth_tag::ApplyOnVarNode) {
    func.Base::operator=(subgraph_detail::apply_on_var_node);
}
void OpMethFallbackFromSubgraph::impl(
        InferOutputAttrsFallible& func, op_meth_tag::InferOutputAttrsFallible) {
    func.Base::operator=(subgraph_detail::infer_output_attrs_fallible);
}
void OpMethFallbackFromSubgraph::impl(GradMaker& func, op_meth_tag::GradMaker) {
    func.Base::operator=(subgraph_detail::make_backward_graph);
}

void OpMethFallback::impl(DecideDispatchMode& func, op_meth_tag::DecideDispatchMode) {
    static auto decide_dispatch_mode = [](const OpDef&,
                                          const SmallVector<LogicalTensorDesc>&) {
        return DispatchMode::KERNEL;
    };
    func.Base::operator=(decide_dispatch_mode);
}
void OpMethFallback::impl(MakeNameFunc& func, op_meth_tag::MakeNameFunc) {
    static auto make_name = [](const OpDef& def) -> std::string {
        return def.trait()->name;
    };
    func.Base::operator=(make_name);
}
}  // namespace detail

OpTrait::OpTrait(const char* name_) : name(name_) {}

OpTrait* OpTrait::find_by_typeinfo(Typeinfo* type) {
    auto&& type2reg = detail::static_data().type2reg;
    auto iter = type2reg.find(type);
    if (iter == type2reg.end()) {
        return nullptr;
    }
    return iter->second;
}

OpTrait* OpTrait::find_by_name(const char* name) {
    auto&& name2reg = detail::static_data().name2reg;
    auto iter = name2reg.find(name);
    if (iter == name2reg.find(name)) {
        return nullptr;
    }
    return iter->second;
}

void OpTrait::for_each_trait(thin_function<void(OpTrait&)> visitor) {
    for (auto& trait : detail::static_data().registries) {
        visitor(trait);
    }
}

OpTraitRegistry& OpTraitRegistry::fallback() {
    using Mode = detail::OpMethFallbackMode;
    uint64_t mode = Mode::None;
    if (trait->make_forward_graph) {
        mode |= Mode::FromSubgraph;
    }
    if (trait->apply_on_var_node) {
        mode |= Mode::ByProxyGraph;
    }
    mode |= Mode::Default;
#define SET_FALLBACK_MODE(meth) trait->meth.fallback_mode = mode;
    FOR_EACH_OP_METH(SET_FALLBACK_MODE)
#undef SET_FALLBACK_MODE

    return *this;
}

void OpTraitRegistry::do_insert(Typeinfo* type) {
    auto&& sd = detail::static_data();
    auto ret = sd.type2reg.emplace(type, trait);
    mgb_assert(
            ret.second || ret.first->second == trait,
            "OpTrait for %s has already been registered", type->name);
}

OpTraitRegistry OpTraitRegistry::do_insert(const char* name) {
    auto&& sd = detail::static_data();
    if (name) {
        auto iter = sd.name2reg.find(name);
        if (iter != sd.name2reg.end()) {
            return {iter->second};
        }
    }
    sd.registries.emplace_back(name);
    auto ret = &sd.registries.back();
    if (name) {
        sd.name2reg.emplace(name, ret);
    }
    return {ret};
}

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
