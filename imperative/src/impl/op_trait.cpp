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

void OpMethFallback::impl(ApplyOnPhysicalTensor& func,
                          op_meth_tag::ApplyOnPhysicalTensor) {
    func.Base::operator=(proxy_graph_detail::apply_on_physical_tensor);
}
void OpMethFallback::impl(Execute& func, op_meth_tag::Execute) {
    func.Base::operator=(proxy_graph_detail::execute);
}
void OpMethFallback::impl(InferOutputMemDesc& func,
                          op_meth_tag::InferOutputMemDesc) {
    func.Base::operator=(proxy_graph_detail::infer_output_mem_desc);
}
void OpMethFallback::impl(InferOutputAttrsFallible& func,
                          op_meth_tag::InferOutputAttrsFallible) {
    func.Base::operator=(proxy_graph_detail::infer_output_attrs_fallible);
}
void OpMethFallback::impl(GradMaker& func, op_meth_tag::GradMaker) {
    func.Base::operator=(proxy_graph_detail::make_backward_graph);
}
void OpMethFallback::impl(DecideDispatchMode& func,
                          op_meth_tag::DecideDispatchMode) {
    static auto decide_dispatch_mode =
            [](const OpDef&, const SmallVector<LogicalTensorDesc>&) {
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
} // detail

OpTrait::OpTrait(const char* name_): name(name_) {}

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

void OpTrait::for_each_trait(thin_function<void(OpTrait&)> visitor){
    for(auto& trait: detail::static_data().registries){
        visitor(trait);
    }
}

OpTraitRegistry& OpTraitRegistry::fallback() {
    if (trait->apply_on_var_node) {
        // fallback to proxy graph impl
        trait->apply_on_physical_tensor.allow_fallback = true;
        trait->execute.allow_fallback = true;
        trait->infer_output_mem_desc.allow_fallback = true;
        trait->infer_output_attrs_fallible.allow_fallback = true;
        trait->make_backward_graph.allow_fallback = true;
    }
    trait->decide_dispatch_mode.allow_fallback = true;
    trait->make_name.allow_fallback = true;
    return *this;
}

void OpTraitRegistry::do_insert(Typeinfo* type) {
    auto&& sd = detail::static_data();
    auto ret = sd.type2reg.emplace(type, trait);
    mgb_assert(ret.second || ret.first->second == trait,
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

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
