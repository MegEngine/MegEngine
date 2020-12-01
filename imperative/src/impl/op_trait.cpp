/**
 * \file imperative/src/impl/op_trait.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <sstream>

#include "megbrain/imperative/ops/opr_attr.h"

#include "./op_trait.h"
#include "megbrain/imperative/proxy_graph_detail.h"

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
        if (!trait->apply_on_physical_tensor) {
            trait->apply_on_physical_tensor =
                    proxy_graph_detail::apply_on_physical_tensor;
        }
        if (!trait->infer_output_attrs_fallible) {
            trait->infer_output_attrs_fallible =
                    proxy_graph_detail::infer_output_attrs_fallible;
        }
        if (!trait->make_backward_graph) {
            trait->make_backward_graph =
                    proxy_graph_detail::make_backward_graph;
        }
    }
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
