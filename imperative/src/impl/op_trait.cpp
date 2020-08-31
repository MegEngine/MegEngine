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
#include "./proxy_graph_detail.h"

namespace mgb {
namespace imperative {

namespace detail {

struct StaticData {
    std::list<OpTrait> registries;
    std::unordered_map<const char*, OpTrait*> name2reg;
    std::unordered_map<Typeinfo*, OpTrait*> type2reg;
};

// use "Construct On First Use" to prevent "static initialization order fiasco"
// (i.e., ensure global registry was initialized before calling opr registration)
StaticData& static_data() {
    static StaticData data;
    return data;
}

template<typename T>
struct __not_implementation__;

template<typename RType, typename ...Args>
struct __not_implementation__<RType(Args...)> {
    static RType raise(Args ...) {
        mgb_throw(MegBrainError, "Not Implemented");
    }
};

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

OpTraitRegistry& OpTraitRegistry::finalize() {
    std::ostringstream msg;
    #define CHECK(field) if (!trait->field) { \
        msg << ", " #field; \
        trait->field = \
            detail::__not_implementation__<decltype(OpDef::field)>::raise; \
    }
    CHECK(make_from_op_node);
    CHECK(apply_on_physical_tensor);
    CHECK(exec);
    CHECK(apply_on_var_node);
    CHECK(infer_output_attrs_fallible);
    CHECK(infer_output_attrs);
    CHECK(make_backward_graph);
    #undef CHECK
    if (msg.tellp() > 0) {
        mgb_log_warn(
            "%s op trait missing: %s",
            trait->name ? trait->name : "(anonymous)",
            msg.str().c_str() + 2 /* skip first ", " */);
    }
    return *this;
}

SmallVector<TensorPtr> fallback_apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto desc = OpDef::infer_output_attrs(def, inputs);
    SmallVector<TensorPtr> outputs;
    for (auto&& i : desc) {
        outputs.push_back(Tensor::make(i.layout, i.comp_node));
    }
    OpDef::exec(def, inputs, outputs);
    return outputs;
}

SmallVector<LogicalTensorDesc> fallback_infer_output_attrs(const OpDef& def,
        const SmallVector<TensorPtr>& inputs){
    SmallVector<LogicalTensorDesc> input_descs;
    for(auto&& input: inputs){
        input_descs.push_back({input->layout(), input->comp_node()});
    }
    return input_descs;
}

OpTraitRegistry& OpTraitRegistry::fallback() {
    if (!trait->exec && trait->apply_on_var_node) {
        trait->exec = proxy_graph_detail::exec;
    }
    if (!trait->infer_output_attrs && trait->apply_on_var_node) {
        trait->infer_output_attrs = proxy_graph_detail::infer_output_attrs;
    }
    if (!trait->infer_output_attrs_fallible && trait->apply_on_var_node) {
        trait->infer_output_attrs_fallible = proxy_graph_detail::infer_output_attrs_fallible;
    }
    if (!trait->make_backward_graph && trait->apply_on_var_node) {
        trait->make_backward_graph = proxy_graph_detail::make_backward_graph;
    }
    if (!trait->apply_on_physical_tensor && trait->infer_output_attrs && trait->exec) {
        trait->apply_on_physical_tensor = fallback_apply_on_physical_tensor;
    }
    if(!trait->infer_output_attrs && trait->infer_output_attrs_fallible){
        trait->infer_output_attrs = fallback_infer_output_attrs;
    }
    return *this;
}

void OpTraitRegistry::do_insert(Typeinfo* type) {
    auto&& sd = detail::static_data();
    mgb_assert(sd.type2reg.emplace(type, trait).second);
}

OpTraitRegistry OpTraitRegistry::do_insert(const char* name) {
    auto&& sd = detail::static_data();
    if (name) {
        mgb_assert(!sd.name2reg.count(name),
            "duplicated opr trait %s", name);
    }
    sd.registries.emplace_back(name);
    auto ret = &sd.registries.back();
    sd.name2reg.emplace(name, ret);
    return {ret};
}

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
