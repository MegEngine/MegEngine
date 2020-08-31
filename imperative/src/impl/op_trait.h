/**
 * \file imperative/src/impl/op_trait.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

using OpDefMaker = thin_function<
        decltype(OpDef::make_from_op_node)>;
using ApplyOnPhysicalTensor = thin_function<
        decltype(OpDef::apply_on_physical_tensor)>;
using PhysicalTensorExecutor = thin_function<
        decltype(OpDef::exec)>;
using ApplyOnVarNode = thin_function<
        decltype(OpDef::apply_on_var_node)>;
using InferOutputAttrsFallible = thin_function<
        decltype(OpDef::infer_output_attrs_fallible)>;
using InferOutputAttrs = thin_function<
        decltype(OpDef::infer_output_attrs)>;
using GradMaker = thin_function<
        decltype(OpDef::make_backward_graph)>;

struct OpTrait {
    const char* name;
    OpDefMaker make_from_op_node;
    ApplyOnPhysicalTensor apply_on_physical_tensor;
    PhysicalTensorExecutor exec;
    ApplyOnVarNode apply_on_var_node;
    InferOutputAttrsFallible infer_output_attrs_fallible;
    InferOutputAttrs infer_output_attrs;
    GradMaker make_backward_graph;
    OpTrait(const char* name);
    static OpTrait* find_by_name(const char* name);
    static OpTrait* find_by_typeinfo(Typeinfo* type);
    static void for_each_trait(thin_function<void(OpTrait&)> visitor);
};

struct OpTraitRegistry {
    OpTrait* trait;
    OpTraitRegistry& make_from_op_node(OpDefMaker f) {
        trait->make_from_op_node = f;
        return *this;
    }
    OpTraitRegistry& apply_on_physical_tensor(ApplyOnPhysicalTensor f) {
        trait->apply_on_physical_tensor = f;
        return *this;
    }
    OpTraitRegistry& physical_tensor_executor(PhysicalTensorExecutor f) {
        trait->exec = f;
        return *this;
    }
    OpTraitRegistry& apply_on_var_node(ApplyOnVarNode f) {
        trait->apply_on_var_node = f;
        return *this;
    }
    OpTraitRegistry& infer_output_attrs_fallible(InferOutputAttrsFallible f) {
        trait->infer_output_attrs_fallible = f;
        return *this;
    }
    OpTraitRegistry& infer_output_attrs(InferOutputAttrs f) {
        trait->infer_output_attrs = f;
        return *this;
    }
    OpTraitRegistry& grad_maker(GradMaker f) {
        trait->make_backward_graph = f;
        return *this;
    }
    OpTraitRegistry& fallback();
    OpTraitRegistry& finalize();

    template<typename T>
    void insert() {
        do_insert(T::typeinfo());
    }

    template<typename T0, typename T1, typename ...Ts>
    void insert() {
        insert<T0>();
        insert<T1, Ts...>();
    }

    template<typename ...Args>
    static OpTraitRegistry insert(const char* name) {
        auto&& ret = do_insert(name);
        ret.insert<Args...>();
        return ret;
    }

    void do_insert(Typeinfo* type);

    static OpTraitRegistry do_insert(const char* name);
};

namespace detail {
struct _RegisterHelper {
    OpTraitRegistry registry;
    ~_RegisterHelper() {
        registry.finalize();
    }
};
} // namespace detail

} // namespace imperative
} // namespace mgb

#define OP_TRAIT_REG(name, ...) \
    static OpTraitRegistry __##name##_global_registry__ = \
        detail::_RegisterHelper{OpTraitRegistry::insert<__VA_ARGS__>(#name)}.registry

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
