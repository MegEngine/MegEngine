/**
 * \file imperative/src/include/megbrain/imperative/ops/utility.h
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

#include "megbrain/utils/hash.h"

#include <pybind11/pybind11.h>

namespace mgb::imperative {

struct GenericPyOp final : OpDefImplBase<GenericPyOp> {
    pybind11::object obj;

    GenericPyOp(pybind11::object obj_) : obj(std::move(obj_)){};

    size_t hash() const override { return pybind11::hash(obj); }

    bool is_same_st(const Hashable& rhs) const override {
        return obj.equal(rhs.cast_final<GenericPyOp>().obj);
    }

    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

struct ShapeInfer final : OpDefImplBase<ShapeInfer> {
    std::shared_ptr<OpDef> op;
    SmallVector<CompNode> devices;
    SmallVector<DType> dtypes;
    EncodedSubgraph graph;
    ShapeInfer() = default;
    ShapeInfer(
            std::shared_ptr<OpDef> op, SmallVector<CompNode> devices,
            SmallVector<DType> dtypes)
            : op{op}, devices{devices}, dtypes{dtypes} {}
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

struct UniqueKey final : Hashable {
public:
    size_t hash() const override { return reinterpret_cast<uintptr_t>(this); }

protected:
    bool is_same_st(const Hashable& rhs) const override {
        return this == &rhs.cast_final_safe<UniqueKey>();
    }
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

struct SubgraphOp final : OpDefImplBase<SubgraphOp> {
    std::string name;
    std::shared_ptr<Subgraph> graph;
    SmallVector<bool> output_grad_mask;
    std::shared_ptr<Hashable> graph_key;
    SubgraphOp() = default;
    SubgraphOp(
            std::string name, std::shared_ptr<Subgraph> graph,
            SmallVector<bool> output_grad_mask = {},
            std::shared_ptr<Hashable> key = nullptr)
            : name{name},
              graph{graph},
              output_grad_mask{output_grad_mask},
              graph_key{std::move(key)} {
        if (this->output_grad_mask.empty()) {
            this->output_grad_mask.resize(graph->outputs.size(), true);
        }
    }
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

struct BackwardOpKey final : Hashable,
                             OpMethArgs<SmallVector<bool>, SmallVector<bool>> {
public:
    using OpMethArgs<SmallVector<bool>, SmallVector<bool>>::OpMethArgs;
    size_t hash() const override {
        return OpMethArgs<SmallVector<bool>, SmallVector<bool>>::hash();
    }

protected:
    bool is_same_st(const Hashable& rhs) const override {
        return OpMethArgs<SmallVector<bool>, SmallVector<bool>>::operator==(
                rhs.cast_final_safe<BackwardOpKey>());
    }
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

struct CompiledOp final : OpDefImplBase<CompiledOp> {
    std::shared_ptr<OpDef> op;
    int gopt_level;
    CompiledOp() = default;
    CompiledOp(std::shared_ptr<OpDef> op, int gopt_level = 2)
            : op{op}, gopt_level{gopt_level} {}
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

}  // namespace mgb::imperative
