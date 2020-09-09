/**
 * \file imperative/src/include/megbrain/imperative/ops/rng.h
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

namespace mgb::imperative {

class RNGMixin {
public:
    using Handle = size_t;

    static Handle new_handle(
        CompNode comp_node={}, uint64_t seed=0);

    static size_t delete_handle(Handle handle);

    Handle handle() const {
        return m_handle;
    }

    uint64_t seed() const;

    CompNode comp_node() const;
protected:
    RNGMixin(Handle handle): m_handle(handle) {}
    RNGMixin(CompNode comp_node);
private:
    Handle m_handle;
};

class GaussianRNG : public OpDefImplBase<GaussianRNG>,
                    public RNGMixin {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    float mean = 1.0f, std = 0.0;
    GaussianRNG(CompNode comp_node_): RNGMixin(comp_node_) {}
    GaussianRNG(float mean_=1.0, float std_=0.0, CompNode comp_node_={}):
        GaussianRNG(comp_node_) { mean = mean_; std = std_; }
    GaussianRNG(float mean_, float std_, Handle handle):
        RNGMixin(handle), mean(mean_), std(std_) {}
    size_t hash() const override {
        XXHash xxhash{};
        auto append = [&xxhash](auto field){
            auto hash_val = HashTrait<decltype(field)>::eval(field);
            xxhash.update(reinterpret_cast<void*>(&hash_val), sizeof(hash_val));
        };
        append(dyn_typeinfo());
        append(seed());
        append(mean);
        append(std);
        return xxhash.digest();
    }


    bool is_same_st(const Hashable& rhs_) const override {
        auto&& rhs = static_cast<const GaussianRNG&>(rhs_);
        return rhs.seed() == seed()
            && rhs.mean == mean
            && rhs.std == std;
    }
};

class UniformRNG : public OpDefImplBase<UniformRNG>,
                   public RNGMixin {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    UniformRNG(CompNode comp_node_={}): RNGMixin(comp_node_) {}
    UniformRNG(Handle handle): RNGMixin(handle) {}

    size_t hash() const override {
        return hash_pair_combine(
                mgb::hash(seed()),
                reinterpret_cast<std::uintptr_t>(dyn_typeinfo()));
    }

    bool is_same_st(const Hashable& rhs_) const override {
        auto&& rhs = static_cast<const UniformRNG&>(rhs_);
        return rhs.dyn_typeinfo() == dyn_typeinfo()
            && rhs.seed() == seed();
    }

};

void set_rng_seed(uint64_t seed);
} // namespace mgb::imperative
