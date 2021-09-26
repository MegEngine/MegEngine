/**
 * \file imperative/src/include/megbrain/imperative/graph_builder.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/subgraph.h"

namespace mgb {
namespace imperative {

// NOTE: only input dtype and comp_node used for hashing, shapes are ignored
template <typename... TExtraArgs>
struct OpMethArgs {
    std::shared_ptr<OpDef> op;
    SmallVector<LogicalTensorDesc> inputs;
    std::tuple<TExtraArgs...> extras;

    size_t hash() const;
    bool operator==(const OpMethArgs& rhs) const {
        if (bool(op) ^ bool(rhs.op)) {
            return false;
        }
        if (op && rhs.op && !op->is_same(*rhs.op)) {
            return false;
        }
        if (inputs.size() != rhs.inputs.size()) {
            return false;
        }
        size_t nr_inputs = inputs.size();
        for (size_t i = 0; i < nr_inputs; ++i) {
            if (inputs[i].comp_node != rhs.inputs[i].comp_node) {
                return false;
            }
            if (inputs[i].layout.dtype != rhs.inputs[i].layout.dtype) {
                return false;
            }
            if (inputs[i].layout.ndim != rhs.inputs[i].layout.ndim) {
                return false;
            }
            if (inputs[i].value.empty() != rhs.inputs[i].value.empty()) {
                return false;
            }
        }
        return extras == rhs.extras;
    }

    struct hash_t {
        size_t operator()(const OpMethArgs& key) const { return key.hash(); }
    };
};

template <typename... TExtraArgs>
inline size_t OpMethArgs<TExtraArgs...>::hash() const {
    XXHash state;
    size_t length = 0;
    size_t data[1 + 4 * inputs.size() + sizeof...(TExtraArgs)];
    auto append = [&](size_t hash) { data[length++] = hash; };
    append(op->hash());
    for (auto&& i : inputs) {
        append(mgb::hash(i.layout.dtype.handle()));
        append(mgb::hash(i.comp_node));
        append(mgb::hash(i.layout.ndim));
        append(mgb::hash(i.value.empty()));
    }
    std::apply([&](auto&&... extras) { (append(mgb::hash(extras)), ...); }, extras);
    mgb_assert(length == sizeof(data) / sizeof(size_t));
    state.update(data, sizeof(data));
    return state.digest();
}

template <typename TValue, typename... TExtraArgs>
struct OpMethResultCache : std::unordered_map<
                                   OpMethArgs<TExtraArgs...>, TValue,
                                   typename OpMethArgs<TExtraArgs...>::hash_t>,
                           CompNodeDepedentObject {
    std::shared_ptr<void> on_comp_node_finalize() override {
        static_cast<std::unordered_map<
                OpMethArgs<TExtraArgs...>, TValue,
                typename OpMethArgs<TExtraArgs...>::hash_t>*>(this)
                ->clear();
        // clear();
        return {};
    }

    using key_t = OpMethArgs<TExtraArgs...>;
};

}  // namespace imperative
}  // namespace mgb
