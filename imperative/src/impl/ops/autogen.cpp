/**
 * \file imperative/src/impl/ops/autogen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"

#include "../op_trait.h"

using namespace megdnn;

// FIXME: remove this when mgb::hash support tuple_hash
namespace mgb {
namespace {

struct HashWrapper {
    size_t hash;
    constexpr operator size_t() {return hash;}

    constexpr HashWrapper operator+(HashWrapper rhs) {
        // NOTE: use a + b + c + d, not a + (b + (c + d)) !!!
        return {hash * 20141203 + rhs.hash};
    }
};

template <typename... Args>
constexpr size_t hash_many(const Args&... args) {
    return (... + HashWrapper{mgb::hash(args)});
}

} // anonymous namespace

template<typename T, typename ...Args>
struct HashTrait<std::tuple<T, Args...>> {
    static size_t eval(const std::tuple<T, Args...> &t) {
        return std::apply(hash_many<T, Args...>, t);
    }
};
} // namespace mgb

namespace mgb::imperative {

#include "./opdef.cpp.inl"

} // namespace mgb::imperative
