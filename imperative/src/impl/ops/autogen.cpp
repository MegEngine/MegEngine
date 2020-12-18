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
template<typename T, size_t ...Ns>
auto tail(T t, std::index_sequence<Ns...>) {
    return std::make_tuple(std::get<Ns+1>(t)...);
}
} // anonymous namespace
template<typename T, typename ...Args>
class HashTrait<std::tuple<T, Args...>> {
    constexpr static size_t length = sizeof...(Args);
public:
    static size_t eval(const std::tuple<T, Args...> &t) {
        const T& val = std::get<0>(t);
        if constexpr (!length) {
            return mgb::hash(val);
        } else {
            return mgb::hash_pair_combine(mgb::hash(val),
                mgb::hash(tail(t, std::make_index_sequence<length - 1>{})));
        }
    }
};
} // namespace mgb

namespace mgb::imperative {

#include "./opdef.cpp.inl"

} // namespace mgb::imperative