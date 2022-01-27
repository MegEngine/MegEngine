/**
 * \file imperative/src/include/megbrain/imperative/dispatch.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <list>
#include <map>
#include <memory>
#include <typeinfo>
#include <vector>

#include "megbrain/common.h"
#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/basic_values.h"
#include "megbrain/imperative/operator.h"
#include "megbrain/imperative/subgraph.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/utils/local_ptr.h"
#include "megbrain/imperative/utils/span.h"
#include "megbrain/imperative/value.h"

namespace mgb {
namespace imperative {

/**
 * \brief dispatch entrance, requests would be forwarded to current top transformation
 * (or fallback)
 *
 * \param op
 * \param inputs
 * \return ValueRefList
 */
ValueRefList apply(const Operator& op, Span<ValueRef> inputs);
ValueRefList apply(const OpDef& def, Span<ValueRef> inputs);
ValueRefList apply(const Subgraph& graph, Span<ValueRef> inputs);

template <typename... TArgs>
constexpr bool is_all_value_ref_v =
        (... && (std::is_base_of_v<ValueRef, std::decay_t<TArgs>> ||
                 std::is_same_v<ValueRef, std::decay_t<TArgs>>));

template <typename T>
static ValueRefList apply(T&& op, const ValueRef& arg) {
    return imperative::apply(std::forward<T&&>(op), Span<ValueRef>{&arg, 1});
}

template <typename T, typename... TArgs>
static auto apply(T&& op, TArgs&&... args) -> std::enable_if_t<
        is_all_value_ref_v<TArgs...> && sizeof...(args) != 1, ValueRefList> {
    ValueRef args_arr[sizeof...(TArgs)] = {std::forward<TArgs&&>(args)...};
    return imperative::apply(
            std::forward<T&&>(op),
            Span<ValueRef>(std::begin(args_arr), std::end(args_arr)));
}

template <typename T, typename TContainer>
static auto apply(T&& op, TContainer&& container) -> std::enable_if_t<
        std::is_same_v<
                std::remove_const_t<std::remove_pointer_t<decltype(container.data())>>,
                ValueRef> &&
                std::is_same_v<decltype(container.size()), size_t> &&
                !std::is_same_v<std::decay_t<TContainer>, Span<ValueRef>>,
        ValueRefList> {
    return imperative::apply(
            std::forward<T&&>(op), Span<ValueRef>(container.data(), container.size()));
}

}  // namespace imperative
}  // namespace mgb
