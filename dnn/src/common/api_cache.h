/**
 * \file dnn/src/common/api_cache.h
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

#include <cstring>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "megdnn/thin/function.h"

namespace megdnn {
template <typename... TArgs>
class FunctionCache {
public:
    using key_t = std::string;
    using value_t = std::string;
    using key_mapper_t = thin_function<key_t(TArgs...)>;
    using value_mapper_t = thin_function<value_t(TArgs...)>;
    using storage_t = std::unordered_map<key_t, value_t>;

    storage_t storage;
    key_mapper_t key_mapper;
    value_mapper_t value_mapper;

    value_t operator()(TArgs... args) {
        key_t key = key_mapper(args...);
        if (storage.count(key) == 0) {
            storage[key] = value_mapper(std::forward<TArgs>(args)...);
        }
        return storage[key];
    }
};

// FIFO
class StringSerializer {
private:
    std::string m_buffer;
    size_t m_cursor = 0;

public:
    template <typename T>
    T read_plain() {
    static_assert(std::is_trivially_copyable<T>::value, "invalid type");
    T ret;
        memcpy(&ret, m_buffer.data() + m_cursor, sizeof(T));
        m_cursor += sizeof(T);
        return ret;
    }
    template <typename T>
    void write_plain(T value) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "type should be trivially copyable");
        m_buffer.append(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    std::string take() {
        return std::move(m_buffer);
    }
    void set(std::string new_buf) {
        m_cursor = 0;
        m_buffer = new_buf;
    }
};

struct Empty {};

template <typename... TParams>
class ParamBundle {
private:
    template <std::size_t N, std::size_t... Seq>
    static std::index_sequence<N + Seq...> add_all(
            std::index_sequence<Seq...>) {
        return {};
    }

    template <std::size_t Min, std::size_t Max>
    using make_index_range =
            decltype(add_all<Min>(std::make_index_sequence<Max - Min>()));

    using storage_t = std::tuple<typename std::remove_reference_t<TParams>...>;
    storage_t m_storage;

    template <typename TFunctor, size_t... Indices>
    auto call_helper(TFunctor functor, std::index_sequence<Indices...>) {
        return functor(std::get<Indices>(m_storage).value...);
    }
    template <size_t Index, size_t... Indices, typename TPrev>
    auto serialize_helper(StringSerializer& ser, TPrev&& prev,
                          std::index_sequence<Index, Indices...>) {
        return serialize_helper(ser,
                                std::get<Index>(m_storage).serialize(ser, prev),
                                std::index_sequence<Indices...>());
    }
    template <typename TPrev>
    auto serialize_helper(StringSerializer& ser, TPrev&& prev,
                          std::index_sequence<>) {}
    template <size_t Index, size_t... Indices, typename TPrev>
    auto deserialize_helper(StringSerializer& ser, TPrev&& prev,
                            std::index_sequence<Index, Indices...>) {
        return deserialize_helper(
                ser, std::get<Index>(m_storage).deserialize(ser, prev),
                std::index_sequence<Indices...>());
    }
    template <typename TPrev>
    auto deserialize_helper(StringSerializer& ser, TPrev&& prev,
                            std::index_sequence<>) {}
    template <size_t Index, size_t... Indices, typename TArg, typename... TArgs>
    void set_values_helper(std::index_sequence<Index, Indices...>, TArg&& arg,
                           TArgs&&... args) {
        std::get<Index>(m_storage).value = arg;
        set_values_helper(std::index_sequence<Indices...>(),
                          std::forward<TArgs>(args)...);
    }
    template <size_t... Indices>
    void set_values_helper(std::index_sequence<Indices...>) {
        static_assert(sizeof...(Indices) == 0, "redundant indices");
    }

public:
    template <typename TFunctor>
    auto call_by(TFunctor&& functor) {
        return call_helper(std::forward<TFunctor>(functor),
                           std::make_index_sequence<sizeof...(TParams)>());
    }
    template <size_t NBegin, size_t NEnd>
    void serialize_params(StringSerializer& ser) {
        static_assert(NEnd >= NBegin, "invalid range");
        serialize_helper(
                ser, Empty{},
                add_all<NBegin>(std::make_index_sequence<NEnd - NBegin>()));
    }
    template <size_t NBegin, size_t NEnd>
    void deserialize_params(StringSerializer& ser) {
        static_assert(NEnd >= NBegin, "invalid range");
        deserialize_helper(
                ser, Empty{},
                add_all<NBegin>(std::make_index_sequence<NEnd - NBegin>()));
    }
    template <size_t NBegin, size_t NEnd, typename... TArgs>
    void set_values(TArgs&&... args) {
        set_values_helper(
                add_all<NBegin>(std::make_index_sequence<NEnd - NBegin>()),
                std::forward<TArgs>(args)...);
    }
};

template <typename T>
class Param {
public:
    T value;
    Empty serialize(StringSerializer& ser, Empty) {
        ser.write_plain(value);
        return Empty{};
    }
    Empty deserialize(StringSerializer& ser, Empty) {
        value = ser.read_plain<T>();
        return Empty{};
    }
};

template <typename TRet = Param<Empty>, typename TInputs = std::tuple<>,
          typename TOutputs = std::tuple<>>
class FunctionCacheBuilder {
private:
    static auto declargs()
            -> decltype(std::tuple_cat(std::declval<TInputs>(),
                                       std::declval<TOutputs>())) {
        return {};
    }
    template <size_t... Indices>
    static auto declfunction_helper(std::index_sequence<Indices...>)
            -> thin_function<decltype(std::declval<TRet>().value)(
                    decltype(std::get<Indices>(declargs()).value)...)> {
        return {};
    }
    static auto declfunction() {
        return declfunction_helper(
                std::make_index_sequence<std::tuple_size<TInputs>::value +
                                         std::tuple_size<TOutputs>::value>());
    }
    template <size_t... Indices>
    static auto declbundle_helper(std::index_sequence<Indices...>)
            -> ParamBundle<decltype(std::get<Indices>(declargs()))...> {
        return {};
    }
    static auto declbundle() {
        return declbundle_helper(
                std::make_index_sequence<std::tuple_size<TInputs>::value +
                                         std::tuple_size<TOutputs>::value>());
    }
    using function_t = decltype(declfunction());
    using bundle_t = decltype(declbundle());

public:
    template <typename TNewRet>
    auto ret() {
        static_assert(std::is_same<TRet, Param<Empty>>::value,
                      "return value redefinition");
        return FunctionCacheBuilder<TNewRet, TInputs, TOutputs>{};
    }
    template <typename TNewInput>
    auto input() {
        using TNewInputs = decltype(
                std::tuple_cat(std::declval<TInputs>(),
                               std::make_tuple(std::declval<TNewInput>())));
        return FunctionCacheBuilder<TRet, TNewInputs, TOutputs>{};
    }
    template <typename TNewOutput>
    auto output() {
        using TNewOutputs = decltype(
                std::tuple_cat(std::declval<TOutputs>(),
                               std::make_tuple(std::declval<TNewOutput>())));
        return FunctionCacheBuilder<TRet, TInputs, TNewOutputs>{};
    }
    template <typename TFunctor>
    function_t build(TFunctor func) {
        FunctionCache<bundle_t> cache;
        cache.key_mapper = [](bundle_t bundle) {
            StringSerializer ser;
            bundle.template serialize_params<0,
                                             std::tuple_size<TInputs>::value>(
                    ser);
            return ser.take();
        };
        cache.value_mapper = [=](bundle_t bundle) {
            StringSerializer ser;
            TRet ret;
            ret.value = bundle.call_by(func);
            ret.serialize(ser, Empty{});
            bundle.template serialize_params<
                    std::tuple_size<TInputs>::value,
                    std::tuple_size<TInputs>::value +
                            std::tuple_size<TOutputs>::value>(ser);
            return ser.take();
        };
        return [=](auto&&... args) mutable {
            bundle_t bundle;
            TRet ret;
            StringSerializer ser;
            static_assert(
                    sizeof...(args) == std::tuple_size<TInputs>::value +
                                               std::tuple_size<TOutputs>::value,
                    "args count mismatch");
            bundle.template set_values<0, sizeof...(args)>(
                    std::forward<decltype(args)>(args)...);
            ser.set(cache(bundle));
            ret.deserialize(ser, Empty{});
            constexpr size_t n_inputs = std::tuple_size<TInputs>::value;
            constexpr size_t n_outputs = std::tuple_size<TOutputs>::value;
            bundle.template deserialize_params<n_inputs, n_inputs + n_outputs>(
                    ser);
            return ret.value;
        };
    }
};

template <typename T>
class RefParam {
public:
    T* value;
    Empty serialize(StringSerializer& ser, Empty) {
        ser.write_plain(*value);
        return Empty{};
    }
    Empty deserialize(StringSerializer& ser, Empty) {
        *value = ser.read_plain<T>();
        return Empty{};
    }
};

template <typename T>
class RefArraySizeParam {
public:
    T* value;
    T serialize(StringSerializer& ser, Empty) {
        ser.write_plain(*value);
        return *value;
    }
    T deserialize(StringSerializer& ser, Empty) {
        return *value = ser.read_plain<T>();
    }
};

template <typename TSize, typename TItem>
class ArrayParam {
public:
    TItem* value;
    Empty serialize(StringSerializer& ser, TSize size) {
        for (TSize i = 0; i < size; ++i) {
            ser.write_plain(value[i]);
        }
        return Empty{};
    }
    Empty deserialize(StringSerializer& ser, TSize size) {
        for (TSize i = 0; i < size; ++i) {
            value[i] = ser.read_plain<TItem>();
        }
        return Empty{};
    }
};

}  // namespace megdnn
