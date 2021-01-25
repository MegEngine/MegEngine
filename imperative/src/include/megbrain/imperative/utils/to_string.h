/**
 * \file imperative/src/include/megbrain/imperative/utils/to_string.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <string>
#include <type_traits>
#include <memory>
#include <tuple>

#include "megbrain/utils/small_vector.h"
#include "megbrain/tensor.h"

namespace mgb::imperative {

template <typename T>
struct ToStringTrait;

template <typename T>
std::string to_string(const T& value) {
    return ToStringTrait<T>{}(value);
}

template <typename T>
struct ToStringTrait{
    std::string operator()(const T& value) const {
        return std::to_string(value);
    }
};

template <>
struct ToStringTrait<std::string>{
    std::string operator()(const std::string& value) const {
        return value;
    }
};

template <typename T, unsigned N>
struct ToStringTrait<SmallVector<T, N>>{
    std::string operator()(const SmallVector<T, N>& sv) const {
        if (sv.empty()) {
            return "[]";
        }
        std::string result = "[";
        result += to_string(sv[0]);
        for (size_t i = 1; i < sv.size(); ++i) {
            result += ", ";
            result += to_string(sv[i]);
        }
        return result + "]";
    }
};

template <typename T>
struct ToStringTrait<std::shared_ptr<T>>{
    std::string operator()(const std::shared_ptr<T>& sp) const {
        return to_string(sp.get());
    }
};

template <typename TKey, typename TValue>
struct ToStringTrait<std::pair<TKey, TValue>>{
    std::string operator()(const std::pair<TKey, TValue>& pr) const {
        return "(" + to_string(pr.first) + ", " + to_string(pr.second) + ")";
    }
};

template <typename TItem, typename... TItems>
struct ToStringTrait<std::tuple<TItem, TItems...>>{
    std::string operator()(const std::tuple<TItem, TItems...>& tp) const {
        auto folder = [&](auto... item){ return ( ...+ ("," + to_string(item))); };
        return "(" + std::apply(folder, tp) + ")";
    }
};

template <typename T>
struct ToStringTrait<T*>{
    std::string operator()(T* p) const {
        return ssprintf("%p", p);
    }
};

template <>
struct ToStringTrait<TensorShape>{
    std::string operator()(TensorShape shape) const {
        if (shape.ndim > TensorShape::MAX_NDIM) {
            printf("ndim: %d\n", (int)shape.ndim);
            return "[]";
        }
        mgb_assert(shape.ndim <= TensorShape::MAX_NDIM);
        if (shape.ndim == 0) {
            return "[ ]";
        }
        std::string result = "[ " + std::to_string(shape[0]);
        for (size_t i = 1; i < shape.ndim; i++) {
            result += ", ";
            result += std::to_string(shape[i]);
        }
        return result + " ]";
    }
};

template <>
struct ToStringTrait<DType>{
    std::string operator()(DType dtype) const {
        return dtype.name();
    }
};

template <>
struct ToStringTrait<CompNode>{
    std::string operator()(CompNode device) const {
        return device.to_string();
    }
};

}
