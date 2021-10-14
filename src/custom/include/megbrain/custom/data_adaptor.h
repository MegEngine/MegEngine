/**
 * \file src/custom/include/megbrain/custom/data_adaptor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/thin/small_vector.h"

namespace custom {

template <typename BuiltinT, typename CustomT>
BuiltinT to_builtin(const CustomT& custom) {
    return *reinterpret_cast<const BuiltinT*>(custom.impl());
}

template <typename BuiltinT, typename CustomT>
CustomT to_custom(const BuiltinT& builtin) {
    return std::move(CustomT(&builtin));
}

template <typename BuiltinT, typename CustomT>
megdnn::SmallVector<BuiltinT> to_builtin(const std::vector<CustomT>& customs) {
    megdnn::SmallVector<BuiltinT> builtins;
    for (size_t i = 0; i < customs.size(); ++i) {
        builtins.push_back(std::move(to_builtin<BuiltinT, CustomT>(customs[i])));
    }
    return std::move(builtins);
}

template <typename BuiltinT, typename CustomT>
std::vector<CustomT> to_custom(const megdnn::SmallVector<BuiltinT>& builtins) {
    std::vector<CustomT> customs;
    for (size_t i = 0; i < builtins.size(); ++i) {
        customs.push_back(std::move(to_custom<BuiltinT, CustomT>(builtins[i])));
    }
    return std::move(customs);
}

}  // namespace custom

#define to_custom_device(expr)  custom::to_custom<CompNode, custom::Device>(expr)
#define to_builtin_device(expr) custom::to_builtin<CompNode, custom::Device>(expr)
#define to_custom_shape(expr) \
    custom::to_custom<megdnn::TensorShape, custom::Shape>(expr)
#define to_builtin_shape(expr) \
    custom::to_builtin<megdnn::TensorShape, custom::Shape>(expr)
#define to_custom_dtype(expr)  custom::to_custom<megdnn::DType, custom::DType>(expr)
#define to_builtin_dtype(expr) custom::to_builtin<megdnn::DType, custom::DType>(expr)
#define to_custom_format(expr) \
    custom::to_custom<megdnn::TensorLayout::Format, custom::Format>(expr)
#define to_builtin_format(expr) \
    custom::to_builtin<megdnn::TensorLayout::Format, custom::Format>(expr)
#define to_custom_tensor(expr)  custom::to_custom<DeviceTensorND, custom::Tensor>(expr)
#define to_builtin_tensor(expr) custom::to_builtin<DeviceTensorND, custom::Tensor>(expr)
