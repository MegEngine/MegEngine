#pragma once

#include "megbrain/custom/op.h"
#include "megbrain/custom/tensor.h"
#include "megbrain/tensor.h"
#include "megdnn/thin/small_vector.h"

namespace custom {

template <typename BuiltinT, typename CustomT>
BuiltinT to_builtin(const CustomT& custom) {
    return *reinterpret_cast<const BuiltinT*>(custom.impl());
}

template <typename BuiltinT, typename CustomT>
CustomT to_custom(const BuiltinT& builtin) {
    return CustomT(&builtin);
}

template <typename BuiltinT, typename CustomT>
megdnn::SmallVector<BuiltinT> to_builtin(const std::vector<CustomT>& customs) {
    megdnn::SmallVector<BuiltinT> builtins;
    for (size_t i = 0; i < customs.size(); ++i) {
        builtins.emplace_back(to_builtin<BuiltinT, CustomT>(customs[i]));
    }
    return builtins;
}

template <typename BuiltinT, typename CustomT>
std::vector<CustomT> to_custom(const megdnn::SmallVector<BuiltinT>& builtins) {
    std::vector<CustomT> customs;
    for (size_t i = 0; i < builtins.size(); ++i) {
        customs.emplace_back(to_custom<BuiltinT, CustomT>(builtins[i]));
    }
    return customs;
}

MGE_WIN_DECLSPEC_FUC void dispatch_custom_op(
        std::shared_ptr<const CustomOp> op, const Param& param,
        std::shared_ptr<::megdnn::SmallVector<::mgb::DeviceTensorND>> inputs,
        std::shared_ptr<::megdnn::SmallVector<::mgb::DeviceTensorND>> outputs);

}  // namespace custom

#define to_custom_device(expr) \
    ::custom::to_custom<::mgb::CompNode, ::custom::Device>(expr)
#define to_builtin_device(expr) \
    ::custom::to_builtin<::mgb::CompNode, ::custom::Device>(expr)
#define to_custom_shape(expr) \
    ::custom::to_custom<::megdnn::TensorShape, ::custom::Shape>(expr)
#define to_builtin_shape(expr) \
    ::custom::to_builtin<::megdnn::TensorShape, ::custom::Shape>(expr)
#define to_custom_dtype(expr) \
    ::custom::to_custom<::megdnn::DType, ::custom::DType>(expr)
#define to_builtin_dtype(expr) \
    ::custom::to_builtin<::megdnn::DType, ::custom::DType>(expr)
#define to_custom_format(expr) \
    ::custom::to_custom<::megdnn::TensorLayout::Format, ::custom::Format>(expr)
#define to_builtin_format(expr) \
    ::custom::to_builtin<::megdnn::TensorLayout::Format, ::custom::Format>(expr)
#define to_custom_tensor(expr) \
    ::custom::to_custom<::mgb::DeviceTensorND, ::custom::Tensor>(expr)
#define to_builtin_tensor(expr) \
    ::custom::to_builtin<::mgb::DeviceTensorND, ::custom::Tensor>(expr)
