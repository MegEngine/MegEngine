/**
 * \file src/custom/include/megbrain/custom/custom.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "op.h"
#include "param.h"
#include "tensor.h"

namespace custom {
std::shared_ptr<CustomOp> op_insert(std::string opname, uint32_t version);
}

#define CUSTOM_OP_REG(OpName) \
    CustomOp& _##OpName = (*(op_insert(#OpName, CUSTOM_OP_VERSION)))

#define CUSTOM_OP_REG_BEGIN(OpName) \
    namespace custom {              \
    namespace OpName {

#define CUSTOM_OP_REG_END(OpName) \
    }                             \
    }

#define CASE_TO_PERFORM_USING_HINT(name, case_type, real_type, hint, ...) \
    case (case_type): {                                                   \
        using hint = real_type;                                           \
        return __VA_ARGS__();                                             \
    }

#define CASE_TO_PERFORM_ON_SCALAR(name, case_type, real_type, ...) \
    CASE_TO_PERFORM_USING_HINT(name, case_type, real_type, scalar_t, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(tensor_dtype, name, ...)                               \
    [&]() {                                                                         \
        const auto& dtype = tensor_dtype;                                           \
        switch (dtype.enumv()) {                                                    \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::float32, float, __VA_ARGS__) \
            default:                                                                \
                custom_assert(                                                      \
                        false, "no implemented %s kernel for dtype %s\n", name,     \
                        dtype.str().c_str());                                       \
        }                                                                           \
    }()

#define DISPATCH_INT_TYPES(tensor_dtype, name, ...)                                   \
    [&]() {                                                                           \
        const auto& dtype = tensor_dtype;                                             \
        switch (dtype.enumv()) {                                                      \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int8, int8_t, __VA_ARGS__)     \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::uint8, uint8_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::uint16, uint16_t, __VA_ARGS__) \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int16, int16_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int32, int32_t, __VA_ARGS__)   \
            default:                                                                  \
                custom_assert(                                                        \
                        false, "no implemented %s kernel for dtype %s\n", name,       \
                        dtype.str().c_str());                                         \
        }                                                                             \
    }()

#define DISPATCH_INT_AND_FLOAT_TYPES(tensor_dtype, name, ...)                         \
    [&]() {                                                                           \
        const auto& dtype = tensor_dtype;                                             \
        switch (dtype.enumv()) {                                                      \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int8, int8_t, __VA_ARGS__)     \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::uint8, uint8_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::uint16, uint16_t, __VA_ARGS__) \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int16, int16_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int32, int32_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::float32, float, __VA_ARGS__)   \
            default:                                                                  \
                custom_assert(                                                        \
                        false, "no implemented %s kernel for dtype %s\n", name,       \
                        dtype.str().c_str());                                         \
        }                                                                             \
    }()

#define DISPATCH_SIGN_INT_TYPES(tensor_dtype, name, ...)                            \
    [&]() {                                                                         \
        const auto& dtype = tensor_dtype;                                           \
        switch (dtype.enumv()) {                                                    \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int8, int8_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int16, int16_t, __VA_ARGS__) \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int32, int32_t, __VA_ARGS__) \
            default:                                                                \
                custom_assert(                                                      \
                        false, "no implemented %s kernel for dtype %s\n", name,     \
                        dtype.str().c_str());                                       \
        }                                                                           \
    }()

#define DISPATCH_SIGN_INT_AND_FLOAT_TYPES(tensor_dtype, name, ...)                  \
    [&]() {                                                                         \
        const auto& dtype = tensor_dtype;                                           \
        switch (dtype.enumv()) {                                                    \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::float32, float, __VA_ARGS__) \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int8, int8_t, __VA_ARGS__)   \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int16, int16_t, __VA_ARGS__) \
            CASE_TO_PERFORM_ON_SCALAR(name, DTypeEnum::int32, int32_t, __VA_ARGS__) \
            default:                                                                \
                custom_assert(                                                      \
                        false, "no implemented %s kernel for dtype %s\n", name,     \
                        dtype.str().c_str());                                       \
        }                                                                           \
    }()
