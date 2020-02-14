/**
 * \file dnn/src/common/utils.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/arch.h"

//! a comma to be used in macro for template params
#define MEGDNN_COMMA    ,
#define MEGDNN_MARK_USED_VAR(v) static_cast<void>(v)

#if MEGDNN_ENABLE_MANGLING
#define megdnn_mangle(x) ("")
#else
#define megdnn_mangle(x) (x)
#endif // MEGDNN_ENABLE_MANGLING

#define megdnn_throw(msg) ::megdnn::ErrorHandler::on_megdnn_error( \
        megdnn_mangle(msg))
#define megdnn_throw_if(cond, err_type, msg) do { \
    if (megdnn_unlikely(cond)) { \
        ::megdnn::ErrorHandler::on_##err_type(megdnn_mangle(msg)); \
    } \
} while(0)

//! megdnn_assert
#if MEGDNN_ENABLE_MANGLING
#define megdnn_assert(expr, ...) \
    do { \
        if (megdnn_unlikely(!(expr))) { \
            ::megdnn::__assert_fail__(NULL, 0, NULL, NULL, NULL); \
        } \
    } while (0)
#else
#define megdnn_assert(expr, ...) \
    do { \
        if (megdnn_unlikely(!(expr))) { \
            ::megdnn::__assert_fail__(__FILE__, __LINE__, \
                    __PRETTY_FUNCTION__, # expr, ## __VA_ARGS__); \
        } \
    } while (0)
#endif // MEGDNN_ENABLE_MANGLING

#define megdnn_assert_internal(expr) \
    do { \
        megdnn_assert(expr, "Impossible: internal error."); \
    } while (0)

#define megdnn_ignore(x) (void)(x)

namespace megdnn {

void __assert_fail__(const char *file, int line, const char *func,
        const char *expr, const char *msg_fmt = nullptr, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 5, 6), noreturn))
#endif
	;

void __dummy_printf__(const char *msg_fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 1, 2)))
#endif
;

//! typetrait, just the same as std::is_same in c++11
template <typename T, typename U>
struct is_same {
    static const bool value = false;
};

template <typename T>
struct is_same<T, T> {
    static const bool value = true;
};

} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
