/**
 * \file src/core/include/megbrain/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#include <memory>
#include <string>
#include <mutex>
#include <exception>

#include <cstdint>
#include <cstddef>
#include <cstdarg>
#include <cstdlib>

namespace mgb {

/* ================ compiler related ================  */

//! comma to be used in macros for template arguments
#define MGB_COMMA   ,

//! branch prediction hint: likely to take
#define mgb_likely(v)   __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define mgb_unlikely(v)   __builtin_expect(static_cast<bool>(v), 0)

//! mark a var to be used
#define MGB_MARK_USED_VAR(var) static_cast<void>(var)

//! remove padding in a struct
#define MGB_PACKED  __attribute__((packed))

//! ask the compiler to not inline a function
#define MGB_NOINLINE  __attribute__((noinline))

//! warn if result of a function is not used
#define MGB_WARN_UNUSED_RESULT __attribute__((warn_unused_result))


/* ================ exception and assertion ================  */

#ifndef mgb_trap
#define mgb_trap() __builtin_trap()
#endif

#if MGB_ENABLE_EXCEPTION

//! throw raw exception object
#define mgb_throw_raw(_exc...) throw _exc
//! try block
#define MGB_TRY try
//! catch block
#define MGB_CATCH(_decl, _stmt) \
    catch(_decl) _stmt

#else

#if MGB_ENABLE_LOGGING
#define mgb_throw_raw(_exc...) ::mgb::__on_exception_throw__(_exc)
void __on_exception_throw__(const std::exception &exc)
    __attribute__((noreturn));
#else
#define mgb_throw_raw(_exc...) mgb_trap()
#endif
#define MGB_TRY
#define MGB_CATCH(_decl, _stmt)

#endif // MGB_ENABLE_EXCEPTION

//! used after try-catch block, like try-finally construct in python
#define MGB_FINALLY(_stmt) \
    MGB_CATCH(..., {_stmt; throw; }) \
    _stmt

//! throw exception with given message
#define mgb_throw(_exc, _msg...) \
    mgb_throw_raw(_exc(::mgb::ssprintf(_msg))) \

//! throw exception with given message if condition is true
#define mgb_throw_if(_cond, _exc, _msg...) \
    do { \
        if (mgb_unlikely((_cond))) \
            mgb_throw(_exc, _msg); \
    } while(0)

// assert
#if MGB_ASSERT_LOC
/*!
 * \brief extended assert
 * extra diagnostics message (in printf format) could be printed when assertion
 * fails; the asserted expression is guaranteed to be evaluated
 */
#define mgb_assert(expr, msg...) \
    do { \
        if (mgb_unlikely(!(expr))) \
            ::mgb::__assert_fail__(__FILE__, __LINE__, \
                    __PRETTY_FUNCTION__, # expr, ##msg); \
    } while(0)
void __assert_fail__(
        const char *file, int line, const char *func,
        const char *expr, const char *msg_fmt = 0, ...)
    __attribute__((format(printf, 5, 6), noreturn));
#else
#define mgb_assert(expr, msg...) \
    do { \
        if (mgb_unlikely(!(expr))) \
            ::mgb::__assert_fail__(); \
    } while(0)
void __assert_fail__() __attribute__((noreturn));
#endif // MGB_ASSERT_LOC

/* ================ logging ================  */
//! caused by need remve sensitive words at opt release
#if MGB_ENABLE_LOGGING
#define mgb_log_debug(fmt...) \
    _mgb_do_log(::mgb::LogLevel::DEBUG, __FILE__, __func__, __LINE__, fmt)
#define mgb_log(fmt...) \
    _mgb_do_log(::mgb::LogLevel::INFO, __FILE__, __func__, __LINE__, fmt)
#define mgb_log_warn(fmt...) \
    _mgb_do_log(::mgb::LogLevel::WARN, __FILE__, __func__, __LINE__, fmt)
#define mgb_log_error(fmt...) \
    _mgb_do_log(::mgb::LogLevel::ERROR, __FILE__, __func__, __LINE__, fmt)
#else
#define mgb_log_debug(fmt...) \
    _mgb_do_log(::mgb::LogLevel::DEBUG, "", "", 1, fmt)
#define mgb_log(fmt...) \
    _mgb_do_log(::mgb::LogLevel::INFO, "", "", 1, fmt)
#define mgb_log_warn(fmt...) \
    _mgb_do_log(::mgb::LogLevel::WARN, "", "", 1, fmt)
#define mgb_log_error(fmt...) \
    _mgb_do_log(::mgb::LogLevel::ERROR, "", "", 1, fmt)
#endif
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

typedef void(*LogHandler)(LogLevel level,
        const char *file, const char *func, int line, const char *fmt,
        va_list ap);

/*!
 * \brief set logging level
 * messages lower than given level would not be sent to log handler
 *
 * \return previous log level
 */
LogLevel set_log_level(LogLevel level);

/*!
 * \brief set callback for receiving log requests
 * \return previous log handler
 */
LogHandler set_log_handler(LogHandler handler);

#if MGB_ENABLE_LOGGING
void __log__(LogLevel level, const char *file, const char *func, int line,
        const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));

#define _mgb_do_log ::mgb::__log__
//! make a string used for log
#define mgb_ssprintf_log ::mgb::ssprintf
//! v if log is enabled, and "" if not
#define mgb_cstr_log(v) v
#else
#define _mgb_do_log(...) do{} while(0)
#define mgb_ssprintf_log(...) ::std::string{}
#define mgb_cstr_log(v) ""
#endif // MGB_ENABLE_LOGGING

/* ================ misc ================  */

#if MGB_ENABLE_GETENV
#define MGB_GETENV  ::std::getenv
#else
#define MGB_GETENV(_name)  static_cast<char*>(nullptr)
#endif

// use some macro tricks to get lock guard with unique variable name
#define MGB_TOKENPASTE(x, y) x ## y
#define MGB_TOKENPASTE2(x, y) MGB_TOKENPASTE(x, y)
#define MGB_LOCK_GUARD_CTOR(mtx) MGB_TOKENPASTE2(__lock_guard_, __LINE__)(mtx)

#define MGB_LOCK_GUARD(mtx) \
    std::lock_guard<decltype(mtx)> MGB_LOCK_GUARD_CTOR(mtx)

#define MGB_LOCK_GUARD_UNIQUE(mtx) \
    std::unique_lock<decltype(mtx)> MGB_LOCK_GUARD_CTOR(mtx)

#define MGB_LOCK_GUARD_SHARED(mtx) \
	std::shared_lock<decltype(mtx)> MGB_LOCK_GUARD_CTOR(mtx)

/*!
 * \brief printf-like std::string constructor
 */
std::string ssprintf(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));

std::string svsprintf(const char *fmt, va_list ap);

#if 0
// used for win32 with vs prior to 2015
const char* convert_fmt_str(const char *fmt);
static inline const char* operator "" _fmt(const char *fmt, std::size_t) {
    return convert_fmt_str(fmt);
}
#else
static inline constexpr const char* convert_fmt_str(const char *fmt) {
    return fmt;
}
static inline constexpr const char* operator "" _fmt(
        const char *fmt, std::size_t) {
    return convert_fmt_str(fmt);
}
inline constexpr std::size_t operator"" _z(unsigned long long n) {
    return n;
}
#endif
}   // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
