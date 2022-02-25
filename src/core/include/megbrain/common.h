#pragma once

#include "megbrain_build_config.h"
#include "megdnn/basic_types.h"
#include "megdnn/common.h"

#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace mgb {

/* ================ compiler related ================  */

//! comma to be used in macros for template arguments
#define MGB_COMMA ,

//! branch prediction hint: likely to take
#define mgb_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define mgb_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

//! mark a var to be used
#define MGB_MARK_USED_VAR(var) static_cast<void>(var)

//! remove padding in a struct
#define MGB_PACKED __attribute__((packed))

//! ask the compiler to not inline a function
#define MGB_NOINLINE __attribute__((noinline))

//! warn if result of a function is not used
#define MGB_WARN_UNUSED_RESULT __attribute__((warn_unused_result))

#if __cplusplus >= 201703L || __clang_major__ >= 4
#define MGB_FALLTHRU [[fallthrough]];
#elif __GNUC__ >= 7
#define MGB_FALLTHRU __attribute__((fallthrough));
#else
#define MGB_FALLTHRU
#endif

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
#define MGB_CATCH(_decl, _stmt) catch (_decl) _stmt

#else

#if MGB_ENABLE_LOGGING
#define mgb_throw_raw(_exc...) ::mgb::__on_exception_throw__(_exc)
void __on_exception_throw__(const std::exception& exc) __attribute__((noreturn));
#else
#define mgb_throw_raw(_exc...) mgb_trap()
#endif
#define MGB_TRY
#define MGB_CATCH(_decl, _stmt)

#endif  // MGB_ENABLE_EXCEPTION

//! used after try-catch block, like try-finally construct in python
#define MGB_FINALLY(_stmt) \
    MGB_CATCH(..., {       \
        _stmt;             \
        throw;             \
    })                     \
    _stmt

#if MGB_ENABLE_LOGGING
//! throw exception with given message
#define mgb_throw(_exc, _msg...) mgb_throw_raw(_exc(::mgb::ssprintf(_msg)))
#else
//! throw exception with given message
#define mgb_throw(_exc, _msg...) mgb_throw_raw(_exc(""))
#endif

//! throw exception with given message if condition is true
#define mgb_throw_if(_cond, _exc, _msg...) \
    do {                                   \
        if (mgb_unlikely((_cond)))         \
            mgb_throw(_exc, _msg);         \
    } while (0)

// assert
MGE_WIN_DECLSPEC_FUC void __assert_fail__(
        const char* file, int line, const char* func, const char* expr,
        const char* msg_fmt = 0, ...) __attribute__((format(printf, 5, 6), noreturn));
#if MGB_ASSERT_LOC
/*!
 * \brief extended assert
 * extra diagnostics message (in printf format) could be printed when assertion
 * fails; the asserted expression is guaranteed to be evaluated
 */
#define mgb_assert(expr, msg...)                                            \
    do {                                                                    \
        if (mgb_unlikely(!(expr)))                                          \
            ::mgb::__assert_fail__(                                         \
                    __FILE__, __LINE__, __PRETTY_FUNCTION__, #expr, ##msg); \
    } while (0)
#else
#define mgb_assert(expr, msg...)                                                    \
    do {                                                                            \
        if (mgb_unlikely(!(expr)))                                                  \
            ::mgb::__assert_fail__(                                                 \
                    "about location info, please build with debug", __LINE__, NULL, \
                    #expr, ##msg);                                                  \
    } while (0)
#endif  // MGB_ASSERT_LOC

/* ================ logging ================  */
#if MGB_ASSERT_LOC
#define mgb_log_debug(fmt...) \
    _mgb_do_log(::mgb::LogLevel::DEBUG, __FILE__, __func__, __LINE__, fmt)
#define mgb_log(fmt...) \
    _mgb_do_log(::mgb::LogLevel::INFO, __FILE__, __func__, __LINE__, fmt)
#define mgb_log_warn(fmt...) \
    _mgb_do_log(::mgb::LogLevel::WARN, __FILE__, __func__, __LINE__, fmt)
#define mgb_log_error(fmt...) \
    _mgb_do_log(::mgb::LogLevel::ERROR, __FILE__, __func__, __LINE__, fmt)
#else
#define LOC                   "about location info, please build with debug"
#define mgb_log_debug(fmt...) _mgb_do_log(::mgb::LogLevel::DEBUG, "", "", __LINE__, fmt)
#define mgb_log(fmt...)       _mgb_do_log(::mgb::LogLevel::INFO, "", "", __LINE__, fmt)
#define mgb_log_warn(fmt...)  _mgb_do_log(::mgb::LogLevel::WARN, "", "", __LINE__, fmt)
#define mgb_log_error(fmt...) \
    _mgb_do_log(::mgb::LogLevel::ERROR, LOC, "", __LINE__, fmt)
#endif
enum class LogLevel { DEBUG, INFO, WARN, ERROR, NO_LOG };

typedef void (*LogHandler)(
        LogLevel level, const char* file, const char* func, int line, const char* fmt,
        va_list ap);

/*!
 * \brief set logging level
 * messages lower than given level would not be sent to log handler
 *
 * \return previous log level
 */
MGE_WIN_DECLSPEC_FUC LogLevel set_log_level(LogLevel level);

/*!
 * \brief get logging level
 *
 * \return current log level
 */
MGE_WIN_DECLSPEC_FUC LogLevel get_log_level();

/*!
 * \brief set callback for receiving log requests
 * \return previous log handler
 */
MGE_WIN_DECLSPEC_FUC LogHandler set_log_handler(LogHandler handler);

#if MGB_ENABLE_LOGGING
MGE_WIN_DECLSPEC_FUC void __log__(
        LogLevel level, const char* file, const char* func, int line, const char* fmt,
        ...) __attribute__((format(printf, 5, 6)));

#define _mgb_do_log ::mgb::__log__
//! make a string used for log
#define mgb_ssprintf_log ::mgb::ssprintf
//! v if log is enabled, and "" if not
#define mgb_cstr_log(v) v
#else
#define _mgb_do_log(...) \
    do {                 \
    } while (0)
#define mgb_ssprintf_log(...) \
    ::std::string {}
#define mgb_cstr_log(v) ""
#endif  // MGB_ENABLE_LOGGING

/* ================ misc ================  */

// use some macro tricks to get lock guard with unique variable name
#define MGB_TOKENPASTE(x, y)     x##y
#define MGB_TOKENPASTE2(x, y)    MGB_TOKENPASTE(x, y)
#define MGB_LOCK_GUARD_CTOR(mtx) MGB_TOKENPASTE2(__lock_guard_, __LINE__)(mtx)

#if __DEPLOY_ON_XP_SP2__
//! refer to
//! https://docs.microsoft.com/en-us/cpp/build/configuring-programs-for-windows-xp?view=msvc-160
//! xp sp2 do not support vc runtime fully, casused by KERNEL32.dll do not
//! implement some base apis for c++ std function, for example,
//! std::mutex/std::thread/std::condition_variable as a workround, we will
//! disable some MegEngine feature on xp sp2 env, for exampe, multi-thread etc!
#define MGB_MUTEX                  size_t
#define MGB_RECURSIVE_MUTEX        size_t
#define MGB_LOCK_GUARD(mtx)        MGB_MARK_USED_VAR(mtx)
#define MGB_LOCK_GUARD_UNIQUE(mtx) MGB_MARK_USED_VAR(mtx)
#define MGB_LOCK_GUARD_SHARED(mtx) MGB_MARK_USED_VAR(MGB_MARK_USED_VAR)
#else
#define MGB_MUTEX           std::mutex
#define MGB_RECURSIVE_MUTEX std::recursive_mutex
#define MGB_LOCK_GUARD(mtx) std::lock_guard<decltype(mtx)> MGB_LOCK_GUARD_CTOR(mtx)

#define MGB_LOCK_GUARD_UNIQUE(mtx) \
    std::unique_lock<decltype(mtx)> MGB_LOCK_GUARD_CTOR(mtx)

#define MGB_LOCK_GUARD_SHARED(mtx) \
    std::shared_lock<decltype(mtx)> MGB_LOCK_GUARD_CTOR(mtx)
#endif

/*!
 * \brief printf-like std::string constructor
 */
MGE_WIN_DECLSPEC_FUC std::string ssprintf(const char* fmt, ...)
        __attribute__((format(printf, 1, 2)));

MGE_WIN_DECLSPEC_FUC std::string svsprintf(const char* fmt, va_list ap);

#if 0
// used for win32 with vs prior to 2015
const char* convert_fmt_str(const char *fmt);
static inline const char* operator "" _fmt(const char *fmt, std::size_t) {
    return convert_fmt_str(fmt);
}
#else
static inline constexpr const char* convert_fmt_str(const char* fmt) {
    return fmt;
}
static inline constexpr const char* operator"" _fmt(const char* fmt, std::size_t) {
    return convert_fmt_str(fmt);
}
inline constexpr std::size_t operator"" _z(unsigned long long n) {
    return n;
}
#endif

#define MGB_DEF_ENUM_CLASS_BIT_OPR(cls) MEGDNN_DEF_ENUM_CLASS_BIT_OPR(cls)

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
