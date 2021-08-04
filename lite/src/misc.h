/**
 * \file include/misc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "lite_build_config.h"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include "lite/common_enum_c.h"
#include "lite/global.h"

namespace lite {
#if LITE_ENABLE_EXCEPTION
/*! \brief The error class in lite.
 *
 * It can be used to represent both an error caused by the invalid
 * input of the caller or an invalid runtime condition.
 *
 * The necessary presumption should be guaranteed by assertions instead of
 * exceptions.
 */
class Error : public std::exception {
public:
    Error(const std::string& msg) : m_msg("Error: " + msg) {}
    const char* what() const noexcept override { return m_msg.c_str(); }

private:
    std::string m_msg;
};
#endif

std::string ssprintf(const char* fmt = 0, ...)
        __attribute__((format(printf, 1, 2)));

/*!
 * \brief Print a message.
 *
 * The message is printed only if level is above or equals to the current log
 * level.
 */
void print_log(LiteLogLevel level, const char* format = 0, ...)
        __attribute__((format(printf, 2, 3)));
}  // namespace lite

#if LITE_ENABLE_LOGGING
#define LITE_LOG_(level, msg...)                     \
    do {                                             \
        lite::print_log(LiteLogLevel::level, ##msg); \
    } while (0)
#else
#define LITE_LOG_(level, msg...) (void)0
#endif

#define LITE_LOG(fmt...) LITE_LOG_(DEBUG, fmt);
#define LITE_DEBUG(fmt...) LITE_LOG_(DEBUG, fmt);
#define LITE_WARN(fmt...) LITE_LOG_(WARN, fmt);
#define LITE_ERROR(fmt...) LITE_LOG_(ERROR, fmt);

#if LITE_ENABLE_EXCEPTION
#define LITE_THROW(msg) throw lite::Error(msg)
#else
#define LITE_THROW(msg)   \
    do {                  \
        LITE_ERROR(msg);  \
        __builtin_trap(); \
    } while (0)
#endif

#if LITE_ENABLE_EXCEPTION
#define LITE_ERROR_HANDLER_BEGIN try {
#define LITE_ERROR_HANDLER_END                                        \
    }                                                                 \
    catch (const ::lite::Error& e) {                                  \
        std::string msg = std::string("Lite exception: ") + e.what(); \
        LITE_ERROR("%s.", msg.c_str());                               \
        throw;                                                        \
    }

#else
#define LITE_ERROR_HANDLER_BEGIN
#define LITE_ERROR_HANDLER_END
#endif

/*! \brief Return an error if the given pointer is null pointer.
 *
 * The macro is used to ensure the validity of the passing context pointer.
 */
#define LITE_CHECK_NON_NULL_POINTER(ptr) \
    LITE_ASSERT(ptr != nullptr, "Input ptr is null.")

//! branch prediction hint: likely to take
#define lite_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define lite_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

#if LITE_ENABLE_LOGGING
#if LITE_ASSERT_LOC
#define LITE_ASSERT(expr, msg...)                                           \
    do {                                                                    \
        if (lite_unlikely(!(expr))) {                                       \
            auto info = lite::ssprintf(msg);                                \
            LITE_THROW(                                                     \
                    lite::ssprintf("Assert \' %s \' failed at file : %s \n" \
                                   "line %d : %s,\nextra "                  \
                                   "message: %s",                           \
                                   #expr, __FILE__, __LINE__,               \
                                   __PRETTY_FUNCTION__, info.c_str()));     \
        }                                                                   \
    } while (0)
#else
#define LITE_ASSERT(expr, msg...)                                          \
    do {                                                                   \
        if (lite_unlikely(!(expr))) {                                      \
            auto info = lite::ssprintf(msg);                               \
            LITE_THROW(lite::ssprintf(                                     \
                    "Assert \' %s \' failed at file : %s \n"               \
                    "line %d : %s,\nextra "                                \
                    "message: %s",                                         \
                    #expr, "about location info, please build with debug", \
                    __LINE__, __PRETTY_FUNCTION__, info.c_str()));         \
        }                                                                  \
    } while (0)
#endif
#else
#define LITE_ASSERT(expr, msg...)                  \
    do {                                           \
        if (lite_unlikely(!(expr))) {              \
            auto msg_string = lite::ssprintf(msg); \
            LITE_THROW(msg_string.c_str());        \
        }                                          \
    } while (0)
#endif

#define LITE_MARK_USED_VAR(var) ((void)var)

namespace lite {
class ScopedTimer {
public:
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::nanoseconds Nsec;

    ScopedTimer(std::string name) : m_name(name) { m_start = Clock::now(); }
    ~ScopedTimer() {
        m_stop = Clock::now();
        std::chrono::duration<double> elapsed = m_stop - m_start;
        Nsec u = std::chrono::duration_cast<Nsec>(elapsed);
        auto msg = ssprintf("%s used time %fms.", m_name.c_str(),
                            static_cast<double>(u.count()) / 1000000.f);
        LITE_LOG("%s", msg.c_str());
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start, m_stop;
    const std::string m_name;
};

class Timer {
public:
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::nanoseconds Nsec;

    Timer(std::string name) : m_name(name) { m_start = Clock::now(); }
    double get_used_time() {
        m_stop = Clock::now();
        std::chrono::duration<double> elapsed = m_stop - m_start;
        Nsec u = std::chrono::duration_cast<Nsec>(elapsed);
        return static_cast<double>(u.count()) / 1000000.0;
    }
    void print_used_time(int iter) {
        m_stop = Clock::now();
        std::chrono::duration<double> elapsed = m_stop - m_start;
        Nsec u = std::chrono::duration_cast<Nsec>(elapsed);
        printf("%s used time %f ms\n", (m_name + std::to_string(iter)).c_str(),
               static_cast<double>(u.count()) / 1000000.0);
    }
    void reset_start() { m_start = Clock::now(); }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start, m_stop;
    const std::string m_name;
};

inline void mark_used_variable() {}
template <typename T, typename... Arg>
inline void mark_used_variable(T firstArg, Arg... args) {
    LITE_MARK_USED_VAR(firstArg);
    mark_used_variable(args...);
}
}  // namespace lite

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#undef CONST
#define F_OK 0
#define RTLD_LAZY 0
// On the windows platform we use a lib_filename without a full path so
// the win-api "LoadLibrary" would uses a standard search strategy to
// find the lib module. As we cannot access to the lib_filename without a
// full path, we should not use "access(a, b)" to verify it.
#define access(a, b) false
static inline void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibrary(file));
}

static inline char* dlerror() {
    const char* errmsg = "dlerror not aviable in windows";
    return const_cast<char*>(errmsg);
}

static inline void* dlsym(void* handle, const char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return reinterpret_cast<void*>(symbol);
}
#elif __linux__ || __unix__ || __APPLE__
#include <dlfcn.h>
#include <unistd.h>
#endif

#if __DEPLOY_ON_XP_SP2__
//! refer to
//! https://docs.microsoft.com/en-us/cpp/build/configuring-programs-for-windows-xp?view=msvc-160
//! xp sp2 do not support vc runtime fully, casused by KERNEL32.dll do not
//! implement some base apis for c++ std function, for example,
//! std::mutex/std::thread/std::condition_variable as a workround, we will
//! disable some MegEngine feature on xp sp2 env, for exampe, multi-thread etc!
#define LITE_MUTEX size_t
#define LITE_RECURSIVE_MUTEX size_t
#define LITE_LOCK_GUARD(mtx) LITE_MARK_USED_VAR(mtx)
#define LITE_LOCK_GUARD_UNIQUE(mtx) LITE_MARK_USED_VAR(mtx)
#define LITE_LOCK_GUARD_SHARED(mtx) LITE_MARK_USED_VAR(LITE_MARK_USED_VAR)
#else
#define LITE_MUTEX std::mutex
#define LITE_RECURSIVE_MUTEX std::recursive_mutex
#define LITE_LOCK_GUARD(mtx) \
    std::lock_guard<decltype(mtx)> LITE_LOCK_GUARD_CTOR(mtx)

#define LITE_LOCK_GUARD_UNIQUE(mtx) \
    std::unique_lock<decltype(mtx)> LITE_LOCK_GUARD_CTOR(mtx)

#define LITE_LOCK_GUARD_SHARED(mtx) \
    std::shared_lock<decltype(mtx)> LITE_LOCK_GUARD_CTOR(mtx)
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
