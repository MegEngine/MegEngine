#pragma once

//! helper function like define: lite/src/misc.h, but we need example code just
//! depends on install header, not depends any lite src file
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

std::string exampe_ssprintf(const char* fmt = 0, ...)
        __attribute__((format(printf, 1, 2)));

#define LITE_EXAMPLE_THROW(msg)          \
    do {                                 \
        std::string msg_str(msg);        \
        printf("%s\n", msg_str.c_str()); \
        __builtin_trap();                \
    } while (0)

//! branch prediction hint: likely to take
#define lite_example_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define lite_example_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

#define LITE_EXAMPLE_ASSERT(expr, msg...)                                           \
    do {                                                                            \
        if (lite_example_unlikely(!(expr))) {                                       \
            auto info = exampe_ssprintf(msg);                                       \
            LITE_EXAMPLE_THROW(exampe_ssprintf(                                     \
                    "Assert \' %s \' failed at file : %s \n"                        \
                    "line %d : %s,\nextra "                                         \
                    "message: %s",                                                  \
                    #expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, info.c_str())); \
        }                                                                           \
    } while (0)

#define LITE_EXAMPLE_MARK_USED_VAR(var) ((void)var)

namespace lite_example_helper {
class ScopedTimer {
public:
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::nanoseconds Nsec;

    ScopedTimer(std::string name) : m_name(name) { m_start = Clock::now(); }
    ~ScopedTimer() {
        m_stop = Clock::now();
        std::chrono::duration<double> elapsed = m_stop - m_start;
        Nsec u = std::chrono::duration_cast<Nsec>(elapsed);
        auto msg = exampe_ssprintf(
                "%s used time %fms.", m_name.c_str(),
                static_cast<double>(u.count()) / 1000000.f);
        printf("%s", msg.c_str());
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
    LITE_EXAMPLE_MARK_USED_VAR(firstArg);
    mark_used_variable(args...);
}
}  // namespace lite_example_helper

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#undef CONST
#define F_OK      0
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
#define LITE_MUTEX                  size_t
#define LITE_RECURSIVE_MUTEX        size_t
#define LITE_LOCK_GUARD(mtx)        LITE_EXAMPLE_MARK_USED_VAR(mtx)
#define LITE_LOCK_GUARD_UNIQUE(mtx) LITE_EXAMPLE_MARK_USED_VAR(mtx)
#define LITE_LOCK_GUARD_SHARED(mtx) \
    LITE_EXAMPLE_MARK_USED_VAR(LITE_EXAMPLE_MARK_USED_VAR)
#else
#define LITE_MUTEX           std::mutex
#define LITE_RECURSIVE_MUTEX std::recursive_mutex
#define LITE_LOCK_GUARD(mtx) std::lock_guard<decltype(mtx)> LITE_LOCK_GUARD_CTOR(mtx)

#define LITE_LOCK_GUARD_UNIQUE(mtx) \
    std::unique_lock<decltype(mtx)> LITE_LOCK_GUARD_CTOR(mtx)

#define LITE_LOCK_GUARD_SHARED(mtx) \
    std::shared_lock<decltype(mtx)> LITE_LOCK_GUARD_CTOR(mtx)
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
