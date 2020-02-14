/**
 * \file src/core/impl/common.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"
#include "megbrain/exception.h"
#include "megbrain/system.h"
#include "megbrain/utils/thread.h"

#include "megdnn/basic_types.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#endif

using namespace mgb;

namespace {
    LogLevel min_log_level;
}

#if MGB_ENABLE_LOGGING

#if MGB_EXTERN_API_TIME
extern "C" {
    void mgb_extern_api_get_time(int64_t *sec, int64_t *nsec);
}
#endif

namespace {
void default_log_handler(LogLevel level,
        const char *file, const char *func, int line, const char *fmt,
        va_list ap) {
    if (level < min_log_level)
        return;

#define HDR_FMT	"[%s %s@%s:%d]%s"
    fmt = convert_fmt_str(fmt);

    // we have to use a Spinlock here, since log handler might be called during
    // global finalization when mtx has been destructed
    static Spinlock mtx;

    static const char *hdr_fmt = nullptr;
    if (!hdr_fmt) {
        if (sys::stderr_ansi_color())
            hdr_fmt = "\x1b[32m" HDR_FMT "\x1b[0m ";
        else
            hdr_fmt = HDR_FMT " ";
    }
    const char *warn_reminder = "";
    switch (level) {
        case LogLevel::ERROR:
            if (sys::stderr_ansi_color())
                warn_reminder = "\x1b[1;4;31m[ERR]\x1b[0m";
            else
                warn_reminder = "[ERR]";
            break;
        case LogLevel::WARN:
            if (sys::stderr_ansi_color())
                warn_reminder = "\x1b[1;31m[WARN]\x1b[0m";
            else
                warn_reminder = "[WARN]";
            break;
        case LogLevel::INFO:
            break;
        case LogLevel::DEBUG:
            if (sys::stderr_ansi_color())
                warn_reminder = "\x1b[36m[DEBUG]\x1b[0m";
            else
                warn_reminder = "[DEBUG]";
            break;
        default:
            mgb_throw(MegBrainError, "bad log level");
    }
    char timestr[64];
#if MGB_EXTERN_API_TIME
    {
        static int64_t sec_start, nsec_start;
        int64_t sec, nsec;
        mgb_extern_api_get_time(&sec, &nsec);
        if (!sec_start) {
            sec_start = sec;
            nsec_start = nsec;
        }
        sec -= sec_start;
        nsec -= nsec_start;
        if (nsec < 0) {
            -- sec;
            nsec += 1000000000;
        }
        snprintf(timestr, sizeof(timestr), "%.3f",
                static_cast<int>(sec) + static_cast<int>(nsec) * 1e-9);
    }
#else
    {
        time_t cur_time;
        MGB_LOCK_GUARD(mtx);
        time(&cur_time);
        strftime(timestr, sizeof(timestr), "%d %H:%M:%S", localtime(&cur_time));
    }
#endif

    {
        // find file basename part
        auto f0 = file;
        file = f0 + strlen(f0) - 1;
        while (file >= f0 && *file != '/' && *file != '\\')
            -- file;
        ++ file;
    }
    {
        MGB_LOCK_GUARD(mtx);
        fprintf(stderr, hdr_fmt, timestr, func, file, line, warn_reminder);
        vfprintf(stderr, fmt, ap);
        fputc('\n', stderr);
    }

#ifdef __ANDROID__
    android_LogPriority android_level;
    switch (level) {
        case LogLevel::WARN:
            android_level = ANDROID_LOG_WARN;
            break;
        case LogLevel::INFO:
            android_level = ANDROID_LOG_INFO;
            break;
        case LogLevel::DEBUG:
            android_level = ANDROID_LOG_DEBUG;
            break;
        default:
            android_level = ANDROID_LOG_ERROR;
    }
    __android_log_vprint(android_level, "megbrain", fmt, ap);
#endif

#undef HDR_FMT
}

LogHandler log_handler = default_log_handler;

class MegDNNLogHandler {
    static void dnn_log_handler(megdnn::LogLevel dnn_level, const char* file,
                                const char* func, int line, const char* fmt,
                                va_list ap) {
        mgb::LogLevel mgb_level;
        switch (dnn_level) {
            case megdnn::LogLevel::DEBUG:
                mgb_level = LogLevel::DEBUG;
                break;
            case megdnn::LogLevel::INFO:
                mgb_level = LogLevel::INFO;
                break;
            case megdnn::LogLevel::WARN:
                mgb_level = LogLevel::WARN;
                break;
            default:
                mgb_level = LogLevel::ERROR;
        }
        if (mgb_level < min_log_level) {
            return;
        }

        std::string new_fmt{"[megdnn] "};
        new_fmt.append(fmt);
        log_handler(mgb_level, file, func, line, new_fmt.c_str(), ap);
    }

public:
    MegDNNLogHandler() { megdnn::set_log_handler(dnn_log_handler); }
};
MegDNNLogHandler g_megdnn_log_handler_init;
}  // anonymous namespace

void mgb::__log__(LogLevel level,
        const char *file, const char *func, int line, const char *fmt, ...) {
    if (level < min_log_level)
        return;

    va_list ap;
    va_start(ap, fmt);
    log_handler(level, file, func, line, fmt, ap);
    va_end(ap);
}

/* ===================== forward log in MegWave ===================== */
// common::Log is a weak symbol in megwave
namespace common {
enum class LogLevel { kInfo, kWarn, kDebug, kFatal };
void Log(LogLevel level, char const* file, int line, char const* func,
        char const *fmt, ...) {

    std::string new_fmt("[megwave] ");
    new_fmt.append(fmt);
    va_list ap;
    va_start(ap, fmt);
    log_handler(level == LogLevel::kWarn ? mgb::LogLevel::WARN :
            mgb::LogLevel::DEBUG,
            file, func, line, new_fmt.c_str(), ap);
    va_end(ap);
}
}

#else // MGB_ENABLE_LOGGING

namespace {
    void default_log_handler(LogLevel ,
        const char *, const char *, int , const char *,
        va_list ) {
    }
    LogHandler log_handler = default_log_handler;
}

#endif // MGB_ENABLE_LOGGING

LogLevel mgb::set_log_level(LogLevel level) {
    auto ret = min_log_level;
    min_log_level = level;
    return ret;
}

LogHandler mgb::set_log_handler(LogHandler handler) {
    auto ret = log_handler;
    log_handler = handler;
    return ret;
}

#if MGB_ASSERT_LOC
void mgb::__assert_fail__(
        const char *file, int line, const char *func,
        const char *expr, const char *msg_fmt, ...) {

    std::string msg = ssprintf("assertion `%s' failed at %s:%d: %s",
            expr, file, line, func);
    if (msg_fmt) {
        msg_fmt = convert_fmt_str(msg_fmt);
        va_list ap;
        va_start(ap, msg_fmt);
        msg.append("\nextra message: ");
        msg.append(svsprintf(msg_fmt, ap));
        va_end(ap);
    }
    mgb_throw_raw(AssertionError{msg});
}
#else
void mgb::__assert_fail__() {
    mgb_throw(AssertionError, "assertion failed");
}
#endif

#if MGB_ENABLE_LOGGING && !MGB_ENABLE_EXCEPTION
void mgb::__on_exception_throw__(const std::exception &exc) {
    mgb_log_error("exception thrown: %s", exc.what());
    mgb_log_error("abort now due to previous error");
    mgb_trap();
}
#endif

std::string mgb::svsprintf(const char *fmt, va_list ap_orig) {
    fmt = convert_fmt_str(fmt);
    int size = 100;     /* Guess we need no more than 100 bytes */
    char *p;

    if ((p = (char*)malloc(size)) == nullptr)
        goto err;

    for (; ;) {
        va_list ap;
        va_copy(ap, ap_orig);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

#ifdef WIN32
        if (n == -1) {
            n = _vscprintf(fmt, ap_orig);
            mgb_assert(n >= size);
        }
#endif

        if (n < 0)
            goto err;

        if (n < size) {
            std::string rst(p);
            free(p);
            return rst;
        }

        size = n + 1;

        char *np = (char*)realloc(p, size);
        if (!np) {
            free(p);
            goto err;
        } else
            p = np;
    }

err:
    fprintf(stderr, "could not allocate memory for svsprintf; fmt=%s\n",
            fmt);
    mgb_trap();
}

std::string mgb::ssprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    auto rst = svsprintf(fmt, ap);
    va_end(ap);
    return rst;
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
