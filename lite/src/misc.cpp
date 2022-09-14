#include "./misc.h"
#include "lite/global.h"

#include <time.h>
#include <chrono>
#include <cstdarg>

#if LITE_BUILD_WITH_MGE
#include "megbrain/common.h"
#endif

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif

using namespace lite;

namespace lite {
namespace log_detail {

LiteLogLevel config_default_log_level() {
    auto default_level = LiteLogLevel::ERROR;
    //! env to config LogLevel
    //!  DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, NO_LOG = 4
    //! for example , export RUNTIME_OVERRIDE_LOG_LEVEL=0, means set LogLevel to
    //! DEBUG
    if (auto env = ::std::getenv("RUNTIME_OVERRIDE_LOG_LEVEL"))
        default_level = static_cast<LiteLogLevel>(std::stoi(env));

#ifdef __ANDROID__
    //! special for Android prop, attention: getprop may need permission
    char buf[PROP_VALUE_MAX];
    if (__system_property_get("RUNTIME_OVERRIDE_LOG_LEVEL", buf) > 0) {
        default_level = static_cast<LiteLogLevel>(atoi(buf));
    }
#endif

    return default_level;
}
LiteLogLevel current_log_level = config_default_log_level();

template <class T, size_t N>
constexpr size_t countof(T (&)[N]) {
    return N;
}
}  // namespace log_detail
}  // namespace lite

namespace {
std::string svsprintf(const char* fmt, va_list ap_orig) {
    int size = 100; /* Guess we need no more than 100 bytes */
    char* p;

    if ((p = (char*)malloc(size)) == nullptr)
        return "svsprintf: malloc failed";

    for (;;) {
        va_list ap;
        va_copy(ap, ap_orig);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

        if (n < 0)
            return "svsprintf: vsnprintf failed";

        if (n < size) {
            std::string rst(p);
            free(p);
            return rst;
        }

        size = n + 1;

        char* np = (char*)realloc(p, size);
        if (!np) {
            free(p);
            return "svsprintf: realloc failed";
        } else
            p = np;
    }
}
}  // namespace

void lite::set_log_level(LiteLogLevel l) {
    log_detail::current_log_level = l;
#if LITE_BUILD_WITH_MGE
    mgb::LogLevel lite_log_level = mgb::LogLevel::DEBUG;
    switch (l) {
        case LiteLogLevel::DEBUG:
            lite_log_level = mgb::LogLevel::DEBUG;
            break;
        case LiteLogLevel::INFO:
            lite_log_level = mgb::LogLevel::INFO;
            break;
        case LiteLogLevel::WARN:
            lite_log_level = mgb::LogLevel::WARN;
            break;
        case LiteLogLevel::ERROR:
            lite_log_level = mgb::LogLevel::ERROR;
            break;
        default:
            LITE_THROW("unkonw loglevel");
    }
    mgb::set_log_level(lite_log_level);
#endif
}

LiteLogLevel lite::get_log_level() {
    return log_detail::current_log_level;
}

std::string lite::ssprintf(const char* format, ...) {
    if (!format)
        return "";
    va_list ap;
    va_start(ap, format);
    auto ret = svsprintf(format, ap);
    va_end(ap);
    return ret;
}

void lite::print_log(LiteLogLevel level, const char* format, ...) {
    if (!format)
        return;
    if (static_cast<uint32_t>(level) < static_cast<uint32_t>(get_log_level())) {
        return;
    }
    using namespace std::chrono;

    auto now = system_clock::now();
    auto now_time_t = system_clock::to_time_t(now);

    tm now_tm;

#if _WIN32
    localtime_s(&now_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &now_tm);
#endif

    auto now_trunc_to_sec = system_clock::from_time_t(mktime(&now_tm));
    auto microsec = duration_cast<microseconds>(now - now_trunc_to_sec);

    char time_buffer[100];
    snprintf(
            time_buffer, log_detail::countof(time_buffer), "%02d:%02d:%02d.%06ld ",
            now_tm.tm_hour, now_tm.tm_min, now_tm.tm_sec, long(microsec.count()));

    const char* prefix[] = {"LITE[DBG] ", "LITE[INF] ", "LITE[WRN] ", "LITE[ERR] "};
    std::string out;
    out += prefix[int(level)];
    out += time_buffer;

    va_list ap;
    va_start(ap, format);
    auto ret = svsprintf(format, ap);
    va_end(ap);
    out += ret;

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "lite", "%s", out.c_str());
#else
    fprintf(stderr, "%s\n", out.c_str());
#endif
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
