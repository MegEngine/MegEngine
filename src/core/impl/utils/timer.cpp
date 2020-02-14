/**
 * \file src/core/impl/utils/timer.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/timer.h"
#include "megbrain/common.h"
#include "megbrain/exception.h"

#include <cstring>

using namespace mgb;
using namespace timer_impl;

#ifdef MGB_EXTERN_API_TIME

extern "C" {
    void mgb_extern_api_get_time(int64_t *sec, int64_t *nsec);
}

class timer_impl::RealTimeTrait {
    public:
        static TimeSpec now() {
            TimeSpec ret;
            mgb_extern_api_get_time(&ret.sec, &ret.nsec);
            return ret;
        }
};

#elif defined(WIN32)
#include <windows.h>
class timer_impl::RealTimeTrait {
    public:
        static TimeSpec now() {
            // on windows, clock() returns wall time
            auto t = clock();
            mgb_assert(t > 0);
            long long
                sec = t / CLOCKS_PER_SEC,
                nsec = t % CLOCKS_PER_SEC * (1000000000ull / CLOCKS_PER_SEC);
            return {sec, nsec};
        }
};

#elif defined(__APPLE__)

#include <mach/mach_time.h>

namespace {
    TimeSpec orwl_gettime() {
        static double orwl_timebase = 0.0;
        static uint64_t orwl_timestart = 0;

        // be more careful in a multithreaded environement
        if (!orwl_timestart) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
            mach_timebase_info_data_t tb = { 0 };
#pragma clang diagnostic pop
            mach_timebase_info(&tb);
            orwl_timebase = tb.numer;
            orwl_timebase /= tb.denom;
            orwl_timestart = mach_absolute_time();
        }
        TimeSpec t;
        double diff = (mach_absolute_time() - orwl_timestart) * orwl_timebase;
        t.sec = diff * 1e-9;
        t.nsec = diff - (t.sec * 1e9);
        return t;
    }

} // anonymous namespace

class timer_impl::RealTimeTrait {
    public:
        static TimeSpec now() {
            return orwl_gettime();
        }
};

#else

#include <sys/time.h>

class timer_impl::RealTimeTrait {
    public:
        static TimeSpec now() {
            struct timeval tv;
            if (mgb_unlikely(gettimeofday(&tv, nullptr))) {
                mgb_throw(SystemError, "gettimeofday failed: %s",
                        strerror(errno));
            }
            return {static_cast<int64_t>(tv.tv_sec),
                static_cast<int64_t>(tv.tv_usec) * 1000};
        }
};
#endif  // end of platform selection

template<class TimerTrait>
TimeSpec Timer<TimerTrait>::get_time() {
    return TimerTrait::now();
}

std::string TimeSpec::to_string() const {
    return ssprintf("%lld.%09lld",
            static_cast<long long>(sec), static_cast<long long>(nsec));
}

template class timer_impl::Timer<timer_impl::RealTimeTrait>;

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

