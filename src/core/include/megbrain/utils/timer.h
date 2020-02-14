/**
 * \file src/core/include/megbrain/utils/timer.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <string>

namespace mgb {

namespace timer_impl {
    struct TimeSpec {
        int64_t sec, nsec;

        std::string to_string() const;

        //! get time untile another time point
        double time_until_secs(const TimeSpec &end) const {
            // cast to int to avoid softfloat on 32bit platforms
            return static_cast<int>(end.sec - sec) +
                static_cast<int>(end.nsec - nsec) * 1e-9;
        }
    };

    template<class TimerTrait>
    class Timer {
        TimeSpec m_start;

        public:
            static TimeSpec get_time();

            Timer() {
                reset();
            }

            void reset() {
                m_start = get_time();
            }

            double get_secs() const {
                return m_start.time_until_secs(get_time());
            }

            //! get milliseconds (one thousandth of a second)
            double get_msecs() {
                return get_secs() * 1e3;
            }

            //! get seconds and reset
            double get_secs_reset() {
                auto ret = get_secs();
                reset();
                return ret;
            }

            //! get milliseconds and reset
            double get_msecs_reset() {
                auto ret = get_msecs();
                reset();
                return ret;
            }
    };

    class RealTimeTrait;
}

using timer_impl::TimeSpec;

/*!
 * \brief measure real time in nanoseconds precision
 */
using RealTimer = timer_impl::Timer<timer_impl::RealTimeTrait>;

}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

