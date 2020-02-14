/**
 * \file src/core/test/system.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/system.h"
#include "megbrain/utils/timer.h"
#include "megbrain/test/helper.h"

#if MGB_BUILD_SLIM_SERVING || defined(ANDROID) || defined(WIN32) || defined(IOS) || defined(__APPLE__)
#pragma message "sys test disabled on unsupported platforms"

#else

#include <unistd.h>

using namespace mgb;
using namespace sys;

using Result = TimedFuncInvoker::Result;
using Param = TimedFuncInvoker::Param;
namespace {
    struct SleepParam {
        double init_time = 0;
        double sleep_time = 0;
    };

    //! double sleep(double secs); return secs * 2 for check
    Result func_sleep(const Param &param) {
        auto sp = param.as_single_pod<SleepParam>();
        mgb_assert(sp.sleep_time >= 0);
        usleep(sp.sleep_time * 1e6);
        return Result::from_pod(sp.sleep_time * 2);
    }

    void func_sleep_init(const Param &param) {
        auto sp = param.as_single_pod<SleepParam>();
        mgb_assert(sp.init_time >= 0);
        if (sp.init_time > 0) {
            usleep(sp.init_time * 1e6);
        }
    }

}

namespace mgb {
namespace sys {
    class TimedFuncInvokerTest {
        static auto do_make(bool has_init) {
            auto ins = TimedFuncInvoker::make_test_ins();
            if (has_init) {
                ins->register_func(0, func_sleep, func_sleep_init);
            } else {
                ins->register_func(0, func_sleep);
            }
            return ins;
        }

        public:
            static auto make_ins(bool has_init = false) {
                auto ins = do_make(has_init);
                auto do_fork = [has_init](const std::string &arg) {
                    auto pid = fork();
                    if (pid)
                        return pid;
                    auto ins = do_make(has_init);
                    ins->fork_exec_impl_mainloop(arg.c_str());
                    mgb_assert(0);
                };
                ins->set_fork_exec_impl(do_fork);
                return ins;
            }
    };
}
}

TEST(TestSystem, TimedFuncInvokerBasic) {
    auto ins = TimedFuncInvokerTest::make_ins();
    double time = 0.1;
    SleepParam sleep_param{0., time};
    RealTimer timer;
    auto ret = ins->invoke(0, Param::from_pod(sleep_param), time * 2);
    auto tused = timer.get_secs();
    ASSERT_GT(tused, time);
    ASSERT_EQ(ret.val().as_single_pod<double>(), time * 2);

    // check max time in the second invocation
    timer.reset();
    ret = ins->invoke(0, Param::from_pod(sleep_param), time * 2);
    tused = timer.get_secs();
    ASSERT_GT(tused, time);
    ASSERT_LT(tused, time * 2);
    ASSERT_EQ(ret.val().as_single_pod<double>(), time * 2);
};

TEST(TestSystem, TimedFuncInvokerTimeout) {
    auto ins = TimedFuncInvokerTest::make_ins();
    double time = 0.1;
    SleepParam sleep_param{0., time};
    auto ret = ins->invoke(0, Param::from_pod(sleep_param), time / 2);
    ASSERT_FALSE(ret.valid());
}

TEST(TestSystem, TimedFuncInvokerThreadSafety) {
    // since TimedFuncInvoker uses a singleton, it is important to be
    // thread-safe
    auto ins = TimedFuncInvokerTest::make_ins();

    std::atomic_size_t nr_ready{0};

    auto worker = [&](double *ret, double sleep_time, double timeout) {
        ++ nr_ready;
        while (nr_ready.load() != 2)
            std::this_thread::yield();
        SleepParam sleep_param{0., sleep_time};
        for (int i = 0; i < 5; ++ i) {
            auto result = ins->invoke(0, Param::from_pod(sleep_param), timeout);
            if (!result.valid())
                *ret = -1;
            else
                *ret = result->as_single_pod<double>();
        }
    };
    double ret0, ret1;
    std::thread
        th0{worker, &ret0, 0.1, 0.15},
        th1{worker, &ret1, 0.2, 0.15};
    th0.join();
    th1.join();

    ASSERT_EQ(0.2, ret0);
    ASSERT_EQ(-1., ret1);
}

TEST(TestSystem, TimedFuncInvokerException) {
    auto ins = TimedFuncInvokerTest::make_ins();
    double time = -1;
    SleepParam sleep_param{0., time};
    ASSERT_THROW(ins->invoke(0, Param::from_pod(sleep_param), 0.1),
            TimedFuncInvoker::RemoteError);
}

TEST(TestSystem, TimedFuncInvokerInitFunc) {
    auto ins = TimedFuncInvokerTest::make_ins(true);
    SleepParam sleep_param;
    sleep_param.init_time = 0.1;
    sleep_param.sleep_time = 0.1;
    RealTimer timer;
    auto ret = ins->invoke(0, Param::from_pod(sleep_param), 0.15);
    ASSERT_GT(timer.get_secs(), 0.2);
    ASSERT_EQ(ret.val().as_single_pod<double>(), 0.2);
    timer.reset();
    ret = ins->invoke(0, Param::from_pod(sleep_param), 0.05);
    ASSERT_FALSE(ret.valid());
}

#endif // disable tests on some platforms

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

