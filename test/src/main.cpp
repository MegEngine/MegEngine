/**
 * \file test/src/main.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./rng_seed.h"

#include "megbrain/comp_node.h"
#include "megbrain/test/helper.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>

extern "C" int gtest_main(int argc, char** argv) {
    if (getenv("MGB_TEST_USECPU")) {
        mgb::CompNode::Locator::set_unspec_device_type(
                mgb::CompNode::DeviceType::CPU);
    }
    if (getenv("MGB_TEST_NO_LOG")) {
        mgb::set_log_level(mgb::LogLevel::ERROR);
    }
#ifdef __linux__
    if (getenv("MGB_TEST_WAIT_GDB")) {
        printf("wait gdb attach: pid: %d ", getpid());
        getchar();
    }
#endif
    auto&& listeners = ::testing::UnitTest::GetInstance()->listeners();
    MGB_TRY {
        srand(time(nullptr));
        ::testing::InitGoogleMock(&argc, argv);
        listeners.Append(&mgb::RNGSeedManager::inst());
        auto rst = RUN_ALL_TESTS();
        mgb::CompNode::finalize();
        listeners.Release(&mgb::RNGSeedManager::inst());
        return rst;
    }
    MGB_CATCH(std::exception & exc,
              { mgb_log_warn("uncaught exception: %s", exc.what()); });
    listeners.Release(&mgb::RNGSeedManager::inst());
    return 0;
}

#if !MGB_NO_MAIN
int main(int argc, char** argv) {
    return gtest_main(argc, argv);
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
