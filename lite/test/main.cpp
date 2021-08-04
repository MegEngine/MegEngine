/**
 * \file test/main.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <gtest/gtest.h>
#include "../src/misc.h"
#include "lite/global.h"

namespace {

class ResetSeedListener : public ::testing::EmptyTestEventListener {
    void OnTestStart(const ::testing::TestInfo&) override {}
};

}  // namespace

int main(int argc, char** argv) {
    ResetSeedListener listener;
    auto&& listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::InitGoogleTest(&argc, argv);
    listeners.Append(&listener);
    lite::set_log_level(LiteLogLevel::WARN);
    auto ret = RUN_ALL_TESTS();
    listeners.Release(&listener);
    return ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
