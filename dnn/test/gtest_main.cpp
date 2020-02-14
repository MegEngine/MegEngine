/**
 * \file dnn/test/gtest_main.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <gtest/gtest.h>
#include "src/common/utils.h"
#include "test/common/random_state.h"

namespace {

class ResetSeedListener : public ::testing::EmptyTestEventListener {
    void OnTestStart(const ::testing::TestInfo&) override {
        megdnn::test::RandomState::reset();
    }
};

void log_handler(megdnn::LogLevel level, const char* file, const char* func,
                 int line, const char* fmt, va_list ap) {
    if (level < megdnn::LogLevel::ERROR) {
        return;
    }
    char msg[1024];
    vsnprintf(msg, sizeof(msg), fmt, ap);
    fprintf(stderr, "[megdnn] %s @%s:%d %s\n", msg, file, line, func);
}

}  // namespace

#if MEGDNN_X86
#include "../src/x86/utils.h"
#endif

extern "C" int gtest_main(int argc, char** argv) {
    ::megdnn::set_log_handler(log_handler);
    ResetSeedListener listener;
    auto&& listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::InitGoogleTest(&argc, argv);
    listeners.Append(&listener);
    auto ret = RUN_ALL_TESTS();
    listeners.Release(&listener);
    return ret;
}

// vim: syntax=cpp.doxygen
