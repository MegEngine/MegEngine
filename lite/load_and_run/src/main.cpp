/**
 * \file lite/load_and_run/src/main.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include <gflags/gflags.h>
#include <string>
#include "strategys/strategy.h"

int main(int argc, char** argv) {
    std::string usage = "load_and_run <model_path> [options...]";
    if (argc < 2) {
        printf("usage: %s\n", usage.c_str());
        return -1;
    }
    gflags::SetUsageMessage(usage);
    gflags::SetVersionString("1.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::string model_path = argv[1];
    auto strategy = lar::StrategyBase::create_strategy(model_path);
    strategy->run();
    gflags::ShutDownCommandLineFlags();

    return 0;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
