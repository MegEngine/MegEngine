
/**
 * \file lite/load_and_run/src/strategys/strategy.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include "strategy.h"
#include <iostream>

using namespace lar;

std::shared_ptr<StrategyBase> StrategyBase::create_strategy(std::string model_path) {
    if (FLAGS_fitting) {
        return std::make_shared<FittingStrategy>(model_path);
    } else {
        return std::make_shared<NormalStrategy>(model_path);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}