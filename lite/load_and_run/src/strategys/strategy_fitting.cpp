/**
 * \file lite/load_and_run/src/strategys/strategy_fitting.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include "strategy.h"
using namespace lar;

FittingStrategy::FittingStrategy(std::string) {
    mgb_assert("this version don't support Fitting Strategy");
};

void FittingStrategy::run() {
    mgb_assert("this version don't support Fitting Strategy");
};

DEFINE_bool(
        fitting, false,
        "whether to use the fitting model, which will auto profile and get "
        "the best option set!");