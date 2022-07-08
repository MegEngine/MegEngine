#pragma once

#include <iostream>
#include <thread>
#include "../load_and_run/src/strategys/strategy.h"
#include "../load_and_run/src/strategys/strategy_normal.h"
#include "megbrain/common.h"
#include "megbrain/utils/timer.h"
#include "megbrain/version.h"
#include "megdnn/version.h"
#include "misc.h"

namespace lar {

//! run load_and_run NormalStrategy to test different options

void run_NormalStrategy(std::string model_path);

}  // namespace lar
#define BOOL_OPTION_WRAP(option)                               \
    struct BoolOptionWrap_##option {                           \
        BoolOptionWrap_##option() { FLAGS_##option = true; }   \
        ~BoolOptionWrap_##option() { FLAGS_##option = false; } \
    };

#define DEFINE_WRAP(option) BoolOptionWrap_##option flags_##option;

#define TEST_BOOL_OPTION(option)                \
    {                                           \
        BoolOptionWrap_##option flags_##option; \
        run_NormalStrategy(model_path);         \
    }
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
