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

#define STRING_OPTION_WRAP(option, default_val)                              \
    struct StringOptionWrap_##option {                                       \
        StringOptionWrap_##option(const char* val) { FLAGS_##option = val; } \
        ~StringOptionWrap_##option() { FLAGS_##option = default_val; }       \
    };

#define INT32_OPTION_WRAP(option, default_val)                          \
    struct Int32OptionWrap_##option {                                   \
        Int32OptionWrap_##option(int32_t val) { FLAGS_##option = val; } \
        ~Int32OptionWrap_##option() { FLAGS_##option = default_val; }   \
    };
#define DEFINE_BOOL_WRAP(option) BoolOptionWrap_##option flags_##option;
#define DEFINE_STRING_WRAP(option, value) \
    StringOptionWrap_##option flags_##option(value);
#define DEFINE_INT32_WRAP(option, value) Int32OptionWrap_##option flags_##option(value);

#define TEST_BOOL_OPTION(option)        \
    {                                   \
        DEFINE_BOOL_WRAP(option);       \
        run_NormalStrategy(model_path); \
    }
#define TEST_STRING_OPTION(option, value)  \
    {                                      \
        DEFINE_STRING_WRAP(option, value); \
        run_NormalStrategy(model_path);    \
    }
#define TEST_INT32_OPTION(option, value)  \
    {                                     \
        DEFINE_INT32_WRAP(option, value); \
        run_NormalStrategy(model_path);   \
    }
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
