#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/opr_param_defs.h"

#include <gtest/gtest.h>

namespace megdnn {
namespace test {
namespace relayout {
// clang-format off
#define FIRST_RELAYOUT_CASE cv

#define FOREACH_RELAYOUT_NONFIRST_CASE(cb) \
    cb(cv_ch3) \
    cb(cv_ch5) \
    cb(broadcast) \
    cb(negative) \
    cb(transpose) \

#define FOREACH_RELAYOUT_CASE(cb) \
    cb(FIRST_RELAYOUT_CASE) \
    FOREACH_RELAYOUT_NONFIRST_CASE(cb)

#define def_tags(name) struct name{};
    FOREACH_RELAYOUT_CASE(def_tags);
#undef def_tags

    template<typename tag>
    void run_test(Handle *handle);

#define t(n) ,n
    typedef ::testing::Types<FIRST_RELAYOUT_CASE
        FOREACH_RELAYOUT_NONFIRST_CASE(t)> test_types;
#undef t
// clang-format on

struct TestArg {
    TensorLayout src;
    TensorLayout dst;
    TestArg() = default;
    TestArg(TensorLayout src_, TensorLayout dst_) : src(src_), dst(dst_) {}
};

void run_cv_benchmark(Handle* handle);

}  // namespace relayout
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
