/**
 * \file dnn/test/common/elemwise.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/handle.h"

#include <gtest/gtest.h>
#include "test/common/fix_gtest_on_platforms_without_exception.inl"

namespace megdnn {
namespace test {
namespace elemwise {
// clang-format off
#define FIRST_ELEMWISE_CASE unary

#define FOREACH_ELEMWISE_NONFIRST_CASE(cb) \
    cb(binary_brdcst) \
    cb(binary_non_contig) \
    cb(ternary) \
    cb(ternary_non_contig) \
    cb(fuse_mul_add3) \
    cb(fuse_mul_add3_non_contig) \
    cb(fuse_mul_add4) \
    cb(rmulh) \
    cb(unary1) \
    cb(binary1) \
    cb(ternary1) \
    cb(unary2) \
    cb(binary2) \
    cb(ternary2) \
    cb(unary3) \
    cb(binary3) \
    cb(all_modes) \

#define FOREACH_ELEMWISE_CASE(cb) \
    cb(FIRST_ELEMWISE_CASE) \
    FOREACH_ELEMWISE_NONFIRST_CASE(cb)

#define def_tags(name) struct name{};
    FOREACH_ELEMWISE_CASE(def_tags);
#undef def_tags

    template<typename tag>
    void run_test(Handle *handle);

#define t(n) ,n
    typedef ::testing::Types<FIRST_ELEMWISE_CASE
        FOREACH_ELEMWISE_NONFIRST_CASE(t)> test_types;
#undef t
// clang-format on

}  // namespace elemwise
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
