/**
 * \file dnn/test/common/elemwise_multi_type.h
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

namespace megdnn {
namespace test {
namespace elemwise_multi_type {
#define FIRST_ELEMWISE_MULTI_TYPE_CASE fuse_mul_add3_int16x32x32x32

#define FOREACH_ELEMWISE_MULTI_TYPE_NONFIRST_CASE(cb)               \
    cb(fuse_mul_add3_iXxf32xf32xi8) cb(round_shr_saturate_iXxi8xi8) \
            cb(fuse_add_rmulh_round_shr_saturate_int16)             \
                    cb(fuse_add_rmulh_round_shr_saturate_int32)

#define FOREACH_ELEMWISE_MULTI_TYPE_CASE(cb) \
    cb(FIRST_ELEMWISE_MULTI_TYPE_CASE)       \
            FOREACH_ELEMWISE_MULTI_TYPE_NONFIRST_CASE(cb)

#define def_tags(name) \
    struct name {};
FOREACH_ELEMWISE_MULTI_TYPE_CASE(def_tags);
#undef def_tags

template <typename tag>
void run_test(Handle* handle);

#define t(n) , n
typedef ::testing::Types<FIRST_ELEMWISE_MULTI_TYPE_CASE
                                 FOREACH_ELEMWISE_MULTI_TYPE_NONFIRST_CASE(t)>
        test_types;
#undef t

}  // namespace elemwise_multi_type
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
