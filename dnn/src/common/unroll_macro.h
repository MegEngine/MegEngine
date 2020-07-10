/**
 * \file dnn/src/common/unroll_macro.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#define UNROLL_RAW1(cb, v0, a...) cb(0, ##a)
#define UNROLL_RAW2(cb, v0, a...) cb(0, ##a) cb(1, ##a)
#define UNROLL_RAW3(cb, v0, a...) UNROLL_RAW2(cb, v0, ##a) cb(2, ##a)
#define UNROLL_RAW4(cb, v0, a...) \
    UNROLL_RAW2(cb, v0, ##a)      \
    cb(2, ##a) cb(3, ##a)
#define UNROLL_RAW5(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a)
#define UNROLL_RAW6(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a) cb(5, ##a)
#define UNROLL_RAW7(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a) cb(5, ##a) cb(6, ##a)
#define UNROLL_RAW8(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a) cb(5, ##a) cb(6, ##a) cb(7, ##a)
#define UNROLL_RAW9(cb, v0, a...) \
    UNROLL_RAW8(cb, v0, ##a)      \
    cb(8, ##a)
#define UNROLL_RAW10(cb, v0, a...) \
    UNROLL_RAW9(cb, v0, ##a)       \
    cb(9, ##a)
#define UNROLL_RAW11(cb, v0, a...) \
    UNROLL_RAW10(cb, v0, ##a)      \
    cb(10, ##a)
#define UNROLL_RAW12(cb, v0, a...) \
    UNROLL_RAW11(cb, v0, ##a)      \
    cb(11, ##a)
#define UNROLL_RAW13(cb, v0, a...) \
    UNROLL_RAW12(cb, v0, ##a)      \
    cb(12, ##a)
#define UNROLL_RAW14(cb, v0, a...) \
    UNROLL_RAW13(cb, v0, ##a)      \
    cb(13, ##a)
#define UNROLL_RAW15(cb, v0, a...) \
    UNROLL_RAW14(cb, v0, ##a)      \
    cb(14, ##a)

// clang-format off
#define UNROLL_RAW16(cb, v0, a...)                                        \
    UNROLL_RAW8(cb, v0, ##a)                                              \
    cb(8, ##a) cb(9, ##a) cb(10, ##a) cb(11, ##a) cb(12, ##a) cb(13, ##a) \
            cb(14, ##a) cb(15, ##a)
#define UNROLL_RAW17(cb, v0, a...) \
    UNROLL_RAW16(cb, v0, ##a)      \
    cb(16, ##a)
#define UNROLL_RAW24(cb, v0, a...)                                          \
    UNROLL_RAW16(cb, v0, ##a)                                               \
    cb(16, ##a) cb(17, ##a) cb(18, ##a) cb(19, ##a) cb(20, ##a) cb(21, ##a) \
            cb(22, ##a) cb(23, ##a)
#define UNROLL_RAW25(cb, v0, a...) \
    UNROLL_RAW24(cb, v0, ##a)      \
    cb(24, ##a)
#define UNROLL_RAW49(cb, v0, a...)                                          \
    UNROLL_RAW25(cb, v0, ##a)                                               \
    cb(25, ##a) cb(26, ##a) cb(27, ##a) cb(28, ##a) cb(29, ##a) cb(30, ##a) \
    cb(31, ##a) cb(32, ##a) cb(33, ##a) cb(34, ##a) cb(35, ##a) cb(36, ##a) \
    cb(37, ##a) cb(38, ##a) cb(39, ##a) cb(40, ##a) cb(41, ##a) cb(42, ##a) \
    cb(43, ##a) cb(44, ##a) cb(45, ##a) cb(46, ##a) cb(47, ##a) cb(48, ##a)

#define UNROLL_CALL0(step, cb, v...) UNROLL_RAW##step(cb, 0, ##v)
#define UNROLL_CALL1(step, cb, v...) UNROLL_CALL0(step, cb, ##v)
#define UNROLL_CALL(step, cb, v...)  \
    do {                             \
        UNROLL_CALL1(step, cb, ##v); \
    } while (0)

#define UNROLL_CALL_RAW(step, cb, v...) UNROLL_CALL1(step, cb, ##v);
#define UNROLL_CALL_NOWRAPPER(step, cb) UNROLL_CALL_RAW(step, cb)

#define UNROLL_CALL0(step, cb, v...) UNROLL_RAW##step(cb, 0, ##v)
#define UNROLL_CALL1(step, cb, v...) UNROLL_CALL0(step, cb, ##v)
#define UNROLL_CALL(step, cb, v...)  \
    do {                             \
        UNROLL_CALL1(step, cb, ##v); \
    } while (0)


///////////////////// unroll with 2 dimension //////////////////////
#define UNROLL_RAW_1x1(cb, v0, a...) cb(0, 0, ##a)
#define UNROLL_RAW_2x2(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(1, 0, ##a) cb(1, 1, ##a)

#define UNROLL_RAW_3x3(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(0, 2, ##a) \
    cb(1, 0, ##a) cb(1, 1, ##a) cb(1, 2, ##a) \
    cb(2, 0, ##a) cb(2, 1, ##a) cb(2, 2, ##a) \

#define UNROLL_RAW_4x4(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(0, 2, ##a) cb(0, 3, ##a) \
    cb(1, 0, ##a) cb(1, 1, ##a) cb(1, 2, ##a) cb(1, 3, ##a) \
    cb(2, 0, ##a) cb(2, 1, ##a) cb(2, 2, ##a) cb(2, 3, ##a) \
    cb(3, 0, ##a) cb(3, 1, ##a) cb(3, 2, ##a) cb(3, 3, ##a)

#define UNROLL_RAW_6x6(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(0, 2, ##a) cb(0, 3, ##a) \
    cb(0, 4, ##a) cb(0, 5, ##a)                             \
    cb(1, 0, ##a) cb(1, 1, ##a) cb(1, 2, ##a) cb(1, 3, ##a) \
    cb(1, 4, ##a) cb(1, 5, ##a)                             \
    cb(2, 0, ##a) cb(2, 1, ##a) cb(2, 2, ##a) cb(2, 3, ##a) \
    cb(2, 4, ##a) cb(2, 5, ##a)                             \
    cb(3, 0, ##a) cb(3, 1, ##a) cb(3, 2, ##a) cb(3, 3, ##a) \
    cb(3, 4, ##a) cb(3, 5, ##a)                             \
    cb(4, 0, ##a) cb(4, 1, ##a) cb(4, 2, ##a) cb(4, 3, ##a) \
    cb(4, 4, ##a) cb(4, 5, ##a)                             \
    cb(5, 0, ##a) cb(5, 1, ##a) cb(5, 2, ##a) cb(5, 3, ##a) \
    cb(5, 4, ##a) cb(5, 5, ##a)                             \

#define UNROLL_RAW_7x7(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(0, 2, ##a) cb(0, 3, ##a) \
    cb(0, 4, ##a) cb(0, 5, ##a) cb(0, 6, ##a)               \
    cb(1, 0, ##a) cb(1, 1, ##a) cb(1, 2, ##a) cb(1, 3, ##a) \
    cb(1, 4, ##a) cb(1, 5, ##a) cb(1, 6, ##a)               \
    cb(2, 0, ##a) cb(2, 1, ##a) cb(2, 2, ##a) cb(2, 3, ##a) \
    cb(2, 4, ##a) cb(2, 5, ##a) cb(2, 6, ##a)               \
    cb(3, 0, ##a) cb(3, 1, ##a) cb(3, 2, ##a) cb(3, 3, ##a) \
    cb(3, 4, ##a) cb(3, 5, ##a) cb(3, 6, ##a)               \
    cb(4, 0, ##a) cb(4, 1, ##a) cb(4, 2, ##a) cb(4, 3, ##a) \
    cb(4, 4, ##a) cb(4, 5, ##a) cb(4, 6, ##a)               \
    cb(5, 0, ##a) cb(5, 1, ##a) cb(5, 2, ##a) cb(5, 3, ##a) \
    cb(5, 4, ##a) cb(5, 5, ##a) cb(5, 6, ##a)               \
    cb(6, 0, ##a) cb(6, 1, ##a) cb(6, 2, ##a) cb(6, 3, ##a) \
    cb(6, 4, ##a) cb(6, 5, ##a) cb(6, 6, ##a)               \


#define UNROLL_RAW_8x8(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(0, 2, ##a) cb(0, 3, ##a) \
    cb(0, 4, ##a) cb(0, 5, ##a) cb(0, 6, ##a) cb(0, 7, ##a) \
    cb(1, 0, ##a) cb(1, 1, ##a) cb(1, 2, ##a) cb(1, 3, ##a) \
    cb(1, 4, ##a) cb(1, 5, ##a) cb(1, 6, ##a) cb(1, 7, ##a) \
    cb(2, 0, ##a) cb(2, 1, ##a) cb(2, 2, ##a) cb(2, 3, ##a) \
    cb(2, 4, ##a) cb(2, 5, ##a) cb(2, 6, ##a) cb(2, 7, ##a) \
    cb(3, 0, ##a) cb(3, 1, ##a) cb(3, 2, ##a) cb(3, 3, ##a) \
    cb(3, 4, ##a) cb(3, 5, ##a) cb(3, 6, ##a) cb(3, 7, ##a) \
    cb(4, 0, ##a) cb(4, 1, ##a) cb(4, 2, ##a) cb(4, 3, ##a) \
    cb(4, 4, ##a) cb(4, 5, ##a) cb(4, 6, ##a) cb(4, 7, ##a) \
    cb(5, 0, ##a) cb(5, 1, ##a) cb(5, 2, ##a) cb(5, 3, ##a) \
    cb(5, 4, ##a) cb(5, 5, ##a) cb(5, 6, ##a) cb(5, 7, ##a) \
    cb(6, 0, ##a) cb(6, 1, ##a) cb(6, 2, ##a) cb(6, 3, ##a) \
    cb(6, 4, ##a) cb(6, 5, ##a) cb(6, 6, ##a) cb(6, 7, ##a) \
    cb(7, 0, ##a) cb(7, 1, ##a) cb(7, 2, ##a) cb(7, 3, ##a) \
    cb(7, 4, ##a) cb(7, 5, ##a) cb(7, 6, ##a) cb(7, 7, ##a)

#define UNROLL_RAW_9x9(cb, v0, a...) \
    cb(0, 0, ##a) cb(0, 1, ##a) cb(0, 2, ##a) cb(0, 3, ##a)               \
    cb(0, 4, ##a) cb(0, 5, ##a) cb(0, 6, ##a) cb(0, 7, ##a) cb(0, 8, ##a) \
    cb(1, 0, ##a) cb(1, 1, ##a) cb(1, 2, ##a) cb(1, 3, ##a)               \
    cb(1, 4, ##a) cb(1, 5, ##a) cb(1, 6, ##a) cb(1, 7, ##a) cb(1, 8, ##a) \
    cb(2, 0, ##a) cb(2, 1, ##a) cb(2, 2, ##a) cb(2, 3, ##a)               \
    cb(2, 4, ##a) cb(2, 5, ##a) cb(2, 6, ##a) cb(2, 7, ##a) cb(2, 8, ##a) \
    cb(3, 0, ##a) cb(3, 1, ##a) cb(3, 2, ##a) cb(3, 3, ##a)               \
    cb(3, 4, ##a) cb(3, 5, ##a) cb(3, 6, ##a) cb(3, 7, ##a) cb(3, 8, ##a) \
    cb(4, 0, ##a) cb(4, 1, ##a) cb(4, 2, ##a) cb(4, 3, ##a)               \
    cb(4, 4, ##a) cb(4, 5, ##a) cb(4, 6, ##a) cb(4, 7, ##a) cb(4, 8, ##a) \
    cb(5, 0, ##a) cb(5, 1, ##a) cb(5, 2, ##a) cb(5, 3, ##a)               \
    cb(5, 4, ##a) cb(5, 5, ##a) cb(5, 6, ##a) cb(5, 7, ##a) cb(5, 8, ##a) \
    cb(6, 0, ##a) cb(6, 1, ##a) cb(6, 2, ##a) cb(6, 3, ##a)               \
    cb(6, 4, ##a) cb(6, 5, ##a) cb(6, 6, ##a) cb(6, 7, ##a) cb(6, 8, ##a) \
    cb(7, 0, ##a) cb(7, 1, ##a) cb(7, 2, ##a) cb(7, 3, ##a)               \
    cb(7, 4, ##a) cb(7, 5, ##a) cb(7, 6, ##a) cb(7, 7, ##a) cb(7, 8, ##a) \
    cb(8, 0, ##a) cb(8, 1, ##a) cb(8, 2, ##a) cb(8, 3, ##a)               \
    cb(8, 4, ##a) cb(8, 5, ##a) cb(8, 6, ##a) cb(8, 7, ##a) cb(8, 8, ##a)

#define UNROLL_CALL0_D2(step, step2, cb, v...) \
    UNROLL_RAW_##step##x##step2(cb, 0, ##v)
#define UNROLL_CALL1_D2(step, step2, cb, v...) \
    UNROLL_CALL0_D2(step, step2, cb, ##v)
#define UNROLL_CALL_D2(step, step2, cb, v...)  \
    do {                                       \
        UNROLL_CALL1_D2(step, step2, cb, ##v); \
    } while (0)

#define UNROLL_CALL_RAW_D2(step, step2, cb, v...) \
    UNROLL_CALL1_D2(step, step2, cb, ##v);
#define UNROLL_CALL_NOWRAPPER_D2(step, step2, cb) \
    UNROLL_CALL_RAW_D2(step, step2, cb)

// clang-format on

// vim: syntax=cpp.doxygen
