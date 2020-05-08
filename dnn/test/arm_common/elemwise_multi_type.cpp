/**
 * \file dnn/test/arm_common/elemwise_multi_type.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/elemwise_multi_type.h"
#include "megdnn/oprs.h"
#include "test/arm_common/fixture.h"
#include "test/common/checker.h"
#include "test/common/timer.h"
#include "test/common/workspace_wrapper.h"

using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class ARM_COMMON_ELEMWISE_MULTI_TYPE : public ARM_COMMON {};
TYPED_TEST_CASE(ARM_COMMON_ELEMWISE_MULTI_TYPE,
                elemwise_multi_type::test_types);
}  // anonymous namespace

TYPED_TEST(ARM_COMMON_ELEMWISE_MULTI_TYPE, run) {
    elemwise_multi_type::run_test<TypeParam>(this->handle());
}

TEST_F(ARM_COMMON, ELEMWISE_QUANTIZED_MODE_UNARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle());

    std::unique_ptr<RNG> rng;
    for (auto mode : {Mode::QRELU, Mode::QABS, Mode::QSIGMOID, Mode::QEXP,
                      Mode::QTANH, Mode::QFAST_TANH, Mode::QH_SWISH}) {
        checker.set_param({mode});

        for (DType src_type : std::vector<DType>{
                     dtype::QuantizedS8(1.4f),
                     dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(4)),
                     dtype::QuantizedS32(1.3f)}) {
            checker.set_dtype(0, src_type);
            if (src_type.enumv() == DTypeEnum::QuantizedS8) {
                rng = std::make_unique<UniformIntRNG>(-127, 127);
                checker.set_dtype(1, dtype::QuantizedS8(1.7f));
            } else if (src_type.enumv() == DTypeEnum::Quantized8Asymm) {
                rng = std::make_unique<UniformIntRNG>(0, 255);
                checker.set_dtype(1, dtype::Quantized8Asymm(
                                             1.7f, static_cast<uint8_t>(10)));
            } else {
                rng = std::make_unique<UniformIntRNG>(INT16_MIN >> 1,
                                                      INT16_MAX >> 1);
            }

            checker.set_rng(0, rng.get());
            auto run = [&]() {
                checker.execs({{3, 4, 5, 6}, {}});

                checker.execs({{3}, {}});
                checker.execs({{9}, {}});
                checker.execs({{17}, {}});
            };

            if (src_type.enumv() == DTypeEnum::QuantizedS32) {
                for (DType dst_type : std::vector<DType>{
                             dtype::QuantizedS8(32718.6f),
                             dtype::Quantized8Asymm(
                                     32729.6f, static_cast<uint8_t>(128))}) {
                    checker.set_dtype(1, dst_type);
                    run();
                }
            } else {
                run();
            }
        }
    }
}

TEST_F(ARM_COMMON, ELEMWISE_QUANTIZED_MODE_BINARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle());
    auto run = [&]() {
        //! nchw44
        checker.execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
        //! VEC + SCALAR
        checker.execs({{3, 4, 5, 6}, {1, 1, 1, 1}, {}});
        checker.execs({{1, 1, 1, 1}, {3, 4, 5, 6}, {}});

        //! VEC + 1C11
        checker.execs({{3, 4, 5, 6}, {1, 4, 1, 1}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 6}, {}});

        //! VEC + VEC
        checker.execs({{3}, {3}, {}});
        checker.execs({{9}, {9}, {}});
        checker.execs({{17}, {17}, {}});
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {}});
    };

    // qint32 to qint8/quint8
    for (auto mode :
         {Mode::QADD, Mode::QFUSE_ADD_RELU, Mode::QFUSE_ADD_H_SWISH}) {
        checker.set_param({mode});
        UniformIntRNG rng{INT16_MIN >> 1, INT16_MAX >> 1};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_dtype(0, dtype::QuantizedS32(1.3f))
                .set_dtype(1, dtype::QuantizedS32(1.2f));

        for (DType dst_type :
             std::vector<DType>{dtype::QuantizedS8(32718.6f),
                                dtype::Quantized8Asymm(
                                        32729.6f, static_cast<uint8_t>(128))}) {
            checker.set_dtype(2, dst_type);
            run();
        }
    }

    for (auto mode : {Mode::QMUL, Mode::QADD, Mode::QMIN, Mode::QMAX,
                      Mode::QSUB, Mode::QFUSE_ADD_RELU, Mode::QFUSE_ADD_SIGMOID,
                      Mode::QFUSE_ADD_H_SWISH}) {
        checker.set_param({mode});

        // qint8 to qint8
        UniformIntRNG rng_int8{-127, 127};
        checker.set_rng(0, &rng_int8)
                .set_rng(1, &rng_int8)
                .set_dtype(0, dtype::QuantizedS8(1.35f))
                .set_dtype(1, dtype::QuantizedS8(1.15f))
                .set_dtype(2, dtype::QuantizedS8(1.75f));

        run();
        // quint8 to quint8
        UniformIntRNG rng_uint8{0, 255};
        checker.set_rng(0, &rng_uint8)
                .set_rng(1, &rng_uint8)
                .set_dtype(0, dtype::Quantized8Asymm(1.35f,
                                                     static_cast<uint8_t>(128)))
                .set_dtype(1, dtype::Quantized8Asymm(1.15f,
                                                     static_cast<uint8_t>(128)))
                .set_dtype(2, dtype::Quantized8Asymm(
                                      1.75f, static_cast<uint8_t>(128)));

        run();
    }

    //! TRUE_DIV : 0.0 / 0.0 will fail
    checker.set_param({Mode::QTRUE_DIV});
    UniformIntRNG rng_int8_1{-127, 127};
    UniformIntRNG rng_int8_2{-127, -1};
    checker.set_rng(0, &rng_int8_1)
            .set_rng(1, &rng_int8_2)
            .set_dtype(0, dtype::QuantizedS8(1.4f))
            .set_dtype(1, dtype::QuantizedS8(1.1f))
            .set_dtype(2, dtype::QuantizedS8(1.7f));

    run();

    // quint8 to quint8
    UniformIntRNG rng_uint8_1{0, 255};
    UniformIntRNG rng_uint8_2{0, 127};
    checker.set_rng(0, &rng_uint8_1)
            .set_rng(1, &rng_uint8_2)
            .set_dtype(0,
                       dtype::Quantized8Asymm(1.35f, static_cast<uint8_t>(128)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128)))
            .set_dtype(2, dtype::Quantized8Asymm(1.75f,
                                                 static_cast<uint8_t>(128)));

    run();

    //! TANH
    checker.set_param({Mode::QFUSE_ADD_TANH});
    UniformIntRNG rng_int8{-5, 5};
    checker.set_rng(0, &rng_int8)
            .set_rng(1, &rng_int8)
            .set_dtype(0, dtype::QuantizedS8(1.1f))
            .set_dtype(1, dtype::QuantizedS8(1.4f))
            .set_dtype(2, dtype::QuantizedS8(1.7f));

    run();

    UniformIntRNG rng_uint8{123, 133};
    checker.set_rng(0, &rng_uint8)
            .set_rng(1, &rng_uint8)
            .set_dtype(0,
                       dtype::Quantized8Asymm(1.1f, static_cast<uint8_t>(128)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(1.4f, static_cast<uint8_t>(128)))
            .set_dtype(2,
                       dtype::Quantized8Asymm(1.7f, static_cast<uint8_t>(128)));

    run();
}

TEST_F(ARM_COMMON, ELEMWISE_QUANTIZED_MODE_TERNARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle());

    auto run = [&]() {
        //! nchw44
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});

        //! nchw44
        checker.execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});

        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {1, 1, 1, 1}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 6}, {1, 4, 1, 1}, {}});

        checker.execs({{3}, {3}, {3}, {}});
        checker.execs({{9}, {9}, {9}, {}});
        checker.execs({{17}, {17}, {17}, {}});
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {3, 4, 5, 6}, {}});
    };

    for (auto mode : {Mode::QFUSE_MUL_ADD3}) {
        checker.set_param({mode});

        // qint8 to qint8
        UniformIntRNG rng_int8{-127, 127};
        checker.set_rng(0, &rng_int8)
                .set_rng(1, &rng_int8)
                .set_rng(2, &rng_int8)
                .set_dtype(0, dtype::QuantizedS8(1.45f))
                .set_dtype(1, dtype::QuantizedS8(1.15f))
                .set_dtype(2, dtype::QuantizedS8(1.75f))
                .set_dtype(3, dtype::QuantizedS8(1.35f));
        run();

        // quint8 to quint8
        UniformIntRNG rng_uint8{0, 225};
        checker.set_rng(0, &rng_uint8)
                .set_rng(1, &rng_uint8)
                .set_rng(2, &rng_uint8)
                .set_dtype(0, dtype::Quantized8Asymm(1.35f,
                                                     static_cast<uint8_t>(128)))
                .set_dtype(1, dtype::Quantized8Asymm(1.15f,
                                                     static_cast<uint8_t>(128)))
                .set_dtype(2, dtype::Quantized8Asymm(1.75f,
                                                     static_cast<uint8_t>(128)))
                .set_dtype(3, dtype::Quantized8Asymm(
                                      1.45f, static_cast<uint8_t>(128)));
        run();
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
