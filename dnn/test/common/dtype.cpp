/**
 * \file dnn/test/common/dtype.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/dtype.h"
#include <gtest/gtest.h>
#include "test/common/fix_gtest_on_platforms_without_exception.inl"

TEST(TestDType, SizeCheck) {
    ASSERT_EQ(static_cast<size_t>(1), ::megdnn::dtype::Int8().size());
    ASSERT_EQ(static_cast<size_t>(1), ::megdnn::dtype::IntB2().size(1));
    ASSERT_EQ(static_cast<size_t>(2), ::megdnn::dtype::IntB2().size(5));
    ASSERT_EQ(static_cast<size_t>(12), ::megdnn::dtype::Int32().size(3));
    ASSERT_EQ(static_cast<size_t>(2), ::megdnn::dtype::UintB4().size(3));
    ASSERT_EQ(static_cast<size_t>(2), ::megdnn::dtype::UintB4().size(4));
    ASSERT_EQ(static_cast<size_t>(2), ::megdnn::dtype::IntB4().size(3));
    ASSERT_EQ(static_cast<size_t>(2), ::megdnn::dtype::IntB4().size(4));
    ASSERT_EQ(static_cast<size_t>(3), ::megdnn::dtype::IntB4().size(5));
    ASSERT_EQ(static_cast<size_t>(2),
              ::megdnn::dtype::Quantized4Asymm(1.0f, static_cast<uint8_t>(12))
                      .size(3));
    ASSERT_EQ(static_cast<size_t>(2),
              ::megdnn::dtype::Quantized4Asymm(2.f, static_cast<uint8_t>(1))
                      .size(4));
    ASSERT_EQ(static_cast<size_t>(2),
              ::megdnn::dtype::QuantizedS4(0.1f).size(3));
    ASSERT_EQ(static_cast<size_t>(2),
              ::megdnn::dtype::QuantizedS4(2.f).size(4));
    ASSERT_EQ(static_cast<size_t>(3),
              ::megdnn::dtype::QuantizedS4(0.086f).size(5));
}

TEST(TestDType, TestQuantized8Asymm) {
    using namespace megdnn;

    dtype::Quantized8Asymm q8(0.1f, static_cast<uint8_t>(128));
    EXPECT_EQ(q8.size(7), 7u);
    EXPECT_FLOAT_EQ(q8.param().scale, 0.1f);
    EXPECT_EQ(q8.param().zero_point, 128);

    dtype::Quantized8Asymm q8_copy = q8;
    EXPECT_NO_THROW(q8_copy.assert_is(q8));
    EXPECT_FLOAT_EQ(q8_copy.param().scale, 0.1f);
    EXPECT_EQ(q8_copy.param().zero_point, static_cast<uint8_t>(128));

    dtype::Quantized8Asymm q8_reconstruct_with_same_param(
            0.1f, static_cast<uint8_t>(128));
    EXPECT_NO_THROW(q8_reconstruct_with_same_param.assert_is(q8));

    dtype::Quantized8Asymm q8_diff_zp(0.1f, static_cast<uint8_t>(233));
    EXPECT_ANY_THROW(q8_diff_zp.assert_is(q8));

    dtype::Quantized8Asymm q8_diff_scale(0.1f + 1e-5f,
                                         static_cast<uint8_t>(128));
    EXPECT_ANY_THROW(q8_diff_scale.assert_is(q8));

    DType parent = q8;
    ASSERT_NO_THROW(dtype::Quantized8Asymm::downcast_from(parent));
    auto param = dtype::Quantized8Asymm::downcast_from(parent).param();
    EXPECT_FLOAT_EQ(param.scale, 0.1f);
    EXPECT_EQ(param.zero_point, 128);

    EXPECT_ANY_THROW(dtype::Quantized8Asymm::downcast_from(dtype::Int8()));
    EXPECT_ANY_THROW(DType::from_enum(DTypeEnum::Quantized8Asymm));
}

TEST(TestDType, TestQuantizedS4) {
    using namespace megdnn;

    dtype::QuantizedS4 qint4(0.1f);
    EXPECT_EQ(qint4.size(7), 4u);
    EXPECT_FLOAT_EQ(qint4.param().scale, 0.1f);

    dtype::QuantizedS4 qint4_copy = qint4;
    EXPECT_NO_THROW(qint4_copy.assert_is(qint4));
    EXPECT_FLOAT_EQ(qint4_copy.param().scale, 0.1f);

    dtype::QuantizedS4 qint4_reconstruct_with_same_param(0.1f);
    EXPECT_NO_THROW(qint4_reconstruct_with_same_param.assert_is(qint4));

    dtype::QuantizedS4 qint4_diff(0.2f);
    EXPECT_ANY_THROW(qint4_diff.assert_is(qint4));

    DType parent = qint4;
    ASSERT_NO_THROW(dtype::QuantizedS4::downcast_from(parent));
    auto param = dtype::QuantizedS4::downcast_from(parent).param();
    EXPECT_FLOAT_EQ(param.scale, 0.1f);

    EXPECT_ANY_THROW(dtype::QuantizedS4::downcast_from(dtype::IntB4()));
    EXPECT_ANY_THROW(DType::from_enum(DTypeEnum::QuantizedS4));
}

// vim: syntax=cpp.doxygen
