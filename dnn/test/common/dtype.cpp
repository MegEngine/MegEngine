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
#include "megdnn/dtype/bfloat16.hpp"
#include <gtest/gtest.h>
#include "test/common/fix_gtest_on_platforms_without_exception.inl"

#include <limits>

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
    ASSERT_EQ(static_cast<size_t>(2), ::megdnn::dtype::Uint16().size(1));
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

TEST(TestDType, BFLOAT16) {
    using namespace megdnn;
    using namespace half_bfloat16;
    //! Basic bfloat16 dtype tests using RNE round.
    bfloat16 m1(2.3515625f), m2(2.351563f), m3(229), m4(-311);
    ASSERT_FLOAT_EQ(static_cast<float>(m1), 2.34375f);
    ASSERT_FLOAT_EQ(static_cast<float>(m2), 2.359375f);
    ASSERT_FLOAT_EQ(static_cast<float>(m3), 229.f);
    ASSERT_FLOAT_EQ(static_cast<float>(m4), -312.f);
    m3 = -2.3515625f;
    m4 = -2.351563f;
    ASSERT_FLOAT_EQ(static_cast<float>(m1), static_cast<float>(-m3));
    ASSERT_FLOAT_EQ(static_cast<float>(m2), static_cast<float>(-m4));
    m3 = 2.34375f;
    m3 += m2;
    m4 = m1;
    m4 *= m2;
    ASSERT_FLOAT_EQ(static_cast<float>(m3), 4.6875f);
    ASSERT_FLOAT_EQ(static_cast<float>(m4), 5.53125f);
    m3 -= 2.359375f;
    m4 /= 2.359375f;
    ASSERT_FLOAT_EQ(static_cast<float>(m3), 2.328125f);
    ASSERT_FLOAT_EQ(static_cast<float>(m4), 2.34375f);
    m3++;
    ++m3;
    m4++;
    ++m4;
    ASSERT_FLOAT_EQ(static_cast<float>(m3), 4.3125f);
    ASSERT_FLOAT_EQ(static_cast<float>(m4), 4.34375f);
    m3--;
    --m3;
    m4--;
    --m4;
    ASSERT_FLOAT_EQ(static_cast<float>(m3), 2.3125f);
    ASSERT_FLOAT_EQ(static_cast<float>(m4), 2.34375f);

    //! Comparison operators
    ASSERT_TRUE(m1 == m4 && m1 >= m4 && m1 <= m4);
    ASSERT_TRUE(m3 != m4 && m4 > m3);
    ASSERT_FALSE(m2 < m4);
    
    //! Arithmetic operators
    ASSERT_FLOAT_EQ(m1 + m2, 4.703125f);
    ASSERT_FLOAT_EQ(m4 - 3.43281f, -1.08906f);
    ASSERT_FLOAT_EQ(-2.34f * m3, -5.41125f);
    ASSERT_FLOAT_EQ(9.92625f / m1, 4.2352f);

    //! Basic mathematical operations
    bfloat16 b1(-0.5f), b2(0.5f), b3(7.21875);
    ASSERT_FLOAT_EQ(abs(b1), abs(b2));
    ASSERT_FLOAT_EQ(acos(b1), 2.094395f);
    ASSERT_FLOAT_EQ(acosh(b3), 2.66499658f);
    ASSERT_FLOAT_EQ(asin(b1), -0.523599f);
    ASSERT_FLOAT_EQ(asinh(b1), -0.48121183f);
    ASSERT_FLOAT_EQ(atan(b1), -0.4636476f);
    ASSERT_FLOAT_EQ(atan2(b1, b3), -0.06915362f);
    ASSERT_FLOAT_EQ(cbrt(b1), -0.79370053f);
    ASSERT_FLOAT_EQ(static_cast<float>(ceil(b1)), 0.0f);
    ASSERT_FLOAT_EQ(cos(b1), 0.87758255f);
    ASSERT_FLOAT_EQ(cosh(b1), 1.12762594f);
    ASSERT_FLOAT_EQ(erf(b1), -0.52049988f);
    ASSERT_FLOAT_EQ(erfc(b1), 1.52049988f);
    ASSERT_FLOAT_EQ(exp(b2), 1.64872122f);
    ASSERT_FLOAT_EQ(exp2(b2), 1.41421356f);
    ASSERT_FLOAT_EQ(expm1(b2), 0.64872127f);
    ASSERT_FLOAT_EQ(fdim(b2, b1), 1.0f);
    ASSERT_FLOAT_EQ(floor(b1), -1.0f);
    ASSERT_FLOAT_EQ(fma(b1, b2, b1), -0.75f);
    ASSERT_FLOAT_EQ(fmax(b1, b2), 0.5f);
    ASSERT_FLOAT_EQ(fmin(b1, b2), -0.5f);
    ASSERT_FLOAT_EQ(fmod(b3, b2), 0.21875f);
    ASSERT_FLOAT_EQ(hypot(b2, b3), 7.23604530f);
    ASSERT_FLOAT_EQ(lgamma(b1), 1.26551212f);
    ASSERT_FLOAT_EQ(log(b3), 1.97668183f);
    ASSERT_FLOAT_EQ(log10(b3), 0.85846198f);
    ASSERT_FLOAT_EQ(log1p(b3), 2.10641813f);
    ASSERT_FLOAT_EQ(log2(b3), 2.85174904f);
    ASSERT_FLOAT_EQ(lrint(b3), 7.f);
    ASSERT_EQ(lround(b1), -1);
    ASSERT_EQ(lround(b2), 1);
    ASSERT_TRUE(isnan(nanh("")));
    ASSERT_FLOAT_EQ(nearbyint(b3), 7.f);
    ASSERT_FLOAT_EQ(pow(b3, 2.53f), 148.56237793f);
    ASSERT_FLOAT_EQ(remainder(b3, b2), 0.21875f);
    ASSERT_FLOAT_EQ(sin(b1), -0.47942555f);
    ASSERT_FLOAT_EQ(sinh(b1), -0.52109528f);
    ASSERT_FLOAT_EQ(sqrt(b3), 2.68677306f);
    ASSERT_FLOAT_EQ(tan(b3), 1.35656071f);
    ASSERT_FLOAT_EQ(tanh(b3), 0.99999893f);
    ASSERT_FLOAT_EQ(tgamma(b3), 1088.50023434f);
    ASSERT_FLOAT_EQ(trunc(b1), 0.0f);
    ASSERT_FLOAT_EQ(trunc(b3), 7.0f);
    ASSERT_FLOAT_EQ(static_cast<float>(copysign(b1, b2)), 0.5f);
    int i = 0;
    ASSERT_FLOAT_EQ(static_cast<float>(frexp(b3, &i)), 0.90234375f);
    ASSERT_EQ(i, 3);
    ASSERT_EQ(ilogb(b3), 2);
    ASSERT_FLOAT_EQ(static_cast<float>(ldexp(b3, 4)), 115.50f);
    ASSERT_FLOAT_EQ(static_cast<float>(logb(b3)), 2.f);
    bfloat16 bf(0.f);
    ASSERT_FLOAT_EQ(static_cast<float>(modf(b3, &bf)), 0.21875f);
    ASSERT_FLOAT_EQ(static_cast<float>(bf), 7.f);
    ASSERT_FLOAT_EQ(static_cast<float>(nextafter(b2, b3)), 0.50390625f);
    ASSERT_FLOAT_EQ(static_cast<float>(nextafter(b2, b1)), 0.49804688f);
    ASSERT_TRUE(signbit(b1));
    ASSERT_FALSE(signbit(b2));

    //! Special(Denormal) number.
    //! flaot -> bfloat16
    float finf = std::numeric_limits<float>::infinity(),
    fnan = std::numeric_limits<float>::quiet_NaN();
    bfloat16 bfinf(finf), bfnan(fnan);
    ASSERT_TRUE(isinf(bfinf));
    ASSERT_FALSE(isfinite(bfinf));
    ASSERT_TRUE(isnan(bfnan));
    ASSERT_FALSE(isnormal(bfnan));

    //! bfloat16 -> float
    bfinf = std::numeric_limits<bfloat16>::infinity();
    finf = bfinf;
    ASSERT_TRUE(std::isinf(finf));
    ASSERT_FALSE(std::isfinite(finf));
}

// vim: syntax=cpp.doxygen
