/**
 * \file src/core/test/dtype.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/dtype.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestDType, DTypeScalarGetterAndSetter) {
    DTypeScalar vf{1.2f};
    ASSERT_EQ(vf.dtype(), dtype::Float32());
    DTypeScalar vi{1};
    ASSERT_EQ(vi.dtype(), dtype::Int32());

    ASSERT_EQ(1.2f, vf.get<dt_float32>());
    ASSERT_EQ(1, vi.get<dt_int32>());

    ASSERT_THROW(vf.get<dt_int32>(), MegDNNError);
    ASSERT_THROW(vi.get<dt_float32>(), MegDNNError);

    ASSERT_EQ(1, vf.get_cast<dt_int32>());
    ASSERT_EQ(1.f, vi.get_cast<dt_float32>());

    ASSERT_EQ(vf, vf);
    ASSERT_EQ(vf, DTypeScalar{1.2f});
    ASSERT_EQ(vi, vi);
    ASSERT_EQ(vi, DTypeScalar{1});
    ASSERT_NE(vf, vi);
    ASSERT_NE(vf, DTypeScalar{1.21f});
    ASSERT_NE(vi, DTypeScalar{2});
    ASSERT_NE(vi, DTypeScalar{1.f});
}

TEST(TestDType, DTypeScalarSetRetain) {
    DTypeScalar v{dtype::Int32()};
    v.set_retain_dtype(2);
    ASSERT_EQ(2, v.get<dt_int32>());

    v.set_retain_dtype(2.3f);
    ASSERT_EQ(2, v.get<dt_int32>());
}

TEST(TestDType, StaticCast) {
    HostTensorGenerator<> gen;
    auto v0 = gen({4});
    int v1[4];
    static_cast_dtype(v1, v0->dtype(), v0->raw_ptr(), 4);
    for (int i = 0; i < 4; ++ i)
        ASSERT_EQ(static_cast<int>(v0->ptr<float>()[i]), v1[i]);
}

TEST(TestDType, StaticCastLowbit2F) {
    megdnn::dt_byte v0;
    v0.as<int8_t>()[0] = 0xFF;
    float v1[8];
    static_cast_dtype(v1, dtype::IntB1(), &v0, 8);
    for(int i = 0; i < 8; ++ i)
        ASSERT_EQ(1, v1[i]);
    static_cast_dtype(v1, dtype::IntB2(), &v0, 4);
    for(int i = 0; i < 4; ++ i)
        ASSERT_EQ(3, v1[i]);
    static_cast_dtype(v1, dtype::IntB4(), &v0, 2);
    for(int i = 0; i < 2; ++ i)
        ASSERT_EQ(15, v1[i]);
}

TEST(TestDType, StaticCastSafeF2I) {
    HostTensorND v0{CompNode::default_cpu(), {4}, dtype::Float32()};
    for (int i = 0; i < 4; ++ i)
        v0.ptr<float>()[i] = i;

    int v1[4];
    static_cast_dtype_safe(v1, v0.dtype(), v0.raw_ptr(), 4);
    for (int i = 0; i < 4; ++ i)
        ASSERT_EQ(i, v1[i]);

    v0.ptr<float>()[3] += 0.1;
    ASSERT_THROW(static_cast_dtype_safe(v1, v0.dtype(), v0.raw_ptr(), 4),
            MegBrainError);
}

TEST(TestDType, StaticCastSafeI2F) {
    HostTensorND v0{CompNode::default_cpu(), {4}, dtype::Int32()};
    for (int i = 0; i < 4; ++ i)
        v0.ptr<dt_int32>()[i] = i;

    dt_float32 v1[4];
    static_cast_dtype_safe(v1, v0.dtype(), v0.raw_ptr(), 4);
    for (int i = 0; i < 4; ++ i)
        ASSERT_EQ(static_cast<float>(i), v1[i]);

    v0.ptr<dt_int32>()[0] = 1 << 25;
    ASSERT_THROW(static_cast_dtype_safe(v1, v0.dtype(), v0.raw_ptr(), 4),
            MegBrainError);

    size_t v2[4];
    static_cast_dtype_safe(v2, v0.dtype(), v0.raw_ptr(), 4);
    for (int i = 0; i < 4; ++ i)
        ASSERT_EQ(static_cast<size_t>(v0.ptr<dt_int32>()[i]), v2[i]);
}

TEST(TestDType, Intb1Memcpy) {
    uint8_t compact = 0;
    int8_t byte[2] = {-1, 1};
    lowbit_memcpy_byte2compact(dtype::IntB1(), &compact, byte, 2);
    ASSERT_EQ(0x02, compact);

    int8_t byte_orig[2];
    for (int i = 0; i < 2; ++ i) {
        byte_orig[i] = byte[i];
        byte[i] = 0xFF;
    }
    lowbit_memcpy_compact2byte(dtype::IntB1(), byte, &compact, 2);
    for (int i = 0; i < 2; ++ i)
        ASSERT_EQ(byte_orig[i], byte[i]);
}

TEST(TestDType, Intb2Memcpy) {
    uint8_t compact[2];
    int8_t byte[7] = {-3, -1, 1, 3, 3, 1, -1};
    lowbit_memcpy_byte2compact(dtype::IntB2(), compact, byte, 7);
    ASSERT_EQ(0xE4, compact[0]);
    ASSERT_EQ(0x1B, compact[1]);

    int8_t byte_orig[7];
    for (int i = 0; i < 7; ++ i) {
        byte_orig[i] = byte[i];
        byte[i] = 0xFF;
    }
    lowbit_memcpy_compact2byte(dtype::IntB2(), byte, compact, 7);
    for (int i = 0; i < 7; ++ i)
        ASSERT_EQ(byte_orig[i], byte[i]);
}

TEST(TestDType, Intb4Memcpy) {
    uint8_t compact[5] = {0, 0, 0, 0, 0};
    int8_t byte[9] = {-15, 15, -1, 3, 7, 7, -9, 5, 1};
    lowbit_memcpy_byte2compact(dtype::IntB4(), compact, byte, 9);
    ASSERT_EQ(0xF0, compact[0]);
    ASSERT_EQ(0x97, compact[1]);
    ASSERT_EQ(0xBB, compact[2]);
    ASSERT_EQ(0xA3, compact[3]);
    ASSERT_EQ(0x08, compact[4]);

    int8_t byte_orig[9];
    for (int i = 0; i < 9; ++ i) {
        byte_orig[i] = byte[i];
        byte[i] = 0xFF;
    }
    lowbit_memcpy_compact2byte(dtype::IntB4(), byte, &compact, 9);
    for (int i = 0; i < 2; ++ i)
        ASSERT_EQ(byte_orig[i], byte[i]);
}

TEST(TestDType, QuantizedDTypePromotion) {
    DType lhs, rhs;

    // QuantizedS8: Allow < 1e-6 difference in scale
    lhs = dtype::QuantizedS8(0.123f);
    rhs = dtype::QuantizedS8(0.123f+1e-9f);
    EXPECT_EQ(mgb::dtype_promotion(lhs, rhs), lhs);

    // QuantizedS8: Disallow too big difference in scale
    lhs = dtype::QuantizedS8(0.123f);
    rhs = dtype::QuantizedS8(0.456f);
    EXPECT_THROW(mgb::dtype_promotion(lhs, rhs), AssertionError);

    // Quantized8Asymm: Allow < 1e-6 difference in scale
    lhs = dtype::Quantized8Asymm(0.123f, (uint8_t)127);
    rhs = dtype::Quantized8Asymm(0.123f+1e-9f, (uint8_t)127);
    EXPECT_EQ(mgb::dtype_promotion(lhs, rhs), lhs);

    // Quantized8Asymm: Disallow different zero_point
    lhs = dtype::Quantized8Asymm(0.123f, (uint8_t)127);
    rhs = dtype::Quantized8Asymm(0.123f, (uint8_t)128);
    EXPECT_THROW(mgb::dtype_promotion(lhs, rhs), AssertionError);

    // Quantized8Asymm: Disallow too big difference in scale
    lhs = dtype::Quantized8Asymm(0.123f, (uint8_t)0);
    rhs = dtype::Quantized8Asymm(0.456f, (uint8_t)0);
    EXPECT_THROW(mgb::dtype_promotion(lhs, rhs), AssertionError);

    // QuantizedS32: Allow < 1e-6 difference in scale
    lhs = dtype::QuantizedS32(0.123f);
    rhs = dtype::QuantizedS32(0.123f+1e-9f);
    EXPECT_EQ(mgb::dtype_promotion(lhs, rhs), lhs);

    // QuantizedS32: Disallow too big difference in scale
    lhs = dtype::QuantizedS32(0.123f);
    rhs = dtype::QuantizedS32(0.456f);
    EXPECT_THROW(mgb::dtype_promotion(lhs, rhs), AssertionError);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

