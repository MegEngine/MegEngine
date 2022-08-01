#include <cmath>
#include <vector>
#if defined(ONLY_BUILD_GI_API)
#include <gtest/gtest.h>
class FALLBACK : public ::testing::Test {};
#else
#include "test/fallback/fixture.h"
#endif

#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"

namespace megdnn {
namespace test {

#define SIMD_LEN    GI_SIMD_LEN_BYTE / sizeof(float)
#define SIMD_LEN_16 GI_SIMD_LEN_BYTE / sizeof(int16_t)
#define SIMD_LEN_8  GI_SIMD_LEN_BYTE / sizeof(int8_t)
template <typename T>
static void init(
        T* dst, const std::vector<T>& value, const size_t simd_len = SIMD_LEN) {
    for (size_t i = 0; i < simd_len; i++) {
        dst[i] = value[i];
    }
}

template <typename T>
static void assert_eq(T* a, const std::vector<T>& b, const size_t simd_len = SIMD_LEN) {
    for (size_t i = 0; i < simd_len; i++) {
        ASSERT_EQ(a[i], b[i]);
    }
}

template <typename T>
static void assert_eq_and_nan(
        T* a, const std::vector<T>& b, const size_t simd_len = SIMD_LEN) {
    for (size_t i = 0; i < simd_len; i++) {
        if (isnan(a[i]) && isnan(b[i])) {
            continue;
        }
        ASSERT_EQ(a[i], b[i]);
    }
}

static void assert_lt(
        float* a, const std::vector<float>& b, const float eps,
        const size_t simd_len = SIMD_LEN) {
    for (size_t i = 0; i < simd_len; i++) {
        ASSERT_LT(std::abs(a[i] - b[i]), eps);
    }
}

static void force_memset_ret(void* dst, const size_t len) {
    memset(dst, 'f', len);
}

TEST_F(FALLBACK, GiGetSimdType) {
    auto t = GiGetSimdType();
    auto should_type = GI_UNKNOWN;
#if defined(GI_AVX_INTRINSICS) || defined(GI_AVX2_INTRINSICS) || \
        defined(GI_FMA_INTRINSICS)
    should_type = GI_AVX;
#elif defined(GI_NEON_INTRINSICS)
    should_type = GI_NEON;
#elif defined(GI_SSE2_INTRINSICS) || defined(GI_SSE42_INTRINSICS)

#if defined(GI_SSE42_INTRINSICS)
    should_type = GI_SSE42;
#elif defined(GI_SSE2_INTRINSICS)
    should_type = GI_SSE2;
#else
    should_type = GI_UNKNOWN;
#error "code issue happened!!"
#endif

#elif defined(GI_RVV_INTRINSICS)
    should_type = GI_RVV;

#else
    should_type = GI_NAIVE;
#endif

    printf("test GiGetSimdType: %d, should_type: %d\n", t, should_type);

    ASSERT_EQ(t, should_type);
}

TEST_F(FALLBACK, GiReinterpretInt8AsInt32) {
    GI_INT32_t ret;
    GI_INT8_t src0;
    std::vector<int8_t> s0{9,  2, -128, 127, 2, 45, 3, 0,
                           11, 2, -128, 127, 2, 55, 3, -1};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretInt8AsInt32(src0);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        int8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int8_t));
        naive.push_back(tmp);
    }

    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiGetSubVectorFloat32V2) {
    GI_FLOAT32_V2_t src0;
    GI_FLOAT32_t ret1, ret2;
    std::vector<float> s0{
            -1.0f, 2.2f, -3.4f, 4.5f, 111.0f, 12.2f, -13.4f, -44.5f,
    };
    s0.resize(SIMD_LEN * 2);
    init((float*)&src0, s0, SIMD_LEN * 2);

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorFloat32V2(src0, 0);
    ret2 = GiGetSubVectorFloat32V2(src0, 1);

    std::vector<float> naive1, naive2;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN], sizeof(float));
        naive2.push_back(tmp);
    }

    assert_eq((float*)&ret1, naive1, SIMD_LEN);
    assert_eq((float*)&ret2, naive2, SIMD_LEN);
}

TEST_F(FALLBACK, GiSetSubVectorFloat32V2) {
    GI_FLOAT32_V2_t ret;
    GI_FLOAT32_t src0, src1;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 4.5f};
    std::vector<float> s1{111.0f, 12.2f, -13.4f, -44.5f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0, SIMD_LEN);
    init((float*)&src1, s1, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    GiSetSubVectorFloat32V2(ret, 0, src0);
    GiSetSubVectorFloat32V2(ret, 1, src1);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s1[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiFloat32Type2FixLenType) {
    GI_FLOAT32_FIXLEN_t ret;
    GI_FLOAT32_t src;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 4.5f};
    s0.resize(SIMD_LEN);
    init((float*)&src, s0, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFloat32Type2FixLenType(src);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiFixLenType2GiFloat32Type) {
    GI_FLOAT32_t ret;
    GI_FLOAT32_FIXLEN_t src;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 4.5f};
    s0.resize(SIMD_LEN);
    init((float*)&src, s0, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFixLenType2GiFloat32Type(src);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiFloat32Type2FixLenV2Type) {
    GI_FLOAT32_FIXLEN_V2_t ret;
    GI_FLOAT32_V2_t src;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 4.5f, 55.1f, 99.0f, -1.9f, -5.3f};
    s0.resize(SIMD_LEN * 2);
    init((float*)&src, s0, SIMD_LEN * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    ret = GiFloat32Type2FixLenV2Type(src);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN * 2; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiFixLenType2GiFloat32V2Type) {
    GI_FLOAT32_V2_t ret;
    GI_FLOAT32_FIXLEN_V2_t src;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 4.5f, 111.0f, 12.2f, -13.4f, -44.5f};
    s0.resize(SIMD_LEN * 2);
    init((float*)&src, s0, SIMD_LEN * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    ret = GiFixLenType2GiFloat32V2Type(src);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN * 2; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiGetSubVectorFloat32V3) {
    GI_FLOAT32_V3_t src0;
    GI_FLOAT32_t ret1, ret2, ret3;
    std::vector<float> s0{-1.0f,  2.2f,   -3.4f, 4.5f,  111.0f, 12.2f,
                          -13.4f, -44.5f, 22.4f, 55.0f, -12.0f, 678.9f};
    s0.resize(SIMD_LEN * 3);
    //! rvv compiler crash when use init on type_x3, use rvv load api as a workaround
#if defined(GI_RVV_INTRINSICS)
    vfloat32m1_t t00, t10, t20;
    t00 = vle32_v_f32m1(s0.data(), SIMD_LEN);
    t10 = vle32_v_f32m1(s0.data() + SIMD_LEN, 4);
    t20 = vle32_v_f32m1(s0.data() + SIMD_LEN * 2, 4);
    src0 = vcreate_f32m1x3(t00, t10, t20);
#else
    init((float*)&src0, s0, SIMD_LEN * 3);
#endif

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret3, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorFloat32V3(src0, 0);
    ret2 = GiGetSubVectorFloat32V3(src0, 1);
    ret3 = GiGetSubVectorFloat32V3(src0, 2);

    std::vector<float> naive1, naive2, naive3;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN], sizeof(float));
        naive2.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN * 2], sizeof(float));
        naive3.push_back(tmp);
    }

    assert_eq((float*)&ret1, naive1, SIMD_LEN);
    assert_eq((float*)&ret2, naive2, SIMD_LEN);
    assert_eq((float*)&ret3, naive3, SIMD_LEN);
}

TEST_F(FALLBACK, GiSetSubVectorFloat32V3) {
    GI_FLOAT32_V3_t ret;
    GI_FLOAT32_t src0, src1, src2;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 4.5f};
    std::vector<float> s1{111.0f, 12.2f, -13.4f, -44.5f};
    std::vector<float> s2{22.4f, 55.0f, -12.0f, 678.9f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0, SIMD_LEN);
    init((float*)&src1, s1, SIMD_LEN);
    init((float*)&src2, s2, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 3);
    GiSetSubVectorFloat32V3(ret, 0, src0);
    GiSetSubVectorFloat32V3(ret, 1, src1);
    GiSetSubVectorFloat32V3(ret, 2, src2);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s1[i], sizeof(float));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s2[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 3);
}

TEST_F(FALLBACK, GiGetSubVectorFloat32V4) {
    GI_FLOAT32_V4_t src0;
    GI_FLOAT32_t ret1, ret2, ret3, ret4;
    std::vector<float> s0{-1.0f, 2.2f,  -3.4f,  4.5f,   111.0f, 12.2f, -13.4f, -44.5f,
                          22.4f, 55.0f, -12.0f, 678.9f, 2.2f,   -3.4f, 4.5f,   111.0f};
    s0.resize(SIMD_LEN * 4);
#if defined(GI_RVV_INTRINSICS)
    vfloat32m1_t t00, t10, t20, t30;
    t00 = vle32_v_f32m1(s0.data(), SIMD_LEN);
    t10 = vle32_v_f32m1(s0.data() + SIMD_LEN, 4);
    t20 = vle32_v_f32m1(s0.data() + SIMD_LEN * 2, 4);
    t30 = vle32_v_f32m1(s0.data() + SIMD_LEN * 3, 4);
    src0 = vcreate_f32m1x4(t00, t10, t20, t30);
#else
    init((float*)&src0, s0, SIMD_LEN * 4);
#endif

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret3, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret4, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorFloat32V4(src0, 0);
    ret2 = GiGetSubVectorFloat32V4(src0, 1);
    ret3 = GiGetSubVectorFloat32V4(src0, 2);
    ret4 = GiGetSubVectorFloat32V4(src0, 3);

    std::vector<float> naive1, naive2, naive3, naive4;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN], sizeof(float));
        naive2.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN * 2], sizeof(float));
        naive3.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN * 3], sizeof(float));
        naive4.push_back(tmp);
    }

    assert_eq((float*)&ret1, naive1, SIMD_LEN);
    assert_eq((float*)&ret2, naive2, SIMD_LEN);
    assert_eq((float*)&ret3, naive3, SIMD_LEN);
    assert_eq((float*)&ret4, naive4, SIMD_LEN);
}

TEST_F(FALLBACK, GiSetSubVectorFloat32V4) {
    GI_FLOAT32_V4_t ret;
    GI_FLOAT32_t src0, src1, src2, src3;
    std::vector<float> s0{-1.0f, 2.2f, -3.4f, 99.0f};
    std::vector<float> s1{4.5f, 111.0f, 12.2f, -13.4f};
    std::vector<float> s2{-44.5f, 22.4f, 55.0f, -12.0f};
    std::vector<float> s3{2.2f, -3.4f, 4.5f, 111.0f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    s3.resize(SIMD_LEN);
    init((float*)&src0, s0, SIMD_LEN);
    init((float*)&src1, s1, SIMD_LEN);
    init((float*)&src2, s2, SIMD_LEN);
    init((float*)&src3, s3, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 4);
    GiSetSubVectorFloat32V4(ret, 0, src0);
    GiSetSubVectorFloat32V4(ret, 1, src1);
    GiSetSubVectorFloat32V4(ret, 2, src2);
    GiSetSubVectorFloat32V4(ret, 3, src3);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s1[i], sizeof(float));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s2[i], sizeof(float));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s3[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 4);
}

TEST_F(FALLBACK, GiGetSubVectorInt32V2) {
    GI_INT32_V2_t src0;
    GI_INT32_t ret1, ret2;
    std::vector<int32_t> s0{1, 2, 3, 4, -4, -3, -2, -1};
    s0.resize(SIMD_LEN * 2);
    init((int32_t*)&src0, s0, SIMD_LEN * 2);

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorInt32V2(src0, 0);
    ret2 = GiGetSubVectorInt32V2(src0, 1);

    std::vector<int32_t> naive1, naive2;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN], sizeof(int32_t));
        naive2.push_back(tmp);
    }

    assert_eq((int32_t*)&ret1, naive1, SIMD_LEN);
    assert_eq((int32_t*)&ret2, naive2, SIMD_LEN);
}

TEST_F(FALLBACK, GiSetSubVectorInt32V2) {
    GI_INT32_V2_t ret;
    GI_INT32_t src0, src1;
    std::vector<int32_t> s0{1, 2, 3, 4};
    std::vector<int32_t> s1{-4, -3, -2, -1};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0, SIMD_LEN);
    init((int32_t*)&src1, s1, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    GiSetSubVectorInt32V2(ret, 0, src0);
    GiSetSubVectorInt32V2(ret, 1, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s1[i], sizeof(int32_t));
        naive.push_back(tmp);
    }

    assert_eq((int32_t*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiInt32Type2FixLenType) {
    GI_INT32_FIXLEN_t ret;
    GI_INT32_t src;
    std::vector<int32_t> s0{3, 4, -4, -3};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src, s0, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiInt32Type2FixLenType(src);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive.push_back(tmp);
    }

    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiFixLenType2GiInt32Type) {
    GI_INT32_t ret;
    GI_INT32_FIXLEN_t src;
    std::vector<int32_t> s0{2, 3, 4, -4};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src, s0, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFixLenType2GiInt32Type(src);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive.push_back(tmp);
    }

    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiUint32Type2FixLenType) {
    GI_UINT32_FIXLEN_t ret;
    GI_UINT32_t src;
    std::vector<uint32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    init((uint32_t*)&src, s0, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiUint32Type2FixLenType(src);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        uint32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(uint32_t));
        naive.push_back(tmp);
    }

    assert_eq((uint32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiFixLenType2GiUint32Type) {
    GI_UINT32_t ret;
    GI_UINT32_FIXLEN_t src;
    std::vector<uint32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    init((uint32_t*)&src, s0, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFixLenType2GiUint32Type(src);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        uint32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(uint32_t));
        naive.push_back(tmp);
    }

    assert_eq((uint32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiGetSubVectorInt32V4) {
    GI_INT32_V4_t src0;
    GI_INT32_t ret1, ret2, ret3, ret4;
    std::vector<int32_t> s0{1,  2,   3,   4,   -4, -3, -2, -1,
                            23, 456, 765, -99, 45, 99, 0,  8};
    s0.resize(SIMD_LEN * 4);
    init((int32_t*)&src0, s0, SIMD_LEN * 4);

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret3, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret4, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorInt32V4(src0, 0);
    ret2 = GiGetSubVectorInt32V4(src0, 1);
    ret3 = GiGetSubVectorInt32V4(src0, 2);
    ret4 = GiGetSubVectorInt32V4(src0, 3);

    std::vector<int32_t> naive1, naive2, naive3, naive4;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN], sizeof(int32_t));
        naive2.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN * 2], sizeof(int32_t));
        naive3.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN * 3], sizeof(int32_t));
        naive4.push_back(tmp);
    }

    assert_eq((int32_t*)&ret1, naive1, SIMD_LEN);
    assert_eq((int32_t*)&ret2, naive2, SIMD_LEN);
    assert_eq((int32_t*)&ret3, naive3, SIMD_LEN);
    assert_eq((int32_t*)&ret4, naive4, SIMD_LEN);
}

TEST_F(FALLBACK, GiSetSubVectorInt32V4) {
    GI_INT32_V4_t ret;
    GI_INT32_t src0, src1, src2, src3;
    std::vector<int32_t> s0{1, 2, 3, 4, -4};
    std::vector<int32_t> s1{3, -2, -1, 23};
    std::vector<int32_t> s2{456, 765, -99, 45};
    std::vector<int32_t> s3{45, 99, 0, 8};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    s3.resize(SIMD_LEN);
    init((int32_t*)&src0, s0, SIMD_LEN);
    init((int32_t*)&src1, s1, SIMD_LEN);
    init((int32_t*)&src2, s2, SIMD_LEN);
    init((int32_t*)&src3, s3, SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 4);
    GiSetSubVectorInt32V4(ret, 0, src0);
    GiSetSubVectorInt32V4(ret, 1, src1);
    GiSetSubVectorInt32V4(ret, 2, src2);
    GiSetSubVectorInt32V4(ret, 3, src3);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s1[i], sizeof(int32_t));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s2[i], sizeof(int32_t));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s3[i], sizeof(int32_t));
        naive.push_back(tmp);
    }

    assert_eq((int32_t*)&ret, naive, SIMD_LEN * 4);
}

TEST_F(FALLBACK, GiGetSubVectorInt16V2) {
    GI_INT16_V2_t src0;
    GI_INT16_t ret1, ret2;
    std::vector<int16_t> s0{-127, 2,  std::numeric_limits<int16_t>::max(),
                            9999, 1,  2,
                            3,    4,  1,
                            2,    3,  4,
                            -4,   -3, -2,
                            -1};
    s0.resize(SIMD_LEN_16 * 2);
    init((int16_t*)&src0, s0, SIMD_LEN_16 * 2);

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorInt16V2(src0, 0);
    ret2 = GiGetSubVectorInt16V2(src0, 1);

    std::vector<int16_t> naive1, naive2;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        int16_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int16_t));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN_16], sizeof(int16_t));
        naive2.push_back(tmp);
    }

    assert_eq((int16_t*)&ret1, naive1, SIMD_LEN_16);
    assert_eq((int16_t*)&ret2, naive2, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiSetSubVectorInt16V2) {
    GI_INT16_V2_t ret;
    GI_INT16_t src0, src1;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    std::vector<int16_t> s1{1, 2, 3, 4, -4, -3, -2, -1};
    s0.resize(SIMD_LEN_16);
    s1.resize(SIMD_LEN_16);
    init((int16_t*)&src0, s0, SIMD_LEN_16);
    init((int16_t*)&src1, s1, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    GiSetSubVectorInt16V2(ret, 0, src0);
    GiSetSubVectorInt16V2(ret, 1, src1);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        int16_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int16_t));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        int16_t tmp;
        memcpy(&tmp, &s1[i], sizeof(int16_t));
        naive.push_back(tmp);
    }

    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16 * 2);
}

TEST_F(FALLBACK, GiInt16Type2FixLenType) {
    GI_INT16_t src;
    GI_INT16_FIXLEN_t ret;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    s0.resize(SIMD_LEN_16);
    init((int16_t*)&src, s0, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiInt16Type2FixLenType(src);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        int16_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int16_t));
        naive.push_back(tmp);
    }

    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiFixLenType2GiInt16Type) {
    GI_INT16_FIXLEN_t src;
    GI_INT16_t ret;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    s0.resize(SIMD_LEN_16);
    init((int16_t*)&src, s0, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFixLenType2GiInt16Type(src);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        int16_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int16_t));
        naive.push_back(tmp);
    }

    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiGetSubVectorInt8V2) {
    GI_INT8_V2_t src0;
    GI_INT8_t ret1, ret2;
    std::vector<int8_t> s0{127,  2,  56,  -128, 1,  2,    3,  4,  127,  2,   56,
                           -128, 1,  2,   3,    4,  127,  2,  56, -128, -14, -22,
                           3,    -4, 127, -22,  56, -128, -1, 2,  -3,   44};
    s0.resize(SIMD_LEN_8 * 2);
    init((int8_t*)&src0, s0, SIMD_LEN_8 * 2);

    force_memset_ret((void*)&ret1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&ret2, GI_SIMD_LEN_BYTE);
    ret1 = GiGetSubVectorInt8V2(src0, 0);
    ret2 = GiGetSubVectorInt8V2(src0, 1);

    std::vector<int8_t> naive1, naive2;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        int8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int8_t));
        naive1.push_back(tmp);

        memcpy(&tmp, &s0[i + SIMD_LEN_8], sizeof(int8_t));
        naive2.push_back(tmp);
    }

    assert_eq((int8_t*)&ret1, naive1, SIMD_LEN_8);
    assert_eq((int8_t*)&ret2, naive2, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiSetSubVectorInt8V2) {
    GI_INT8_V2_t ret;
    GI_INT8_t src0, src1;
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    std::vector<int8_t> s1{127, 2,   56, -128, -14, -22, 3,  -4,
                           127, -22, 56, -128, -1,  2,   -3, 44};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    GiSetSubVectorInt8V2(ret, 0, src0);
    GiSetSubVectorInt8V2(ret, 1, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        int8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int8_t));
        naive.push_back(tmp);
    }
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        int8_t tmp;
        memcpy(&tmp, &s1[i], sizeof(int8_t));
        naive.push_back(tmp);
    }

    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8 * 2);
}

TEST_F(FALLBACK, GiUint8Type2FixLenType) {
    GI_UINT8_FIXLEN_t ret;
    GI_UINT8_t src;
    std::vector<uint8_t> s0{127, 2, 56, 255, 1, 2, 3, 4, 127, 2, 56, 0, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((uint8_t*)&src, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiUint8Type2FixLenType(src);

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        uint8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(uint8_t));
        naive.push_back(tmp);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiFixLenType2GiUint8Type) {
    GI_UINT8_t ret;
    GI_UINT8_FIXLEN_t src;
    std::vector<uint8_t> s0{127, 2, 56, 255, 1, 2, 3, 4, 127, 2, 56, 0, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((uint8_t*)&src, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFixLenType2GiUint8Type(src);

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        uint8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(uint8_t));
        naive.push_back(tmp);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiInt8Type2FixLenType) {
    GI_INT8_FIXLEN_t ret;
    GI_INT8_t src;
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, 0, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiInt8Type2FixLenType(src);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        int8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int8_t));
        naive.push_back(tmp);
    }

    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiFixLenType2GiInt8Type) {
    GI_INT8_t ret;
    GI_INT8_FIXLEN_t src;
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, 0, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiFixLenType2GiInt8Type(src);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        int8_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int8_t));
        naive.push_back(tmp);
    }

    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiAndInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    std::vector<int32_t> s1{5, 6, 7, 8};
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAndInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] & s1[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiOrInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    std::vector<int32_t> s1{5, 6, 7, 8};
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiOrInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] | s1[i]);
    }

    assert_eq((int*)&ret, naive);
}

TEST_F(FALLBACK, GiAndNotInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    std::vector<int32_t> s1{5, 6, 7, 8};
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAndNotInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(~s0[i] & s1[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiXorInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    std::vector<int32_t> s1{5, 6, 7, 8};
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiXorInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] ^ s1[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiBroadcastFloat32) {
    GI_FLOAT32_t ret;
    float b = 2022.0420;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBroadcastFloat32(b);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(b);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiBroadcastInt32) {
    GI_INT32_t ret;
    int32_t b = 20220420;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBroadcastInt32(b);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(b);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiBroadcastInt8) {
    GI_INT8_t ret;
    int8_t b = 6;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBroadcastInt8(b);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(b);
    }

    assert_eq((int8_t*)&ret, naive);
}

TEST_F(FALLBACK, GiReinterpretAsInt32) {
    GI_INT32_t ret;
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.0f, 2.2f, 3.4f, 4.5f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretAsInt32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(int32_t));
        naive.push_back(tmp);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiReinterpretAsUint32) {
    GI_UINT32_t ret;
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.0f, 2.2f, 3.4f, 4.5f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretAsUint32(src0);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        uint32_t tmp;
        memcpy(&tmp, &s0[i], sizeof(uint32_t));
        naive.push_back(tmp);
    }

    assert_eq((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiReintInt32ToFloat32) {
    GI_FLOAT32_t ret;
    GI_INT32_t src0;
    std::vector<int32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReintInt32ToFloat32(src0);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiReintUint32ToFloat32) {
    GI_FLOAT32_t ret;
    GI_UINT32_t src0;
    std::vector<uint32_t> s0{1, 2, 3, 4};
    s0.resize(SIMD_LEN);
    init((uint32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReintUint32ToFloat32(src0);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        float tmp;
        memcpy(&tmp, &s0[i], sizeof(float));
        naive.push_back(tmp);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiRoundAsInt32) {
    GI_FLOAT32_t src0;
    GI_INT32_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, -4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiRoundAsInt32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back((int32_t)round(s0[i]));
    }

    assert_eq((int*)&ret, naive);
}

TEST_F(FALLBACK, GiCastToInt32) {
    GI_FLOAT32_t src0;
    GI_INT32_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiCastToInt32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back((int32_t)(s0[i]));
    }

    assert_eq((int*)&ret, naive);
}

TEST_F(FALLBACK, GiCastToFloat32) {
    GI_INT32_t src0;
    GI_FLOAT32_t ret;
    std::vector<int32_t> s0{100, 200, 300, 400};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiCastToFloat32(src0);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back((float)s0[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiLoadBroadcastFloat32) {
    GI_FLOAT32_t ret;
    float p = 2022.0420;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadBroadcastFloat32(&p);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(p);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiZeroFloat32) {
    GI_FLOAT32_t ret;
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    float p = 0;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiZeroFloat32();

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(p);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiLoadFloat32) {
    GI_FLOAT32_t ret;
    std::vector<float> s0{2.3f, 4.7f, -1.4f, 1223.6f};
    s0.resize(SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadFloat32(s0.data());

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiLoadFloat32V2) {
    GI_FLOAT32_V2_t ret;
    std::vector<float> s0{2.3f, 4.7f, -1.4f, 1223.6f, 1.1f, 4.0f, 99.7f, 1234.9f};
    s0.resize(SIMD_LEN * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    ret = GiLoadFloat32V2(s0.data());

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN * 2; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiLoadFloat32LowHalf) {
    GI_FLOAT32_t ret;
    std::vector<float> s0{2.3f, 4.7f, -1.4f, 1223.6f};
    s0.resize(SIMD_LEN);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadFloat32LowHalf(s0.data());

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        if (i < SIMD_LEN / 2) {
            naive.push_back(s0[i]);
        } else {
            naive.push_back(0);
        }
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMlaqFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{1.2f, -3.1f, 9.0f, 11.2f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMlaqFloat32(src0, src1, src2);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] + (s1[i] * s2[i]));
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiUzpqFloat32) {
    GI_FLOAT32_t src0, src1;
    GI_FLOAT32_V2_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    ret = GiUzpqFloat32(src0, src1);

    std::vector<float> naive;
    naive.push_back(s0[0]);
    naive.push_back(s0[2]);
    naive.push_back(s1[0]);
    naive.push_back(s1[2]);
    naive.push_back(s0[1]);
    naive.push_back(s0[3]);
    naive.push_back(s1[1]);
    naive.push_back(s1[3]);

    assert_eq((float*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiDupFloat32) {
    float32x2_t ret;
    float t = 3.1415;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiDupFloat32(t);

    auto r = (float*)&ret;
    ASSERT_EQ(*r, t);
    ASSERT_EQ(*(r + 1), t);
}

TEST_F(FALLBACK, GiLdFloat32) {
    float32x2_t ret;
    std::vector<float> s0{1.1f, -3.1415f};

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiLdFloat32(s0.data());

    auto r = (float*)&ret;
    ASSERT_EQ(*r, s0[0]);
    ASSERT_EQ(*(r + 1), s0[1]);
}

TEST_F(FALLBACK, GiAddDFloat32) {
    float32x2_t src0, src1, ret;
    std::vector<float> s0{1.1f, -3.1415f};
    std::vector<float> s1{2.3f, 3.14777f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);
    memcpy(&src1, s1.data(), sizeof(float) * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiAddDFloat32(src0, src1);

    auto r = (float*)&ret;

    auto naive0 = s0[0] + s1[0];
    auto naive1 = s0[1] + s1[1];
    ASSERT_EQ(*r, naive0);
    ASSERT_EQ(*(r + 1), naive1);
}

TEST_F(FALLBACK, GiGetLaneFloat32) {
    float32x2_t src0;
    std::vector<float> s0{1.1f, -3.1415f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);

    float ret = 0;
    ret = GiGetLaneFloat32(src0, 0);
    ASSERT_EQ(ret, s0[0]);

    ret = 0;
    ret = GiGetLaneFloat32(src0, 1);
    ASSERT_EQ(ret, s0[1]);
}

TEST_F(FALLBACK, GiSetLaneFloat32) {
    float32x2_t src0, ret;
    std::vector<float> s0{2.1f, -3.1415f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);
    float p = 2022.0420;

    auto r = (float*)&ret;
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiSetLaneFloat32(p, src0, 0);
    ASSERT_EQ(*r, p);
    ASSERT_EQ(*(r + 1), s0[1]);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiSetLaneFloat32(p, src0, 1);
    ASSERT_EQ(*r, s0[0]);
    ASSERT_EQ(*(r + 1), p);
}

TEST_F(FALLBACK, GiSt1Float32) {
    float32x2_t src0;
    std::vector<float> s0{2.1f, -3.1415f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);

    std::vector<float> ret{0, 0};
    GiSt1Float32(ret.data(), src0);
    ASSERT_EQ(ret[0], s0[0]);
    ASSERT_EQ(ret[1], s0[1]);
}

TEST_F(FALLBACK, GiLoadUzipFloat32) {
    GI_FLOAT32_V2_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f, 2312.1f, 345.244f, 3.59f, -12.8f};

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    ret = GiLoadUzipFloat32V2(s0.data());

    std::vector<float> naive0;
    std::vector<float> naive1;
    naive0.push_back(s0[0]);
    naive0.push_back(s0[2]);
    naive0.push_back(s0[4]);
    naive0.push_back(s0[6]);
    naive1.push_back(s0[1]);
    naive1.push_back(s0[3]);
    naive1.push_back(s0[5]);
    naive1.push_back(s0[7]);

    assert_eq((float*)&ret, naive0);
    assert_eq((float*)&ret + SIMD_LEN, naive1);
}

TEST_F(FALLBACK, GiExtqFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{-9.1f, 34234.6f, 9.0f, 34.1f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        size_t t_count = SIMD_LEN;
        size_t a_count = t_count - n;
        for (size_t i = 0; i < a_count; i++) {
            naive[i] = s0[i + n];
        }
        for (size_t i = 0; i < n; i++) {
            naive[i + a_count] = s1[i];
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                        \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE); \
    ret = GiExtqFloat32(src0, src1, n);              \
    compare(n);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiMultiplySubFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{-9.1f, 34234.6f, 9.0f, 34.1f};
    std::vector<float> s2{0.4f, 9.9f, 4.3f, 6.2f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplySubFloat32(src0, src1, src2);
    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] - (s1[i] * s2[i]));
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiLd1qLaneFloat32) {
    GI_FLOAT32_t src0, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);
    std::vector<float> naive = {0, 0, 0, 0};

    float buffer = 3.14159;

    auto compare = [&](const size_t n) {
        memcpy(naive.data(), s0.data(), GI_SIMD_LEN_BYTE);
        naive[n] = buffer;
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                        \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE); \
    ret = GiLd1qLaneFloat32(&buffer, src0, n);       \
    compare(n);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiSetqLaneFloat32) {
    GI_FLOAT32_t src0, ret;
    std::vector<float> s0{2.1f, 6.2f, -9.5f, 2.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);
    std::vector<float> naive = {0, 0, 0, 0};

    float buffer = 6.14159;

    auto compare = [&](const size_t n) {
        memcpy(naive.data(), s0.data(), GI_SIMD_LEN_BYTE);
        naive[n] = buffer;
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                        \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE); \
    ret = GiSetqLaneFloat32(buffer, src0, n);        \
    compare(n);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiMlaqLaneFloat32HighHalf) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{-9.1f, 34234.6f, 9.0f, 34.1f};
    std::vector<float> s2{0.4f, 9.9f, 4.3f, 6.2f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);
    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] + (s1[i] * s2[n + 2]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                             \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);      \
    ret = GiMlaqLaneFloat32HighHalf(src0, src1, src2, n); \
    compare(n);

    CB(0)
    CB(1)
#undef CB
}

TEST_F(FALLBACK, GiVmlaqLaneFloat32LowHalf) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{-9.1f, 34234.6f, 9.0f, 34.1f};
    std::vector<float> s2{0.4f, 9.9f, 4.3f, 6.2f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);
    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] + (s1[i] * s2[n]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                             \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);      \
    ret = GiVmlaqLaneFloat32LowHalf(src0, src1, src2, n); \
    compare(n);

    CB(0)
    CB(1)
#undef CB
}

TEST_F(FALLBACK, GiStoreFloat32) {
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);
    std::vector<float> ret{0};
    ret.resize(SIMD_LEN);

    GiStoreFloat32(ret.data(), src0);
    assert_eq(ret.data(), s0);
}

TEST_F(FALLBACK, GiStoreFloat32V2) {
    GI_FLOAT32_V2_t src0;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f, -1.1f, -2.2f, -3.5f, -4.9};
    s0.resize(SIMD_LEN * 2);
    init((float*)&src0, s0, SIMD_LEN * 2);
    std::vector<float> ret{0};
    ret.resize(SIMD_LEN * 2);

    GiStoreFloat32V2(ret.data(), src0);
    assert_eq(ret.data(), s0, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiStoreLaneXXFloat32) {
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);
    float ret{0};

#define CB(n)                            \
    GiStoreLane##n##Float32(&ret, src0); \
    ASSERT_EQ(ret, s0[n]);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiExtractLaneXXFloat32) {
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);
    float ret{0};

#define CB(n)                              \
    ret = GiExtractLane##n##Float32(src0); \
    ASSERT_EQ(ret, s0[n]);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiZipqFloat32) {
    GI_FLOAT32_t src0, src1;
    GI_FLOAT32_V2_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 2);
    ret = GiZipqFloat32(src0, src1);

    std::vector<float> naive;
    naive.push_back(s0[0]);
    naive.push_back(s1[0]);
    naive.push_back(s0[1]);
    naive.push_back(s1[1]);
    naive.push_back(s0[2]);
    naive.push_back(s1[2]);
    naive.push_back(s0[3]);
    naive.push_back(s1[3]);

    assert_eq((float*)&ret, naive, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiInterleaveLowFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiInterleaveLowFloat32(src0, src1);

    std::vector<float> naive;
    naive.resize(SIMD_LEN);

    for (size_t i = 0; i < SIMD_LEN / 2; i++) {
        naive[2 * i] = s0[i];
        naive[2 * i + 1] = s1[i];
    }
    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiInterleaveHighFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiInterleaveHighFloat32(src0, src1);

    std::vector<float> naive;
    naive.resize(SIMD_LEN);

    for (size_t i = 0; i < SIMD_LEN / 2; i++) {
        naive[2 * i] = s0[i + SIMD_LEN / 2];
        naive[2 * i + 1] = s1[i + SIMD_LEN / 2];
    }
    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiAddFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAddFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] + s1[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiSubtractFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiSubtractFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] - s1[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplyFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] * s1[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplyScalerFloat32) {
    GI_FLOAT32_t src0, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    float scalar = 3.1415;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyScalerFloat32(src0, scalar);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] * scalar);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplyAddFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{12.1f, 35.244f, 23.59f, -112.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyAddFloat32(src0, src1, src2);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s1[i] * s2[i] + s0[i]);
    }

    assert_lt((float*)&ret, naive, 1e-3);
}

TEST_F(FALLBACK, GiMultiplyAddScalarFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    float scalar = 3.1415;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyAddScalarFloat32(src0, src1, scalar);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s1[i] * scalar + s0[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplySubScalarFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    float scalar = 3.1415;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplySubScalarFloat32(src0, src1, scalar);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] - s1[i] * scalar);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplyAddLanXXFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{12.1f, 35.244f, 23.59f, -112.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] + (s1[i] * s2[n]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                             \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);      \
    ret = GiMultiplyAddLan##n##Float32(src0, src1, src2); \
    compare(n);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiDivideFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiDivideFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] / s1[i]);
    }

    assert_lt((float*)&ret, naive, 1e-3);
}

TEST_F(FALLBACK, GiRecpeSFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiRecpeSFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(2.0f - s0[i] * s1[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiRecpeFloat32) {
    GI_FLOAT32_t src0, ret;
    std::vector<float> s0{100.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiRecpeFloat32(src0);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(1.0f / s0[i]);
    }

    assert_lt((float*)&ret, naive, 1e-3);
}

TEST_F(FALLBACK, GiNegFloat32) {
    GI_FLOAT32_t src0, ret;
    std::vector<float> s0{-1.1f, 2.2f, 3.5f, 4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiNegFloat32(src0);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(-s0[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiGreaterThanFloat32) {
    GI_FLOAT32_t src0, src1;
    GI_UINT32_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.59f, 4.9f};
    std::vector<float> s1{2312.1f, 0.1f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiGreaterThanFloat32(src0, src1);

    std::vector<int32_t> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] > s1[i] ? 0xFFFFFFFF : 0);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiLessThanEqFloat32) {
    GI_FLOAT32_t src0, src1;
    GI_UINT32_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.59f, 4.9f};
    std::vector<float> s1{2312.1f, 0.1f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLessThanEqFloat32(src0, src1);

    std::vector<int32_t> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] <= s1[i] ? 0xFFFFFFFF : 0);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiLessThanFloat32) {
    GI_FLOAT32_t src0, src1;
    GI_UINT32_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{1.1f, 0.1f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLessThanFloat32(src0, src1);

    std::vector<int32_t> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] < s1[i] ? 0xFFFFFFFF : 0);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiAndFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAndFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp0, tmp1, tmp;
        float tmp2;
        memcpy(&tmp0, &s0[i], sizeof(int32_t));
        memcpy(&tmp1, &s1[i], sizeof(int32_t));
        tmp = tmp0 & tmp1;
        memcpy(&tmp2, &tmp, sizeof(float));
        naive.push_back(tmp2);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiOrFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{2, 2, 3, 4};
    std::vector<float> s1{6, 6, 7, 8};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiOrFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp0, tmp1, tmp;
        float tmp2;
        memcpy(&tmp0, &s0[i], sizeof(int32_t));
        memcpy(&tmp1, &s1[i], sizeof(int32_t));
        tmp = tmp0 | tmp1;
        memcpy(&tmp2, &tmp, sizeof(float));
        naive.push_back(tmp2);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiAndNotFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAndNotFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp0, tmp1, tmp;
        float tmp2;
        memcpy(&tmp0, &s0[i], sizeof(int32_t));
        memcpy(&tmp1, &s1[i], sizeof(int32_t));
        tmp = ~tmp0 & tmp1;
        memcpy(&tmp2, &tmp, sizeof(float));
        naive.push_back(tmp2);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiXorFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiXorFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        int32_t tmp0, tmp1, tmp;
        float tmp2;
        memcpy(&tmp0, &s0[i], sizeof(int32_t));
        memcpy(&tmp1, &s1[i], sizeof(int32_t));
        tmp = tmp0 ^ tmp1;
        memcpy(&tmp2, &tmp, sizeof(float));
        naive.push_back(tmp2);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiBSLFloat32) {
    GI_FLOAT32_t src0, src1, ret, na;

#if defined(GI_RVV_INTRINSICS)
    vuint32m1_t mask = vundefined_u32m1();
#else
    GI_UINT32_t mask = {0u, 0u};
#endif
    std::vector<float> s0{1.1f, 2.2f, 4.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<std::vector<uint32_t>> s2s = {
            {1, 2, 3, 0},       {0u, 0u, 0u, 0u},    {~0u, 0u, 0u, 0u},
            {~0u, ~0u, 0u, 0u}, {~0u, ~0u, ~0u, 0u}, {~0u, ~0u, ~0u, ~0u}};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    for (auto& s2 : s2s) {
        init((uint32_t*)&mask, s2);

        force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
        ret = GiBSLFloat32(mask, src0, src1);
        na = GiBlendFloat32(src0, src1, GiReintUint32ToFloat32(mask));

        std::vector<float> naive;
        naive.resize(SIMD_LEN);
        memcpy(naive.data(), &na, GI_SIMD_LEN_BYTE);

        assert_eq_and_nan((float*)&ret, naive);
    }
}

TEST_F(FALLBACK, GiMaximumFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 4.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMaximumFloat32(src0, src1);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(Max(s0[i], s1[i]));
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMinimumFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 4.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMinimumFloat32(src0, src1);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(Min(s0[i], s1[i]));
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMaxNanFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 4.5f, NAN};
    std::vector<float> s1{2312.1f, 345.244f, NAN, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMaxNanFloat32(src0, src1);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        auto t = MAX_NAN(s0[i], s1[i]);
        naive.push_back(t);
    }

    assert_eq_and_nan((float*)&ret, naive);
}

TEST_F(FALLBACK, GiMinNanFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{NAN, 2.2f, NAN, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMinNanFloat32(src0, src1);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        auto t = MIN_NAN(s0[i], s1[i]);
        naive.push_back(t);
    }

    assert_eq_and_nan((float*)&ret, naive);
}

TEST_F(FALLBACK, GiClampFloat32) {
    GI_FLOAT32_t src0, src1, ret, na;
    std::vector<float> s0{1.1f, 2.2f, 4.5f, 4.9f};
    std::vector<float> s1{1.1f, 2.2f, 4.5f, 4.9f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    float LowerRange = 3.1415;
    float UpperRange = 4.876;

    auto naive_c = [](GI_FLOAT32_t Value, float LowerRange,
                      float UpperRange) -> GI_FLOAT32_t {
        Value = GiMaximumFloat32(GiBroadcastFloat32(LowerRange), Value);
        Value = GiMinimumFloat32(GiBroadcastFloat32(UpperRange), Value);
        return Value;
    };
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiClampFloat32(src0, LowerRange, UpperRange);
    na = naive_c(src1, LowerRange, UpperRange);

    std::vector<float> naive;
    naive.resize(SIMD_LEN);
    memcpy(naive.data(), &na, GI_SIMD_LEN_BYTE);

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiReduceAddFloat32) {
    GI_FLOAT32_t src0;
    float ret{0};
    std::vector<float> s0{1.1f, 2.2f, 4.5f, -4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    ret = GiReduceAddFloat32(src0);

    float naive{0};
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive += s0[i];
    }

    ASSERT_LT(std::abs(ret - naive), 1e-3);
}

TEST_F(FALLBACK, GiReduceMultiplyFloat32) {
    GI_FLOAT32_t src0;
    float ret{0};
    std::vector<float> s0{1.1f, 2.2f, 4.5f, -4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    ret = GiReduceMultiplyFloat32(src0);

    float naive{1};
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive *= s0[i];
    }

    ASSERT_LT(std::abs(ret - naive), 1e-3);
}

TEST_F(FALLBACK, GiReduceMaxNanFloat32) {
    GI_FLOAT32_t src0;
    float ret{0};
    std::vector<float> s0{1.1f, 2.2f, 4.9f, -4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    ret = GiReduceMaxNanFloat32(src0);

    float naive = s0[0];
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive = MAX_NAN(naive, s0[i]);
    }

    ASSERT_EQ(ret, naive);
    ret = 0;
    s0 = {1.1f, 2.2f, 4.9f, NAN};
    init((float*)&src0, s0);

    ret = GiReduceMaxNanFloat32(src0);
    ASSERT_TRUE(isnan(ret));
}

TEST_F(FALLBACK, GiReduceMinNanFloat32) {
    GI_FLOAT32_t src0;
    float ret{0};
    std::vector<float> s0{1.1f, 2.2f, 4.5f, -4.9f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    ret = GiReduceMinNanFloat32(src0);

    float naive = s0[0];
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive = MIN_NAN(naive, s0[i]);
    }

    ASSERT_EQ(ret, naive);
    ret = 0;
    s0 = {-1.1f, 2.2f, 4.9f, NAN};
    init((float*)&src0, s0);

    ret = GiReduceMaxNanFloat32(src0);
    ASSERT_TRUE(isnan(ret));
}

TEST_F(FALLBACK, GiAbsFloat32) {
    GI_FLOAT32_t src0, ret;
    std::vector<float> s0{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAbsFloat32(src0);

    std::vector<float> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] > 0 ? s0[i] : -s0[i]);
    }

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiZip1qS64) {
    GI_INT64_t src0, src1, ret;
    std::vector<int64_t> s0{234242423424245, 42342342422323};
    std::vector<int64_t> s1{23424245, -4234234242232};
    s0.resize(SIMD_LEN / 2);
    s1.resize(SIMD_LEN / 2);
    memcpy(&src0, s0.data(), GI_SIMD_LEN_BYTE);
    memcpy(&src1, s1.data(), GI_SIMD_LEN_BYTE);

    ret = GiZip1qS64(src0, src1);

    std::vector<int64_t> naive;
    naive.push_back(s0[0]);
    naive.push_back(s1[0]);
    auto p = (int64_t*)&ret;
    ASSERT_EQ(naive[0], p[0]);
    ASSERT_EQ(naive[1], p[1]);
}

TEST_F(FALLBACK, GiZip2qS64) {
    GI_INT64_t src0, src1, ret;
    std::vector<int64_t> s0{234242423424245, 42342342422323};
    std::vector<int64_t> s1{23424245, -4234234242232};
    s0.resize(SIMD_LEN / 2);
    s1.resize(SIMD_LEN / 2);
    memcpy(&src0, s0.data(), GI_SIMD_LEN_BYTE);
    memcpy(&src1, s1.data(), GI_SIMD_LEN_BYTE);

    ret = GiZip2qS64(src0, src1);

    std::vector<int64_t> naive;
    naive.push_back(s0[1]);
    naive.push_back(s1[1]);
    auto p = (int64_t*)&ret;
    ASSERT_EQ(naive[0], p[0]);
    ASSERT_EQ(naive[1], p[1]);
}

TEST_F(FALLBACK, GiReinterpretqS64ToFloat32) {
    GI_INT64_t src0;
    GI_FLOAT32_t ret;
    std::vector<int64_t> s0{234242423424245, 42342342422323};
    s0.resize(SIMD_LEN / 2);
    memcpy(&src0, s0.data(), GI_SIMD_LEN_BYTE);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretqS64ToFloat32(src0);

    std::vector<float> naive;
    naive.resize(SIMD_LEN);
    memcpy(naive.data(), s0.data(), GI_SIMD_LEN_BYTE);

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiReinterpretqFloat32ToS64) {
    GI_FLOAT32_t src0;
    GI_INT64_t ret;
    std::vector<float> s0{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretqFloat32ToS64(src0);

    std::vector<float> naive;
    naive.resize(SIMD_LEN);
    memcpy(naive.data(), s0.data(), GI_SIMD_LEN_BYTE);

    assert_eq((float*)&ret, naive);
}

TEST_F(FALLBACK, GiSimdFmaLane) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{12.1f, 2.2f, 89.0f, -112.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] + (s1[i] * s2[n]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                        \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE); \
    ret = GiSimdFmaLane(src0, src1, src2, n);        \
    compare(n);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiMlaqLowLaneFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{12.1f, 2.2f, 89.0f, -112.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] + (s1[i] * s2[n]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                        \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE); \
    ret = GiMlaqLowLaneFloat32(src0, src1, src2, n); \
    compare(n);

    CB(0)
    CB(1)
#undef CB
}

TEST_F(FALLBACK, GiMlaqHighLaneFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{12.1f, 2.2f, 89.0f, -112.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] + (s1[i] * s2[n]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                         \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);  \
    ret = GiMlaqHighLaneFloat32(src0, src1, src2, n); \
    compare(n);

    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiFmsqLaneQFloat32) {
    GI_FLOAT32_t src0, src1, src2, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    std::vector<float> s2{12.1f, 2.2f, 89.0f, -112.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);
    init((float*)&src2, s2);

    std::vector<float> naive = {0, 0, 0, 0};

    auto compare = [&](const size_t n) {
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
            naive[i] = s0[i] - (s1[i] * s2[n]);
        }
        assert_eq((float*)&ret, naive);
    };

#define CB(n)                                        \
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE); \
    ret = GiFmsqLaneQFloat32(src0, src1, src2, n);   \
    compare(n);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
#undef CB
}

TEST_F(FALLBACK, GiBroadcastUint32) {
    int32_t src0 = 20220422;
    GI_UINT32_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBroadcastUint32(src0);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(src0);
    }

    assert_eq((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiLoadInt32) {
    std::vector<int32_t> s0{1, 2, -200, 999};
    GI_INT32_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadInt32(s0.data());

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiLoadInt16) {
    std::vector<int16_t> s0{1, 2, -200, 32767, -32768, 45, 3, 0};
    GI_INT16_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadInt16(s0.data());

    auto p = (int16_t*)&ret;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        ASSERT_EQ(p[i], s0[i]);
    }
}

TEST_F(FALLBACK, GiLoadInt8) {
    std::vector<int8_t> s0{9,  2, -128, 127, 2, 45, 3, 0,
                           11, 2, -128, 127, 2, 55, 3, -1};
    GI_INT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadInt8(s0.data());

    auto p = (int8_t*)&ret;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        ASSERT_EQ(p[i], s0[i]);
    }
}

TEST_F(FALLBACK, GiLoadUzipInt8V2) {
    std::vector<int8_t> s0{9,   2, -128, 127, 2,    45, 3,  0,   11, 2,  -128,
                           127, 2, 55,   3,   -1,   7,  8,  -18, 17, 12, 35,
                           7,   8, 10,   22,  -108, 27, 21, 45,  13, -11};
    GI_INT8_V2_t ret;

    force_memset_ret((void*)&ret, 2 * GI_SIMD_LEN_BYTE);
    ret = GiLoadUzipInt8V2(s0.data());

    auto p = (int8_t*)&ret;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        ASSERT_EQ(p[i], s0[2 * i]);
        ASSERT_EQ(p[SIMD_LEN_8 + i], s0[2 * i + 1]);
    }
}

TEST_F(FALLBACK, GiLoadUzipInt8V3) {
    std::vector<int8_t> s0{9,   2,  -128, 127, 2,  45, 3,    0,   11, 2,  -128, 127,
                           2,   55, 3,    -1,  7,  8,  -18,  17,  12, 35, 7,    8,
                           10,  22, -108, 27,  21, 45, 13,   -11, 11, 14, -11,  12,
                           111, 32, 6,    9,   16, 29, -118, 67,  28, 15, 19,   -10};
    GI_INT8_V3_t ret;

    force_memset_ret((void*)&ret, 3 * GI_SIMD_LEN_BYTE);
    ret = GiLoadUzipInt8V3(s0.data());

    auto p = (int8_t*)&ret;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        ASSERT_EQ(p[i], s0[3 * i]);
        ASSERT_EQ(p[SIMD_LEN_8 + i], s0[3 * i + 1]);
        ASSERT_EQ(p[2 * SIMD_LEN_8 + i], s0[3 * i + 2]);
    }
}

TEST_F(FALLBACK, GiLoadUzipInt8V4) {
    std::vector<int8_t> s0{
            9,  2,  -128, 127, 2,   45, 3,  0,  11, 2,  -128, 127, 2,  55, 3,  -1,
            7,  8,  -18,  17,  12,  35, 7,  8,  10, 22, -108, 27,  21, 45, 13, -11,
            11, 14, -11,  12,  111, 32, 6,  9,  16, 29, -118, 67,  28, 15, 19, -10,
            9,  4,  -108, 27,  22,  43, 13, 10, 31, 12, -108, 117, 22, 25, 31, -10,
    };
    GI_INT8_V4_t ret;

    force_memset_ret((void*)&ret, 4 * GI_SIMD_LEN_BYTE);
    ret = GiLoadUzipInt8V4(s0.data());

    auto p = (int8_t*)&ret;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        ASSERT_EQ(p[i], s0[4 * i]);
        ASSERT_EQ(p[SIMD_LEN_8 + i], s0[4 * i + 1]);
        ASSERT_EQ(p[2 * SIMD_LEN_8 + i], s0[4 * i + 2]);
        ASSERT_EQ(p[3 * SIMD_LEN_8 + i], s0[4 * i + 3]);
    }
}

TEST_F(FALLBACK, GiStoreInt32) {
    GI_INT32_t src0;
    std::vector<int32_t> s0{1, 2, -200, 999};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    std::vector<int32_t> ret{0};
    ret.resize(SIMD_LEN);
    GiStoreInt32(ret.data(), src0);

    assert_eq<int32_t>(ret.data(), s0);
}

TEST_F(FALLBACK, GiStoreLaneXXInt32) {
    GI_INT32_t src0;
    std::vector<int32_t> s0{1, 2, -200, 999};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    int32_t ret = 8888;

#define CB(n)                          \
    GiStoreLane##n##Int32(&ret, src0); \
    ASSERT_EQ(s0[n], ret);

    CB(0)
    CB(1)
    CB(2)
    CB(3)
}

TEST_F(FALLBACK, GiReinterInt32ToInt8) {
    GI_INT32_t src0;
    GI_INT8_t ret, naive;
    std::vector<int32_t> s0{65536, 2, -200, 999};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterInt32ToInt8(src0);
    memcpy(&naive, &src0, GI_SIMD_LEN_BYTE);

    ASSERT_FALSE(memcmp(&ret, &naive, GI_SIMD_LEN_BYTE));
}

TEST_F(FALLBACK, GiStoreInt16) {
    GI_INT16_t src0;
    std::vector<int16_t> s0{32767, 2, -200, -32768, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_16);
    init((int16_t*)&src0, s0, SIMD_LEN_16);

    std::vector<int16_t> ret{0};
    ret.resize(SIMD_LEN_16);
    GiStoreInt16(ret.data(), src0);

    assert_eq<int16_t>(ret.data(), s0, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiStoreInt8) {
    GI_INT8_t src0;
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    std::vector<int8_t> ret{0};
    ret.resize(SIMD_LEN_8);
    GiStoreInt8(ret.data(), src0);

    assert_eq<int8_t>(ret.data(), s0, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiStoreLowInt8) {
    GI_INT8_t src0;
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    std::vector<int8_t> ret{0};
    ret.resize(SIMD_LEN_8 / 2);
    GiStoreLowInt8(ret.data(), src0);

    assert_eq<int8_t>(ret.data(), s0, SIMD_LEN_8 / 2);
}

TEST_F(FALLBACK, GiStoreHighInt8) {
    GI_INT8_t src0;
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    std::vector<int8_t> ret{0};
    ret.resize(SIMD_LEN_8 / 2);
    GiStoreHighInt8(ret.data(), src0);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8 / 2; i++) {
        naive.push_back(s0[SIMD_LEN_8 / 2 + i]);
    }

    assert_eq<int8_t>(ret.data(), naive, SIMD_LEN_8 / 2);
}

TEST_F(FALLBACK, GiNegInt32) {
    GI_INT32_t src0, ret;
    std::vector<int32_t> s0{
            std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(),
            -3, 4};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiNegInt32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(-s0[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiNegInt8) {
    GI_INT8_t src0, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiNegInt8(src0);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(-s0[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiTestAndSetUint32) {
    GI_UINT32_t src0, src1, ret;
    std::vector<uint32_t> s0{
            8, 2, std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::min()};
    std::vector<uint32_t> s1{
            8, 4, std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max()};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((uint32_t*)&src0, s0);
    init((uint32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiTestAndSetUint32(src0, src1);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] & s1[i] ? 0xFFFFFFFF : 0);
    }

    assert_eq<uint32_t>((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiAddInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{127, 2, std::numeric_limits<int32_t>::max(), 9999};
    std::vector<int32_t> s1{1, 2, std::numeric_limits<int32_t>::max(), -9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAddInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] + s1[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiAddUint32) {
    GI_UINT32_t src0, src1, ret;
    std::vector<uint32_t> s0{127, 2, std::numeric_limits<uint32_t>::max(), 9999};
    std::vector<uint32_t> s1{1, 2, std::numeric_limits<uint32_t>::max(), 9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((uint32_t*)&src0, s0);
    init((uint32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAddUint32(src0, src1);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] + s1[i]);
    }

    assert_eq((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiAddInt16) {
    GI_INT16_t src0, src1, ret;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    std::vector<int16_t> s1{1,
                            2,
                            std::numeric_limits<int16_t>::max(),
                            std::numeric_limits<int16_t>::min(),
                            -1,
                            23,
                            -3,
                            -5};
    s0.resize(SIMD_LEN_16);
    s1.resize(SIMD_LEN_16);
    init((int16_t*)&src0, s0, SIMD_LEN_16);
    init((int16_t*)&src1, s1, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAddInt16(src0, src1);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive.push_back(s0[i] + s1[i]);
    }

    assert_eq<int16_t>((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiAddInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAddInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] + s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiSubtractInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{127, 2, std::numeric_limits<int32_t>::max(), 9999};
    std::vector<int32_t> s1{1, 2, std::numeric_limits<int32_t>::max(), -9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiSubtractInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] - s1[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiSubtractUint32) {
    GI_UINT32_t src0, src1, ret;
    std::vector<uint32_t> s0{127, 2, std::numeric_limits<uint32_t>::max(), 9999};
    std::vector<uint32_t> s1{1, 2, std::numeric_limits<uint32_t>::max(), 9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((uint32_t*)&src0, s0);
    init((uint32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiSubtractUint32(src0, src1);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] - s1[i]);
    }

    assert_eq((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiSubtractInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiSubtractInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] - s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiMultiplyInt32) {
    GI_INT32_t src0, src1, ret;
    std::vector<int32_t> s0{127, 2, 202204, 99};
    std::vector<int32_t> s1{1, 2, -4, -9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyInt32(src0, src1);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] * s1[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplyInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] * s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiMultiplyAddInt32) {
    GI_INT32_t src0, src1, src2, ret;
    std::vector<int32_t> s0{127, 2, 67, 9999};
    std::vector<int32_t> s1{1, 2, 90, -9};
    std::vector<int32_t> s2{-1, 12, 4, -9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);
    init((int32_t*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyAddInt32(src0, src1, src2);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] + s1[i] * s2[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiMultiplyAddInt8) {
    GI_INT8_t src0, src1, src2, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    std::vector<int8_t> s2{
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            5,
            8,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    s2.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);
    init((int8_t*)&src2, s2, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMultiplyAddInt8(src0, src1, src2);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] + s1[i] * s2[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiAndInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAndInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] & s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiEOrUint32) {
    GI_UINT32_t src0, src1, ret;
    std::vector<uint32_t> s0{127, 2, std::numeric_limits<uint32_t>::max(), 9999};
    std::vector<uint32_t> s1{1, 2, std::numeric_limits<uint32_t>::max(), 9};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((uint32_t*)&src0, s0);
    init((uint32_t*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiEOrUint32(src0, src1);

    std::vector<uint32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] ^ s1[i]);
    }

    assert_eq((uint32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiOrInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiOrInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] | s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiAndNotInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAndNotInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back((~s0[i]) & s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiXorInt8) {
    GI_INT8_t src0, src1, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiXorInt8(src0, src1);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back((s0[i]) ^ s1[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiShiftRight23Int32) {
    GI_INT32_t src0, ret;
    std::vector<int32_t> s0{1, 2, 3, -4};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiShiftRight23Int32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] >> 23);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiShiftLeft23Int32) {
    GI_INT32_t src0, ret;
    std::vector<int32_t> s0{1, 2, 3, -4};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiShiftLeft23Int32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] << 23);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiBlendInt32) {
    GI_INT32_t src0, src1, src2, ret, na;
    std::vector<int32_t> s0{1, 2, 3, -4};
    std::vector<int32_t> s1{12, 22, 32, -43};
    std::vector<int32_t> s2{-1, 21, 34, 4};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    s2.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);
    init((int32_t*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBlendInt32(src0, src1, src2);

    na = GiOrInt32(GiAndInt32(src1, src2), GiAndNotInt32(src2, src0));

    std::vector<int32_t> naive;
    auto p = (int32_t*)&na;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(p[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiBlendInt8) {
    GI_INT8_t src0, src1, src2, ret, na;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    std::vector<int8_t> s2{
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            5,
            8,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    s2.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);
    init((int8_t*)&src2, s2, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBlendInt8(src0, src1, src2);
    na = GiOrInt8(GiAndInt8(src1, src2), GiAndNotInt8(src2, src0));

    std::vector<int8_t> naive;
    auto p = (int8_t*)&na;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(p[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiAbsInt32) {
    GI_INT32_t src0, ret;
    std::vector<int32_t> s0{-1, 2, -3, 4};
    s0.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAbsInt32(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] > 0 ? s0[i] : -s0[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiAbsInt16) {
    GI_INT16_t src0, ret;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    s0.resize(SIMD_LEN_16);
    init((int16_t*)&src0, s0, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAbsInt16(src0);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive.push_back(s0[i] > 0 ? s0[i] : -s0[i]);
    }

    assert_eq<int16_t>((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiAbsInt8) {
    GI_INT8_t src0, ret;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiAbsInt8(src0);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i] > 0 ? s0[i] : -s0[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiMaximumInt32) {
    GI_INT32_t src0, src1, src2, ret, na;
    std::vector<int32_t> s0{1, -2, 3, 4};
    s0.resize(SIMD_LEN);
    std::vector<int32_t> s1{5, 6, 7, -8};
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    std::vector<int32_t> s2;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        s2.push_back(s0[i] > s1[i] ? 0xFFFFFFFF : 0);
    }
    s2.resize(SIMD_LEN);
    init((int32_t*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMaximumInt32(src0, src1);

    na = GiBlendInt32(src1, src0, src2);
    std::vector<int32_t> naive;
    auto p = (int32_t*)&na;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(p[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiMinimumInt32) {
    GI_INT32_t src0, src1, src2, ret, na;
    std::vector<int32_t> s0{1, -2, 3, 4};
    s0.resize(SIMD_LEN);
    std::vector<int32_t> s1{5, 6, 7, -8};
    s1.resize(SIMD_LEN);
    init((int32_t*)&src0, s0);
    init((int32_t*)&src1, s1);

    std::vector<int32_t> s2;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        s2.push_back(s1[i] > s0[i] ? 0xFFFFFFFF : 0);
    }
    s2.resize(SIMD_LEN);
    init((int32_t*)&src2, s2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMinimumInt32(src0, src1);

    na = GiBlendInt32(src1, src0, src2);
    std::vector<int32_t> naive;
    auto p = (int32_t*)&na;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(p[i]);
    }

    assert_eq((int32_t*)&ret, naive);
}

TEST_F(FALLBACK, GiBlendInt8x16) {
    GI_INT8_t src0, src1, src2, ret, na;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    std::vector<int8_t> s2{
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            5,
            8,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    s2.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);
    init((int8_t*)&src2, s2, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBlendInt8x16(src0, src1, src2);
    na = GiOrInt8(GiAndInt8(src1, src2), GiAndNotInt8(src2, src0));

    std::vector<int8_t> naive;
    auto p = (int8_t*)&na;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(p[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiMaximumInt8) {
    GI_INT8_t src0, src1, src2, ret, na;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    std::vector<int8_t> s2;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        s2.push_back(s1[i] < s0[i] ? 0xFF : 0);
    }
    s2.resize(SIMD_LEN_8);
    init((int8_t*)&src2, s2, SIMD_LEN_8);
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMaximumInt8(src0, src1);

    na = GiBlendInt8(src1, src0, src2);

    std::vector<int8_t> naive;
    auto p = (int8_t*)&na;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(p[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiMinimumInt8) {
    GI_INT8_t src0, src1, src2, ret, na;
    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            3,
            4};
    std::vector<int8_t> s1{
            3,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            1,
            2,
            4};
    s0.resize(SIMD_LEN_8);
    s1.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);
    init((int8_t*)&src1, s1, SIMD_LEN_8);

    std::vector<int8_t> s2;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        s2.push_back(s1[i] > s0[i] ? 0xFF : 0);
    }
    s2.resize(SIMD_LEN_8);
    init((int8_t*)&src2, s2, SIMD_LEN_8);
    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMinimumInt8(src0, src1);

    na = GiBlendInt8(src1, src0, src2);

    std::vector<int8_t> naive;
    auto p = (int8_t*)&na;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(p[i]);
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiMoveHighLongInt8) {
    GI_INT8_t src0;
    GI_INT16_t ret;

    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            3,
            4};

    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMoveHighLongInt8(src0);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8 / 2; i++) {
        naive.push_back(s0[i + SIMD_LEN_8 / 2]);
    }

    assert_eq<int16_t>((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiMoveLowLongInt8) {
    GI_INT8_t src0;
    GI_INT16_t ret;

    std::vector<int8_t> s0{
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            56,
            -128,
            1,
            2,
            3,
            4,
            127,
            2,
            56,
            -128,
            std::numeric_limits<int8_t>::max(),
            std::numeric_limits<int8_t>::min(),
            3,
            4};

    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMoveLowLongInt8(src0);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8 / 2; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq<int16_t>((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiMoveHighLongInt16) {
    GI_INT16_t src0;
    GI_INT32_t ret;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    s0.resize(SIMD_LEN_16);
    init((int16_t*)&src0, s0, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMoveHighLongInt16(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16 / 2; i++) {
        naive.push_back(s0[i + SIMD_LEN_16 / 2]);
    }

    assert_eq<int32_t>((int32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiMoveLowLongInt16) {
    GI_INT16_t src0;
    GI_INT32_t ret;
    std::vector<int16_t> s0{-127, 2, std::numeric_limits<int16_t>::max(), 9999, 1, 2,
                            3,    4};
    s0.resize(SIMD_LEN_16);
    init((int16_t*)&src0, s0, SIMD_LEN_16);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiMoveLowLongInt16(src0);

    std::vector<int32_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16 / 2; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq<int32_t>((int32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiReduceAddInt8) {
    GI_INT8_t src0;
    int32_t ret{0};
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    ret = GiReduceAddInt8(src0);

    int16_t naive{0};
    for (auto i : s0) {
        naive += i;
    }

    ASSERT_EQ(ret, naive);
}

TEST_F(FALLBACK, GiReduceMaxInt8) {
    GI_INT8_t src0;
    int8_t ret{0};
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    ret = GiReduceMaxInt8(src0);

    int8_t naive{s0[0]};
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive = Max(naive, s0[i]);
    }

    ASSERT_EQ(ret, naive);
}

TEST_F(FALLBACK, GiReduceMinInt8) {
    GI_INT8_t src0;
    int8_t ret{0};
    std::vector<int8_t> s0{127, 2, 56, -128, 1, 2, 3, 4, 127, 2, 56, -128, 1, 2, 3, 4};
    s0.resize(SIMD_LEN_8);
    init((int8_t*)&src0, s0, SIMD_LEN_8);

    ret = GiReduceMinInt8(src0);

    int8_t naive{s0[0]};
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive = Min(naive, s0[i]);
    }

    ASSERT_EQ(ret, naive);
}

TEST_F(FALLBACK, GiCvtFromFloat32ToInt8) {
    GI_INT8_t ret;
    GI_FLOAT32_t src0;
    std::vector<float> s0{
            1.0f, -2.2f, std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min()};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiCvtFromFloat32ToInt8(src0);

    std::vector<int8_t> naive;
    naive.resize(SIMD_LEN_8);

    for (size_t i = 0; i < SIMD_LEN; i++) {
        int8_t data = Saturate(round(s0[i]), -128, 127);
        naive[i] = data;
        naive[SIMD_LEN + i] = data;
        naive[2 * SIMD_LEN + i] = data;
        naive[3 * SIMD_LEN + i] = data;
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiCvtFromFloat32V2ToInt8) {
    GI_INT8_t ret;
    GI_FLOAT32_V2_t src0;
    std::vector<float> s0{
            1.0f,
            -2.2f,
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min(),
            1.1f,
            2.2f,
            -9.0f,
            899999.0f};
    s0.resize(SIMD_LEN * 2);
    init((float*)&src0, s0, SIMD_LEN * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiCvtFromFloat32V2ToInt8(src0);

    std::vector<int8_t> naive;

    for (size_t i = 0; i < SIMD_LEN * 2; i++) {
        naive.push_back(Saturate(round(s0[i]), -128, 127));
    }

    for (size_t i = 0; i < SIMD_LEN * 2; i++) {
        naive.push_back(Saturate(round(s0[i]), -128, 127));
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiCvtFromFloat32V4ToInt8) {
    GI_INT8_t ret;
    GI_FLOAT32_V4_t src0;
    std::vector<float> s0{
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min(),
            1.0f,
            -2.2f,
            3.1f,
            4.2f,
            -5.0f,
            6.0f,
            7.0f,
            8.0f,
            -9.9f,
            10.9f,
            -11.9f,
            12.9f,
            13.9f,
            -14.9f};
    s0.resize(SIMD_LEN * 4);
    init((float*)&src0, s0, SIMD_LEN * 4);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiCvtFromFloat32V4ToInt8(src0);

    std::vector<int8_t> naive;

    for (size_t i = 0; i < SIMD_LEN * 4; i++) {
        naive.push_back(Saturate(round(s0[i]), -128, 127));
    }

    assert_eq<int8_t>((int8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiCombineFloat32) {
    float32x2_t src0, src1;
    GI_FLOAT32_t ret;
    std::vector<float> s0{1.1f, -3.1415f};
    std::vector<float> s1{2.3f, 3.14777f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);
    memcpy(&src1, s1.data(), sizeof(float) * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiCombineFloat32(src0, src1);

    std::vector<float> naive;
    naive.push_back(s0[0]);
    naive.push_back(s0[1]);
    naive.push_back(s1[0]);
    naive.push_back(s1[1]);

    assert_eq<float>((float*)&ret, naive);
}

TEST_F(FALLBACK, GiGetLowFloat32) {
    float32x2_t ret;
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.0f, 2.2f, 3.4f, 4.5f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiGetLowFloat32(src0);
    auto r = (float*)&ret;

    ASSERT_EQ(*r, s0[0]);
    ASSERT_EQ(*(r + 1), s0[1]);
}

TEST_F(FALLBACK, GiGetHighFloat32) {
    float32x2_t ret;
    GI_FLOAT32_t src0;
    std::vector<float> s0{1.0f, 2.2f, 3.4f, 4.5f};
    s0.resize(SIMD_LEN);
    init((float*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiGetHighFloat32(src0);
    auto r = (float*)&ret;

    ASSERT_EQ(*r, s0[2]);
    ASSERT_EQ(*(r + 1), s0[3]);
}

TEST_F(FALLBACK, GiPaddFloat32) {
    float32x2_t src0, src1, ret;
    std::vector<float> s0{1.1f, -3.1415f};
    std::vector<float> s1{2.3f, 3.14777f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);
    memcpy(&src1, s1.data(), sizeof(float) * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiPaddFloat32(src0, src1);

    std::vector<float> naive;
    naive.push_back(s0[0] + s0[1]);
    naive.push_back(s1[0] + s1[1]);

    auto r = (float*)&ret;
    ASSERT_LT(std::abs(naive[0] - r[0]), 1e-3);
    ASSERT_LT(std::abs(naive[1] - r[1]), 1e-3);
}

TEST_F(FALLBACK, GiPmaxFloat32) {
    float32x2_t src0, src1, ret;
    std::vector<float> s0{1.1f, -3.1415f};
    std::vector<float> s1{2.3f, 3.14777f};
    memcpy(&src0, s0.data(), sizeof(float) * 2);
    memcpy(&src1, s1.data(), sizeof(float) * 2);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE / 2);
    ret = GiPmaxFloat32(src0, src1);

    std::vector<float> naive;
    auto t0 = MAX_NAN(s0[0], s0[1]);
    auto t1 = MAX_NAN(s1[0], s1[1]);
    naive.push_back(t0);
    naive.push_back(t1);

    auto r = (float*)&ret;
    ASSERT_LT(std::abs(naive[0] - r[0]), 1e-3);
    ASSERT_LT(std::abs(naive[1] - r[1]), 1e-3);
}

TEST_F(FALLBACK, GiStoreZipFloat32V2) {
    GI_FLOAT32_V2_t src0;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f, 2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN * 2);
    init((float*)&src0, s0, SIMD_LEN * 2);
    std::vector<float> ret;
    ret.resize(SIMD_LEN * 2);
    std::vector<float> ret_cmp;
    ret_cmp.resize(SIMD_LEN * 2);

    GiStoreZipFloat32V2(ret.data(), src0);

    GI_FLOAT32_V2_t tmp;
    tmp = GiZipqFloat32(
            GiGetSubVectorFloat32V2(src0, 0), GiGetSubVectorFloat32V2(src0, 1));
    GiStoreFloat32(ret_cmp.data(), GiGetSubVectorFloat32V2(tmp, 0));
    GiStoreFloat32(ret_cmp.data() + SIMD_LEN, GiGetSubVectorFloat32V2(tmp, 1));

    assert_eq(ret.data(), ret_cmp, SIMD_LEN * 2);
}

TEST_F(FALLBACK, GiLoadUzipFloat32V3) {
    GI_FLOAT32_V3_t ret;
    std::vector<float> s0{1.1f,  2.2f,   3.5f, 4.9f, 2312.1f, 345.244f,
                          3.59f, -12.8f, 2.2f, 6.0f, 90.0f,   89.3f};
    s0.resize(SIMD_LEN * 3);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 3);
    ret = GiLoadUzipFloat32V3(s0.data());
    std::vector<float> naive;
    for (size_t i = 0; i < 3; i++) {
        naive.push_back(s0[0 + i]);
        naive.push_back(s0[3 + i]);
        naive.push_back(s0[6 + i]);
        naive.push_back(s0[9 + i]);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 3);
}

TEST_F(FALLBACK, GiLoadUzipFloat32V4) {
    GI_FLOAT32_V4_t ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f,  4.9f,  2312.1f, 345.244f, 3.59f, -12.8f,
                          2.2f, 6.0f, 90.0f, 89.3f, 2.1f,    -3.5f,    4.9f,  -2312.1f};
    s0.resize(SIMD_LEN * 4);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE * 4);
    ret = GiLoadUzipFloat32V4(s0.data());
    std::vector<float> naive;
    for (size_t i = 0; i < 4; i++) {
        naive.push_back(s0[0 + i]);
        naive.push_back(s0[4 + i]);
        naive.push_back(s0[8 + i]);
        naive.push_back(s0[12 + i]);
    }

    assert_eq((float*)&ret, naive, SIMD_LEN * 4);
}

TEST_F(FALLBACK, GiStoreZipFloat32V3) {
    GI_FLOAT32_V3_t src0;
    std::vector<float> s0{1.1f,  2.2f,   3.5f,  4.9f,   2312.1f, 345.244f,
                          3.59f, -12.8f, 3.59f, -12.8f, 2.2f,    6.0};
    s0.resize(SIMD_LEN * 3);
    //! rvv compiler crash when use init on type_x3, use rvv load api as a workaround
#if defined(GI_RVV_INTRINSICS)
    vfloat32m1_t t00, t10, t20;
    t00 = vle32_v_f32m1(s0.data(), SIMD_LEN);
    t10 = vle32_v_f32m1(s0.data() + SIMD_LEN, 4);
    t20 = vle32_v_f32m1(s0.data() + SIMD_LEN * 2, 4);
    src0 = vcreate_f32m1x3(t00, t10, t20);
#else
    init((float*)&src0, s0, SIMD_LEN * 3);
#endif

    std::vector<float> ret;
    ret.resize(SIMD_LEN * 3);

    GiStoreZipFloat32V3(ret.data(), src0);

    std::vector<float> ret_cmp;
    for (size_t i = 0; i < SIMD_LEN; i++) {
        ret_cmp.push_back(s0[0 + i]);
        ret_cmp.push_back(s0[4 + i]);
        ret_cmp.push_back(s0[8 + i]);
    }

    assert_eq(ret.data(), ret_cmp, SIMD_LEN * 3);
}

TEST_F(FALLBACK, GiDivFloat32) {
    GI_FLOAT32_t src0, src1, ret;
    std::vector<float> s0{1.1f, 2.2f, 3.5f, 4.9f};
    std::vector<float> s1{2312.1f, 345.244f, 3.59f, -12.8f};
    s0.resize(SIMD_LEN);
    s1.resize(SIMD_LEN);
    init((float*)&src0, s0);
    init((float*)&src1, s1);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiDivFloat32(src0, src1);

    std::vector<float> naive;

    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive.push_back(s0[i] / s1[i]);
    }

    assert_lt((float*)&ret, naive, 1e-3);
}

TEST_F(FALLBACK, GiLoadUint8) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255};
    GI_UINT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadUint8(s0.data());

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiReverseUint8) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    GI_UINT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadUint8(s0.data());
    ret = GiReverseUint8(ret);

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[SIMD_LEN_8 - i - 1]);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiStoreUint8) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255};
    GI_UINT8_t src;
    std::vector<uint8_t> ret;
    ret.resize(SIMD_LEN_8);
    force_memset_ret((void*)&src, GI_SIMD_LEN_BYTE);
    src = GiLoadUint8(s0.data());
    GiStoreUint8(ret.data(), src);
    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i]);
    }

    assert_eq(ret.data(), naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiLoadUzip0V3Uint8) {
    std::vector<uint8_t> s0{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 255};
    GI_UINT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadUzip0V3Uint8(s0.data());

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i * 3]);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiLoadUzip1V3Uint8) {
    std::vector<uint8_t> s0{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 255};
    GI_UINT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadUzip1V3Uint8(s0.data());

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i * 3 + 1]);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiLoadUzip2V3Uint8) {
    std::vector<uint8_t> s0{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 255};
    GI_UINT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadUzip2V3Uint8(s0.data());

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8; i++) {
        naive.push_back(s0[i * 3 + 2]);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiStoreZipUint8V3) {
    std::vector<uint8_t> s0{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 255};
    GI_UINT8_t src0, src1, src2;
    std::vector<uint8_t> ret;
    ret.resize(SIMD_LEN_8 * 3);

    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src2, GI_SIMD_LEN_BYTE);
    src0 = GiLoadUzip0V3Uint8(s0.data());
    src1 = GiLoadUzip1V3Uint8(s0.data());
    src2 = GiLoadUzip2V3Uint8(s0.data());

    GiStoreZipUint8V3(ret.data(), src0, src1, src2);

    std::vector<uint8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8 * 3; i++) {
        naive.push_back(s0[i]);
    }
    assert_eq(ret.data(), naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiShiftRightInt16ToUint8) {
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    GI_INT16_t src;
    force_memset_ret((void*)&src, GI_SIMD_LEN_BYTE);
    src = GiLoadInt16(s0.data());

    std::vector<uint8_t> naive;
    naive.resize(SIMD_LEN_8);
    GI_UINT8_t ret;
#define TEST_BLOCK(shift)                                     \
    ret = GiShiftRightInt16ToUint8(src, shift);               \
    for (size_t i = 0; i < SIMD_LEN_16; i++) {                \
        uint8_t val = Saturate(s0[i] >> shift, 0, UINT8_MAX); \
        naive[i] = val;                                       \
        naive[i + SIMD_LEN_16] = val;                         \
    }                                                         \
    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);

    TEST_BLOCK(1);
    TEST_BLOCK(2);
    TEST_BLOCK(3);
    TEST_BLOCK(4);
    TEST_BLOCK(5);
    TEST_BLOCK(6);
    TEST_BLOCK(7);
    TEST_BLOCK(8);
#undef TEST_BLOCK
}

TEST_F(FALLBACK, GiCombineInt16Low) {
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    std::vector<int16_t> s1{1, 2, 3, -4, 5, -6, 7, -8};
    GI_INT16_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s1.data());

    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    GI_INT16_t ret = GiCombineInt16Low(src0, src1);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[i] = s0[i];
        naive[i + SIMD_LEN] = s1[i];
    }
    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiCombineUint8Low) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    GI_UINT8_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadUint8(s0.data());
    src1 = GiLoadUint8(s0.data());

    std::vector<uint8_t> naive;
    naive.resize(SIMD_LEN_8);
    GI_UINT8_t ret = GiCombineUint8Low(src0, src1);
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive[i] = s0[i];
        naive[i + SIMD_LEN_16] = s0[i];
    }
    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiZipV0Int8) {
    std::vector<int8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    GI_INT8_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt8(s0.data());
    src1 = GiLoadInt8(s0.data());

    std::vector<int8_t> naive;
    naive.resize(SIMD_LEN_8);
    GI_INT8_t ret = GiZipV0Int8(src0, src1);
    for (size_t i = 0; i < SIMD_LEN_16; ++i) {
        naive[2 * i] = s0[i];
        naive[2 * i + 1] = s0[i];
    }
    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiZipV1Int8) {
    std::vector<int8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    GI_INT8_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt8(s0.data());
    src1 = GiLoadInt8(s0.data());

    std::vector<int8_t> naive;
    naive.resize(SIMD_LEN_8);
    GI_INT8_t ret = GiZipV1Int8(src0, src1);
    for (size_t i = 0; i < SIMD_LEN_16; ++i) {
        naive[2 * i] = s0[i + SIMD_LEN_16];
        naive[2 * i + 1] = s0[i + SIMD_LEN_16];
    }
    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiReinterpretInt8AsInt16) {
    GI_INT8_t src0;
    GI_INT16_t ret, naive;
    std::vector<int8_t> s0{1, 2, -2, -1, INT8_MAX, INT8_MIN, 5,  6,
                           7, 8, 9,  10, 11,       12,       13, 14};
    s0.resize(SIMD_LEN);
    init((int8_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretInt8AsInt16(src0);
    memcpy(&naive, &src0, GI_SIMD_LEN_BYTE);

    ASSERT_FALSE(memcmp(&ret, &naive, GI_SIMD_LEN_BYTE));
}
TEST_F(FALLBACK, GiZipV0Int16) {
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    GI_INT16_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s0.data());

    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    GI_INT16_t ret = GiZipV0Int16(src0, src1);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[2 * i] = s0[i];
        naive[2 * i + 1] = s0[i];
    }
    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiZipV1Int16) {
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    GI_INT16_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s0.data());

    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    GI_INT16_t ret = GiZipV1Int16(src0, src1);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[2 * i] = s0[i + SIMD_LEN];
        naive[2 * i + 1] = s0[i + SIMD_LEN];
    }
    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiReinterpretInt16AsInt32) {
    GI_INT16_t src0;
    GI_INT32_t ret, naive;
    std::vector<int16_t> s0{1, 2, -2, -1, INT16_MAX, INT16_MIN, 5, 6};
    s0.resize(SIMD_LEN);
    init((int16_t*)&src0, s0);

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiReinterpretInt16AsInt32(src0);
    memcpy(&naive, &src0, GI_SIMD_LEN_BYTE);

    ASSERT_FALSE(memcmp(&ret, &naive, GI_SIMD_LEN_BYTE));
}
TEST_F(FALLBACK, GiZipV0Int32) {
    std::vector<int32_t> s0{INT32_MAX, INT32_MIN, 0x00005678, -0x00005678};
    GI_INT32_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt32(s0.data());
    src1 = GiLoadInt32(s0.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret = GiZipV0Int32(src0, src1);
    for (size_t i = 0; i < SIMD_LEN / 2; i++) {
        naive[2 * i] = s0[i];
        naive[2 * i + 1] = s0[i];
    }
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}
TEST_F(FALLBACK, GiZipV1Int32) {
    std::vector<int32_t> s0{INT16_MAX, INT16_MIN, 0x00005678, -0x00005678};
    GI_INT32_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt32(s0.data());
    src1 = GiLoadInt32(s0.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret = GiZipV1Int32(src0, src1);
    for (size_t i = 0; i < SIMD_LEN / 2; i++) {
        naive[2 * i] = s0[i + SIMD_LEN / 2];
        naive[2 * i + 1] = s0[i + SIMD_LEN / 2];
    }
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}
TEST_F(FALLBACK, GiCombineInt32Low) {
    std::vector<int32_t> s0{INT16_MAX, INT16_MIN, 0x00005678, -0x00005678};
    GI_INT32_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt32(s0.data());
    src1 = GiLoadInt32(s0.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret = GiCombineInt32Low(src0, src1);
    for (size_t i = 0; i < SIMD_LEN / 2; i++) {
        naive[i] = s0[i];
        naive[i + SIMD_LEN / 2] = s0[i];
    }
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}
TEST_F(FALLBACK, GiCombineInt32High) {
    std::vector<int32_t> s0{INT16_MAX, INT16_MIN, 0x00005678, -0x00005678};
    GI_INT32_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt32(s0.data());
    src1 = GiLoadInt32(s0.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret = GiCombineInt32High(src0, src1);
    for (size_t i = 0; i < SIMD_LEN / 2; i++) {
        naive[i] = s0[i + SIMD_LEN / 2];
        naive[i + SIMD_LEN / 2] = s0[i + SIMD_LEN / 2];
    }
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}

TEST_F(FALLBACK, GiStoreZipInt8V3) {
    std::vector<int8_t> s0{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 127};
    GI_INT8_t src0, src1, src2;
    GI_INT8_V3_t src;
    std::vector<int8_t> ret;
    ret.resize(SIMD_LEN_8 * 3);
    force_memset_ret((void*)&src, GI_SIMD_LEN_BYTE * 3);
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src2, GI_SIMD_LEN_BYTE);
    src = GiLoadUzipInt8V3(s0.data());
    src0 = GiGetSubVectorInt8V3(src, 0);
    src1 = GiGetSubVectorInt8V3(src, 1);
    src2 = GiGetSubVectorInt8V3(src, 2);

    GiStoreZipInt8V3(ret.data(), src0, src1, src2);

    std::vector<int8_t> naive;
    for (size_t i = 0; i < SIMD_LEN_8 * 3; i++) {
        naive.push_back(s0[i]);
    }
    assert_eq(ret.data(), naive, SIMD_LEN_8);
}

TEST_F(FALLBACK, GiShiftRightInt32) {
    std::vector<int32_t> s0{INT32_MAX, INT32_MIN, 0x12345678, -0x12345678};
    GI_INT32_t src;
    force_memset_ret((void*)&src, GI_SIMD_LEN_BYTE);
    src = GiLoadInt32(s0.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret;
#define TEST_BLOCK(shift)                   \
    ret = GiShiftRightInt32(src, shift);    \
    for (size_t i = 0; i < SIMD_LEN; i++) { \
        naive[i] = s0[i] >> shift;          \
    }                                       \
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);

    TEST_BLOCK(1);
    TEST_BLOCK(2);
    TEST_BLOCK(3);
    TEST_BLOCK(4);
    TEST_BLOCK(5);
    TEST_BLOCK(6);
    TEST_BLOCK(7);
    TEST_BLOCK(8);
    TEST_BLOCK(9);
    TEST_BLOCK(10);
    TEST_BLOCK(11);
    TEST_BLOCK(12);
    TEST_BLOCK(13);
    TEST_BLOCK(14);
    TEST_BLOCK(15);
    TEST_BLOCK(16);

#undef TEST_BLOCK
}
TEST_F(FALLBACK, GiShiftLeftInt32) {
    std::vector<int32_t> s0{INT32_MAX, INT32_MIN, 0x12345678, -0x12345678};
    GI_INT32_t src;
    force_memset_ret((void*)&src, GI_SIMD_LEN_BYTE);
    src = GiLoadInt32(s0.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret;
#define TEST_BLOCK(shift)                   \
    ret = GiShiftLeftInt32(src, shift);     \
    for (size_t i = 0; i < SIMD_LEN; i++) { \
        naive[i] = s0[i] << shift;          \
    }                                       \
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);

    TEST_BLOCK(1);
    TEST_BLOCK(2);
    TEST_BLOCK(3);
    TEST_BLOCK(4);
    TEST_BLOCK(5);
    TEST_BLOCK(6);
    TEST_BLOCK(7);
    TEST_BLOCK(8);
    TEST_BLOCK(9);
    TEST_BLOCK(10);
    TEST_BLOCK(11);
    TEST_BLOCK(12);
    TEST_BLOCK(13);
    TEST_BLOCK(14);
    TEST_BLOCK(15);
    TEST_BLOCK(16);

#undef TEST_BLOCK
}

TEST_F(FALLBACK, GiBroadcastInt16) {
    int16_t src0 = 5;
    GI_INT16_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiBroadcastInt16(src0);

    std::vector<int16_t> naive;
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive.push_back(src0);
    }

    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiAndInt16) {
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    GI_INT16_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s0.data());

    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    GI_INT16_t ret = GiAndInt16(src0, src1);
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive[i] = s0[i] & s0[i];
    }
    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiCvtInt32ToInt16) {
    std::vector<int32_t> s0{INT32_MAX, INT32_MIN, 0x12345678, -0x12345678};
    GI_INT32_t src;
    force_memset_ret((void*)&src, GI_SIMD_LEN_BYTE);
    src = GiLoadInt32(s0.data());

    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    GI_INT16_t ret;
    ret = GiCvtInt32ToInt16(src);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        int16_t val = Saturate(s0[i], INT16_MIN, INT16_MAX);
        naive[i] = val;
        naive[i + SIMD_LEN] = val;
    }
    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiInterleave4Int8) {
    std::vector<int8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    GI_INT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadInt8(s0.data());
    ret = GiInterleave4Int8(ret);

    std::vector<int8_t> naive;
    naive.resize(SIMD_LEN_8);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[i] = s0[i * 4];
        naive[i + 4] = s0[i * 4 + 1];
        naive[i + 2 * 4] = s0[i * 4 + 2];
        naive[i + 3 * 4] = s0[i * 4 + 3];
    }

    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiCvtUint8toInt16Low) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 127};
    GI_INT16_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    GI_UINT8_t src = GiLoadUint8(s0.data());
    ret = GiCvtUint8toInt16Low(src);
    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive[i] = s0[i];
    }

    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiCvtUint8toInt16High) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 127};
    GI_INT16_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    GI_UINT8_t src = GiLoadUint8(s0.data());
    ret = GiCvtUint8toInt16High(src);
    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive[i] = s0[i + SIMD_LEN_16];
    }

    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}
TEST_F(FALLBACK, GiMultiplyAddInt16LongLow) {
    GI_INT16_t src0, src1;
    GI_INT32_t src2;
    std::vector<int32_t> s1{1, 2, 3, 4};
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src2, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s0.data());
    src2 = GiLoadInt32(s1.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret = GiMultiplyAddInt16LongLow(src2, src0, src1);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[i] = (int32_t)s1[i] + (int32_t)s0[i] * (int32_t)s0[i];
    }
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}
TEST_F(FALLBACK, GiMultiplyAddInt16LongHigh) {
    GI_INT16_t src0, src1;
    GI_INT32_t src2;
    std::vector<int32_t> s1{1, 2, 3, 4};
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src2, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s0.data());
    src2 = GiLoadInt32(s1.data());

    std::vector<int32_t> naive;
    naive.resize(SIMD_LEN);
    GI_INT32_t ret = GiMultiplyAddInt16LongHigh(src2, src0, src1);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[i] =
                (int32_t)s1[i] + (int32_t)s0[i + SIMD_LEN] * (int32_t)s0[i + SIMD_LEN];
    }
    assert_eq((int32_t*)&ret, naive, SIMD_LEN);
}
TEST_F(FALLBACK, GiCvtFromInt32V4ToUint8) {
    std::vector<int32_t> s0{INT16_MAX, INT16_MIN, 0x00005678, -0x00005678};
    GI_INT32_t src0, src1, src2, src3;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt32(s0.data());
    src1 = GiLoadInt32(s0.data());
    src2 = GiLoadInt32(s0.data());
    src3 = GiLoadInt32(s0.data());
    GI_UINT8_t ret = GiCvtFromInt32V4ToUint8(src0, src1, src2, src3);
    std::vector<uint8_t> naive;
    naive.resize(SIMD_LEN_8);
    for (size_t i = 0; i < SIMD_LEN; i++) {
        naive[i] = Saturate(s0[i], 0, UINT8_MAX);
        naive[i + SIMD_LEN] = Saturate(s0[i], 0, UINT8_MAX);
        naive[i + 2 * SIMD_LEN] = Saturate(s0[i], 0, UINT8_MAX);
        naive[i + 3 * SIMD_LEN] = Saturate(s0[i], 0, UINT8_MAX);
    }

    assert_eq((uint8_t*)&ret, naive, SIMD_LEN_8);
}
TEST_F(FALLBACK, GiSubtractInt16) {
    std::vector<int16_t> s0{INT16_MAX,  INT16_MIN,   0x00005678, -0x00005678,
                            0x00001234, -0x00001234, 0x00000fff, -0x00000fff};
    GI_INT16_t src0, src1;
    force_memset_ret((void*)&src0, GI_SIMD_LEN_BYTE);
    force_memset_ret((void*)&src1, GI_SIMD_LEN_BYTE);
    src0 = GiLoadInt16(s0.data());
    src1 = GiLoadInt16(s0.data());

    std::vector<int16_t> naive;
    naive.resize(SIMD_LEN_16);
    GI_INT16_t ret = GiSubtractInt16(src0, src1);
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive[i] = s0[i] - s0[i];
    }
    assert_eq((int16_t*)&ret, naive, SIMD_LEN_16);
}

TEST_F(FALLBACK, GiInterleave2UInt8) {
    std::vector<uint8_t> s0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    GI_UINT8_t ret;

    force_memset_ret((void*)&ret, GI_SIMD_LEN_BYTE);
    ret = GiLoadUint8(s0.data());
    ret = GiInterleave2Uint8(ret);

    std::vector<int8_t> naive;
    naive.resize(SIMD_LEN_8);
    for (size_t i = 0; i < SIMD_LEN_16; i++) {
        naive[i] = s0[2 * i];
        naive[i + SIMD_LEN_16] = s0[2 * i + 1];
    }

    assert_eq((int8_t*)&ret, naive, SIMD_LEN_8);
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
