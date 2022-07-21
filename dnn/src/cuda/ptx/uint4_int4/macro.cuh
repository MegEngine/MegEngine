#pragma once

//! ============= i2f ===============
__device__ __forceinline__ void i2f(int2& a) {
    ((float*)&a)[0] = static_cast<float>(a.x);
    ((float*)&a)[1] = static_cast<float>(a.y);
}

//! ============= mul ===============
template <typename T>
__device__ __forceinline__ void mul_v4(int4& c, const int4 a, const T alpha);

template <>
__device__ __forceinline__ void mul_v4<float>(
        int4& c, const int4 a, const float alpha) {
    ((float*)&c)[0] = ((float*)&a)[0] * alpha;
    ((float*)&c)[1] = ((float*)&a)[1] * alpha;
    ((float*)&c)[2] = ((float*)&a)[2] * alpha;
    ((float*)&c)[3] = ((float*)&a)[3] * alpha;
}

//! ============= fma ===============
__device__ __forceinline__ void fma2(
        int2& c0, const int2 a0, int2& c1, const int2 a1, const float alpha,
        const int4 b) {
    asm("fma.rz.f32 %0, %1, %2, %3;"
        : "=f"(((float*)&c0)[0])
        : "f"(((float*)&a0)[0]), "f"(alpha), "f"(((float*)&b)[0]));
    asm("fma.rz.f32 %0, %1, %2, %3;"
        : "=f"(((float*)&c0)[1])
        : "f"(((float*)&a0)[1]), "f"(alpha), "f"(((float*)&b)[1]));
    asm("fma.rz.f32 %0, %1, %2, %3;"
        : "=f"(((float*)&c1)[0])
        : "f"(((float*)&a1)[0]), "f"(alpha), "f"(((float*)&b)[2]));
    asm("fma.rz.f32 %0, %1, %2, %3;"
        : "=f"(((float*)&c1)[1])
        : "f"(((float*)&a1)[1]), "f"(alpha), "f"(((float*)&b)[3]));
}

__device__ __forceinline__ void fuse_z_1x8(
        int4* a, const int& j, const int4& fuse_z, const float& gamma,
        const int32_t& zero_point) {
    const int2 z[2] = {
            *reinterpret_cast<const int2*>(&fuse_z),
            *(reinterpret_cast<const int2*>(&fuse_z) + 1)};
    for (int k = 0; k < 4; k++) {
        int f = ((z[0].x >> (k * 8)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k]))[0] += (f - zero_point) * gamma;
        f = ((z[0].x >> (k * 8 + 4)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k]))[1] += (f - zero_point) * gamma;

        f = ((z[1].x >> (k * 8)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k]))[2] += (f - zero_point) * gamma;
        f = ((z[1].x >> (k * 8 + 4)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k]))[3] += (f - zero_point) * gamma;
    }
    for (int k = 0; k < 4; k++) {
        int f = ((z[0].y >> (k * 8)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k + 4]))[0] += (f - zero_point) * gamma;
        f = ((z[0].y >> (k * 8 + 4)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k + 4]))[1] += (f - zero_point) * gamma;

        f = ((z[1].y >> (k * 8)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k + 4]))[2] += (f - zero_point) * gamma;
        f = ((z[1].y >> (k * 8 + 4)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k + 4]))[3] += (f - zero_point) * gamma;
    }
}

__device__ __forceinline__ void fuse_z_1x8(
        int2* a, const int& j, const int2& fuse_z, const float& gamma,
        const int32_t& zero_point) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int f = ((fuse_z.x >> (k * 8)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k]))[0] += (f - zero_point) * gamma;
        f = ((fuse_z.x >> (k * 8 + 4)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k]))[1] += (f - zero_point) * gamma;
    }
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int f = ((fuse_z.y >> (k * 8)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k + 4]))[0] += (f - zero_point) * gamma;
        f = ((fuse_z.y >> (k * 8 + 4)) & 15);
        f = (f << 28) >> 28;
        ((float*)&(a[j + k + 4]))[1] += (f - zero_point) * gamma;
    }
}

__device__ __forceinline__ void pack_f2i(
        int& d0, int& d1, const int4 s0, const int4 s1, const int4 s2, const int4 s3,
        const uint32_t relu, float& dst_zero_point) {
    // uint32_t ix, iy, iz, iw;
    uint32_t x0, y0, z0, w0;
    uint32_t x1, y1, z1, w1;
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x0) : "f"(((float*)&s0)[0]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(y0) : "f"(((float*)&s0)[1]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(z0) : "f"(((float*)&s1)[0]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(w0) : "f"(((float*)&s1)[1]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x1) : "f"(((float*)&s2)[0]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(y1) : "f"(((float*)&s2)[1]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(z1) : "f"(((float*)&s3)[0]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(w1) : "f"(((float*)&s3)[1]));

    asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.u4.s32.b32   r4, %8, %7, 0;"
            "cvt.pack.sat.u4.s32.b32   r4, %6, %5, r4;"
            "cvt.pack.sat.u4.s32.b32   r4, %4, %3, r4;"
            "cvt.pack.sat.u4.s32.b32   %0, %2, %1, r4;"
            "}"
            : "=r"(d0)
            : "r"(x0), "r"(y0), "r"(z0), "r"(w0), "r"(x1), "r"(y1), "r"(z1), "r"(w1));

    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x0) : "f"(((float*)&s0)[2]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(y0) : "f"(((float*)&s0)[3]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(z0) : "f"(((float*)&s1)[2]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(w0) : "f"(((float*)&s1)[3]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x1) : "f"(((float*)&s2)[2]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(y1) : "f"(((float*)&s2)[3]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(z1) : "f"(((float*)&s3)[2]));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(w1) : "f"(((float*)&s3)[3]));

    asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.u4.s32.b32   r4, %8, %7, 0;"
            "cvt.pack.sat.u4.s32.b32   r4, %6, %5, r4;"
            "cvt.pack.sat.u4.s32.b32   r4, %4, %3, r4;"
            "cvt.pack.sat.u4.s32.b32   %0, %2, %1, r4;"
            "}"
            : "=r"(d1)
            : "r"(x0), "r"(y0), "r"(z0), "r"(w0), "r"(x1), "r"(y1), "r"(z1), "r"(w1));
}

__device__ __forceinline__ void pack_f2i_with_relu(
        int& d0, const int2 s0, const int2 s1, const int2 s2, const int2 s3,
        const uint32_t relu, float& dst_zero_point) {
    uint32_t x[8];

    if (relu > 0) {
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[0]) : "f"(((float*)&s0)[0]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[1]) : "f"(((float*)&s0)[1]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[2]) : "f"(((float*)&s1)[0]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[3]) : "f"(((float*)&s1)[1]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[4]) : "f"(((float*)&s2)[0]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[5]) : "f"(((float*)&s2)[1]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[6]) : "f"(((float*)&s3)[0]));
        asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(x[7]) : "f"(((float*)&s3)[1]));
        x[0] += dst_zero_point;
        x[1] += dst_zero_point;
        x[2] += dst_zero_point;
        x[3] += dst_zero_point;
        x[4] += dst_zero_point;
        x[5] += dst_zero_point;
        x[6] += dst_zero_point;
        x[7] += dst_zero_point;
    } else if (relu == 0) {
        ((float*)&s0)[0] += dst_zero_point;
        ((float*)&s0)[1] += dst_zero_point;
        ((float*)&s1)[0] += dst_zero_point;
        ((float*)&s1)[1] += dst_zero_point;
        ((float*)&s2)[0] += dst_zero_point;
        ((float*)&s2)[1] += dst_zero_point;
        ((float*)&s3)[0] += dst_zero_point;
        ((float*)&s3)[1] += dst_zero_point;
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[0]) : "f"(((float*)&s0)[0]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[1]) : "f"(((float*)&s0)[1]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[2]) : "f"(((float*)&s1)[0]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[3]) : "f"(((float*)&s1)[1]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[4]) : "f"(((float*)&s2)[0]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[5]) : "f"(((float*)&s2)[1]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[6]) : "f"(((float*)&s3)[0]));
        asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(x[7]) : "f"(((float*)&s3)[1]));
    }

    if (relu > 1) {
        int r1, r2;
        r1 = (x[0] >= relu);
        x[0] *= r1;
        r2 = (x[1] >= relu);
        x[1] *= r2;
        r1 = (x[2] >= relu);
        x[2] *= r1;
        r2 = (x[3] >= relu);
        x[3] *= r2;
        r1 = (x[4] >= relu);
        x[4] *= r1;
        r2 = (x[5] >= relu);
        x[5] *= r2;
        r1 = (x[6] >= relu);
        x[6] *= r1;
        r2 = (x[7] >= relu);
        x[7] *= r2;
    }

    asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.u4.s32.b32   r4, %8, %7, 0;"
            "cvt.pack.sat.u4.s32.b32   r4, %6, %5, r4;"
            "cvt.pack.sat.u4.s32.b32   r4, %4, %3, r4;"
            "cvt.pack.sat.u4.s32.b32   %0, %2, %1, r4;"
            "}"
            : "=r"(d0)
            : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]),
              "r"(x[6]), "r"(x[7]));
}

#define I2F_1x8(a, i, j) \
    i2f(a[i][j]);        \
    i2f(a[i][j + 1]);    \
    i2f(a[i][j + 2]);    \
    i2f(a[i][j + 3]);    \
    i2f(a[i][j + 4]);    \
    i2f(a[i][j + 5]);    \
    i2f(a[i][j + 6]);    \
    i2f(a[i][j + 7]);

#define I2F_4x8(a, i, j) \
    I2F_1x8(a, i, j) I2F_1x8(a, i + 1, j) I2F_1x8(a, i + 2, j) I2F_1x8(a, i + 3, j)

#define FMA_1x8(a, i, j, alpha, bias0, bias1, bias2, bias3)                     \
    fma2(a[i][j], reg_acc[i][j], a[i][j + 1], reg_acc[i][j + 1], alpha, bias0); \
    fma2(a[i][j + 2], reg_acc[i][j + 2], a[i][j + 3], reg_acc[i][j + 3], alpha, \
         bias1);                                                                \
    fma2(a[i][j + 4], reg_acc[i][j + 4], a[i][j + 5], reg_acc[i][j + 5], alpha, \
         bias2);                                                                \
    fma2(a[i][j + 6], reg_acc[i][j + 6], a[i][j + 7], reg_acc[i][j + 7], alpha, bias3);

#define FMA_4x8(a, i, j, alpha, bias0, bias1, bias2, bias3)                 \
    FMA_1x8(a, i, j, alpha, bias0, bias1, bias2, bias3)                     \
            FMA_1x8(a, i + 1, j, alpha, bias0, bias1, bias2, bias3)         \
                    FMA_1x8(a, i + 2, j, alpha, bias0, bias1, bias2, bias3) \
                            FMA_1x8(a, i + 3, j, alpha, bias0, bias1, bias2, bias3)

// pack 1x(8 int2) to int2
#define PACK_F2I_WITH_RELU_1x8(a, i, j, relu, dst_zero_point)                    \
    pack_f2i_with_relu(                                                          \
            a[i][j].x, a[i][j], a[i][j + 1], a[i][j + 2], a[i][j + 3], relu,     \
            dst_zero_point);                                                     \
    pack_f2i_with_relu(                                                          \
            a[i][j].y, a[i][j + 4], a[i][j + 5], a[i][j + 6], a[i][j + 7], relu, \
            dst_zero_point);

// pack 4x8 int2 float to 4 int2
#define PACK_F2I_WITH_RELU_4x8(a, i, j, relu, dst_zero_point)                 \
    PACK_F2I_WITH_RELU_1x8(a, i, j, relu, dst_zero_point)                     \
            PACK_F2I_WITH_RELU_1x8(a, i + 1, j, relu, dst_zero_point)         \
                    PACK_F2I_WITH_RELU_1x8(a, i + 2, j, relu, dst_zero_point) \
                            PACK_F2I_WITH_RELU_1x8(a, i + 3, j, relu, dst_zero_point)

#define STG(d, s, idx, n_reuse, hw_reuse, g)                                \
    n_reuse = nhw_post##idx / param.div_ohow;                               \
    hw_reuse = nhw_post##idx % param.div_ohow;                              \
    d = g_dst_ptr + n_reuse * param.obs + hw_reuse * (packed_channel >> 1); \
    g = nhw_post##idx < param.nhw;                                          \
    if (stg_oc < param.oc && g) {                                           \
        *(reinterpret_cast<int2*>(d)) = *(reinterpret_cast<int2*>(&s));     \
    }

#define STG_4x1(d, a, i, j)                                                     \
    STG(d[0], a[i][j], 0, reg_src_cache[0].x, reg_src_cache[1].x, stg_guard[i]) \
    STG(d[1], a[i + 1][j], 1, reg_src_cache[0].y, reg_src_cache[1].y,           \
        stg_guard[i + 1])                                                       \
    STG(d[2], a[i + 2][j], 2, reg_src_cache[0].z, reg_src_cache[1].z,           \
        stg_guard[i + 2])                                                       \
    STG(d[3], a[i + 3][j], 3, reg_src_cache[0].w, reg_src_cache[1].w, stg_guard[i + 3])

#define FUSE_Z_4x8(a, i, j, fuse_z, gamma, zero_point)         \
    fuse_z_1x8(a[i], j, fuse_z[i], gamma, zero_point);         \
    fuse_z_1x8(a[i + 1], j, fuse_z[i + 1], gamma, zero_point); \
    fuse_z_1x8(a[i + 2], j, fuse_z[i + 2], gamma, zero_point); \
    fuse_z_1x8(a[i + 3], j, fuse_z[i + 3], gamma, zero_point);

#define FUSE_Z_4x8(a, i, j, fuse_z, gamma, zero_point)         \
    fuse_z_1x8(a[i], j, fuse_z[i], gamma, zero_point);         \
    fuse_z_1x8(a[i + 1], j, fuse_z[i + 1], gamma, zero_point); \
    fuse_z_1x8(a[i + 2], j, fuse_z[i + 2], gamma, zero_point); \
    fuse_z_1x8(a[i + 3], j, fuse_z[i + 3], gamma, zero_point);

// 1x8 1x(2x8 int2) to 2 int2
#define PACK_F2I_1x8(a, i, j)                                                       \
    pack_f2i(a[i][j].x, a[i][j].z, a[i][j], a[i][j + 1], a[i][j + 2], a[i][j + 3]); \
    pack_f2i(a[i][j].y, a[i][j].w, a[i][j + 4], a[i][j + 5], a[i][j + 6], a[i][j + 7]);

// 4x8 int4
#define PACK_F2I_4x8(a, i, j)                                                 \
    PACK_F2I_1x8(a, i, j) PACK_F2I_1x8(a, i + 1, j) PACK_F2I_1x8(a, i + 2, j) \
            PACK_F2I_1x8(a, i + 3, j)

#define LDG(d, s, idx, n_reuse, hw_reuse, g)                    \
    n_reuse = nhw_post##idx / param.div_ohow;                   \
    hw_reuse = nhw_post##idx % param.div_ohow;                  \
    s = n_reuse * param.obs + hw_reuse * (packed_channel >> 1); \
    g = nhw_post##idx < param.nhw;                              \
    if (stg_oc < param.oc && g) {                               \
        *(reinterpret_cast<int2*>(&d)) =                        \
                *(reinterpret_cast<const int2*>(g_z_ptr + s));  \
    }

#define LDG_4x1(d, s, i)                                                     \
    LDG(d[i], s[i], 0, reg_src_cache[0].x, reg_src_cache[1].x, stg_guard[i]) \
    LDG(d[i + 1], s[i + 1], 1, reg_src_cache[0].y, reg_src_cache[1].y,       \
        stg_guard[i + 1])                                                    \
    LDG(d[i + 2], s[i + 2], 2, reg_src_cache[0].z, reg_src_cache[1].z,       \
        stg_guard[i + 2])                                                    \
    LDG(d[i + 3], s[i + 3], 3, reg_src_cache[0].w, reg_src_cache[1].w, stg_guard[i + 3])

#define COMPUTE_OFFSET(d, s, idx, n_reuse, hw_reuse, g)         \
    n_reuse = nhw_post##idx / param.div_ohow;                   \
    hw_reuse = nhw_post##idx % param.div_ohow;                  \
    s = n_reuse * param.obs + hw_reuse * (packed_channel >> 1); \
    g = nhw_post##idx < param.nhw;

#define COMPUTE_OFFSET_4x1(d, s, i)                                              \
    COMPUTE_OFFSET(                                                              \
            d[i], s[i], 0, reg_src_cache[0].x, reg_src_cache[1].x, stg_guard[i]) \
    COMPUTE_OFFSET(                                                              \
            d[i + 1], s[i + 1], 1, reg_src_cache[0].y, reg_src_cache[1].y,       \
            stg_guard[i + 1])                                                    \
    COMPUTE_OFFSET(                                                              \
            d[i + 2], s[i + 2], 2, reg_src_cache[0].z, reg_src_cache[1].z,       \
            stg_guard[i + 2])                                                    \
    COMPUTE_OFFSET(                                                              \
            d[i + 3], s[i + 3], 3, reg_src_cache[0].w, reg_src_cache[1].w,       \
            stg_guard[i + 3])

#define STG_AFTER_LDG(d, s, g)                                                      \
    if (stg_oc < param.oc && g) {                                                   \
        *(reinterpret_cast<int2*>(g_dst_ptr + d)) = *(reinterpret_cast<int2*>(&s)); \
    }

#define STG_AFTER_LDG_4x1(d, a, i, j)                      \
    STG_AFTER_LDG(d[i], a[i][j], stg_guard[i])             \
    STG_AFTER_LDG(d[i + 1], a[i + 1][j], stg_guard[i + 1]) \
    STG_AFTER_LDG(d[i + 2], a[i + 2][j], stg_guard[i + 2]) \
    STG_AFTER_LDG(d[i + 3], a[i + 3][j], stg_guard[i + 3])
// vim: syntax=cpp.doxygen
