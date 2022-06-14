#pragma once
#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/resize/opr_impl.h"

namespace megdnn {
namespace fallback {
namespace resize {

using InterpolationMode = Resize::InterpolationMode;

template <typename ctype>
struct SIMDHelper {};

template <>
struct SIMDHelper<float> {
    using simd_type = GI_FLOAT32_t;
    using simd_fixlen_type = GI_FLOAT32_FIXLEN_t;
    using simd_type_x2 = GI_FLOAT32_V2_t;
    using simd_type_x4 = GI_FLOAT32_V4_t;
    using ctype = float;
    static constexpr size_t simd_width = 4;

    static GI_FORCEINLINE simd_type load(const ctype* src_ptr) {
        return GiLoadFloat32(src_ptr);
    }
    static GI_FORCEINLINE void store(ctype* dst_ptr, const simd_type& rdst) {
        GiStoreFloat32(dst_ptr, rdst);
    }
    static GI_FORCEINLINE void store2_interleave(
            ctype* dst_ptr, const simd_type& rdst1, const simd_type& rdst2) {
        simd_type_x2 rdst;
        GiSetSubVectorFloat32V2(rdst, 0, rdst1);
        GiSetSubVectorFloat32V2(rdst, 1, rdst2);
        GiStoreZipFloat32V2(dst_ptr, rdst);
    }
    static GI_FORCEINLINE simd_type
    fma(const simd_type& a, const simd_type& b, ctype n) {
        return GiMultiplyAddScalarFloat32(a, b, n);
    }
    static GI_FORCEINLINE simd_type
    fma(const simd_type& a, const simd_type& b, const simd_type& c) {
        return GiMlaqFloat32(a, b, c);
    }
    static GI_FORCEINLINE simd_type dup(float val) { return GiBroadcastFloat32(val); }
};

static GI_FORCEINLINE int get_nearest_src(float scale, int size, int idx) {
    return std::min(static_cast<int>(idx / scale), size - 1);
}

static GI_FORCEINLINE std::tuple<float, int, float, int> get_nearest_linear_coord(
        InterpolationMode imode, float scale, int size, int idx) {
    if (size == 1) {
        return std::make_tuple(1.0f, 0, 0.0f, 0);
    }

    float alpha = (idx + 0.5f) / scale - 0.5f;
    int origin_idx = static_cast<int>(floor(alpha));
    alpha -= origin_idx;

    if (imode == InterpolationMode::INTER_NEAREST) {
        origin_idx = get_nearest_src(scale, size, idx);
        alpha = 0;
    }

    if (origin_idx < 0) {
        origin_idx = 0;
        alpha = 0;
    } else if (origin_idx + 1 >= size) {
        origin_idx = size - 2;
        alpha = 1;
    }

    return std::make_tuple(1 - alpha, origin_idx, alpha, origin_idx + 1);
}
};  // namespace resize
};  // namespace fallback
};  // namespace megdnn
