#pragma once
#include "src/common/unroll_macro.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"

namespace megdnn {
namespace {

struct Vld1qF32S {
    static GI_FORCEINLINE GI_FLOAT32_t impl(const float32_t* ptr) {
        return GiLoadFloat32(ptr);
    }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"

#ifdef __GNUC__
#ifndef __has_warning
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#else
#if __has_warning("-Wmaybe-uninitialized")
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
#endif

template <typename ctype>
struct ParamElemFixLenVisitor;

template <typename ctype>
struct ParamElemFixLenVisitorV2;

#define cb(_ctype, _simd_type, _simd_fixlen_type, _fun_suffix, _simd_type_v2)     \
    template <>                                                                   \
    struct ParamElemFixLenVisitor<_ctype> {                                       \
        _simd_type operator()(const _simd_fixlen_type& s0) const {                \
            return GiFixLenType2Gi##_fun_suffix##Type(s0);                        \
        }                                                                         \
    };                                                                            \
    template <>                                                                   \
    struct ParamElemFixLenVisitorV2<_ctype> {                                     \
        _simd_type_v2 operator()(                                                 \
                const _simd_fixlen_type& s0, const _simd_fixlen_type& s1) const { \
            _simd_type_v2 ret;                                                    \
            GiSetSubVector##_fun_suffix##V2(                                      \
                    ret, 0, GiFixLenType2Gi##_fun_suffix##Type(s0));              \
            GiSetSubVector##_fun_suffix##V2(                                      \
                    ret, 1, GiFixLenType2Gi##_fun_suffix##Type(s1));              \
            return ret;                                                           \
        }                                                                         \
    };

cb(dt_float32, GI_FLOAT32_t, GI_FLOAT32_FIXLEN_t, Float32, GI_FLOAT32_V2_t);
#undef cb

template <
        int weight_number, int base_offset, int ptr_step, int oc_block, typename Func,
        typename T, typename T2, typename... XT>
struct LoadHelper {
    static GI_FORCEINLINE void impl(T& weight, T2 ptr, int oc_offset, XT... args);
};

template <
        int weight_number, int base_offset, int ptr_step, int oc_block, typename T,
        typename T2>
struct LoadPtrHelper {
    static GI_FORCEINLINE void impl(T& weight, T2 ptr, int oc_offset);
};

#define WEIGHT_CB(step)                   \
    src[step] = GiFloat32Type2FixLenType( \
            Func::impl(ptr + base_offset + step * ptr_step, args...));

#define LOAD_HELPER(step)                                                          \
    template <                                                                     \
            int base_offset, int ptr_step, typename Func, typename T, typename T2, \
            typename... XT>                                                        \
    struct LoadHelper<step, base_offset, ptr_step, 0, Func, T, T2, XT...> {        \
        static GI_FORCEINLINE void impl(T& src, T2 ptr, int, XT... args) {         \
            UNROLL_CALL_RAW(step, WEIGHT_CB);                                      \
        }                                                                          \
    }

LOAD_HELPER(1);
LOAD_HELPER(2);
LOAD_HELPER(3);
LOAD_HELPER(4);
LOAD_HELPER(5);
LOAD_HELPER(6);
LOAD_HELPER(7);
LOAD_HELPER(8);
LOAD_HELPER(9);
LOAD_HELPER(10);
LOAD_HELPER(11);
LOAD_HELPER(12);
LOAD_HELPER(13);
LOAD_HELPER(14);
LOAD_HELPER(15);
LOAD_HELPER(16);

#undef LOAD_HELPER
#undef WEIGHT_CB

#define WEIGHT_PTR_CB(step) src[step] = ptr + base_offset + step * ptr_step;

#define LOAD_PTR_HELPER(step)                                         \
    template <int base_offset, int ptr_step, typename T, typename T2> \
    struct LoadPtrHelper<step, base_offset, ptr_step, 0, T, T2> {     \
        static GI_FORCEINLINE void impl(T& src, T2 ptr, int) {        \
            UNROLL_CALL_RAW(step, WEIGHT_PTR_CB);                     \
        }                                                             \
    }

LOAD_PTR_HELPER(1);
LOAD_PTR_HELPER(2);
LOAD_PTR_HELPER(3);
LOAD_PTR_HELPER(4);
LOAD_PTR_HELPER(5);
LOAD_PTR_HELPER(6);
LOAD_PTR_HELPER(7);
LOAD_PTR_HELPER(8);
LOAD_PTR_HELPER(9);
LOAD_PTR_HELPER(10);
LOAD_PTR_HELPER(11);
LOAD_PTR_HELPER(12);
LOAD_PTR_HELPER(13);
LOAD_PTR_HELPER(14);
LOAD_PTR_HELPER(15);
LOAD_PTR_HELPER(16);

#undef LOAD_PTR_HELPER
#undef WEIGHT_PTR_CB

///////////////////////////c_dim = 1/////////////////////////
#define WEIGHT_CB(step) \
    src[0][step] =      \
            GiFloat32Type2FixLenType(Func::impl(ptr + base_offset + step * ptr_step));

#define LOAD_HELPER(step)                                                            \
    template <int base_offset, int ptr_step, typename Func, typename T, typename T2> \
    struct LoadHelper<step, base_offset, ptr_step, 1, Func, T, T2> {                 \
        static GI_FORCEINLINE void impl(T& src, T2 ptr, int) {                       \
            UNROLL_CALL_RAW(step, WEIGHT_CB);                                        \
        }                                                                            \
    }

LOAD_HELPER(1);
LOAD_HELPER(2);
LOAD_HELPER(3);
LOAD_HELPER(4);
LOAD_HELPER(5);
LOAD_HELPER(6);
LOAD_HELPER(7);
LOAD_HELPER(8);
LOAD_HELPER(9);

#undef LOAD_HELPER
#undef WEIGHT_CB

#define WEIGHT_PTR_CB(step) src[0][step] = ptr + base_offset + step * ptr_step;

#define LOAD_PTR_HELPER(step)                                         \
    template <int base_offset, int ptr_step, typename T, typename T2> \
    struct LoadPtrHelper<step, base_offset, ptr_step, 1, T, T2> {     \
        static GI_FORCEINLINE void impl(T& src, T2 ptr, int) {        \
            UNROLL_CALL_RAW(step, WEIGHT_PTR_CB);                     \
        }                                                             \
    }

LOAD_PTR_HELPER(1);
LOAD_PTR_HELPER(2);
LOAD_PTR_HELPER(3);
LOAD_PTR_HELPER(4);
LOAD_PTR_HELPER(5);
LOAD_PTR_HELPER(6);
LOAD_PTR_HELPER(7);
LOAD_PTR_HELPER(8);
LOAD_PTR_HELPER(9);

#undef LOAD_PTR_HELPER
#undef WEIGHT_PTR_CB

/////////////////////////c_dim = 2///////////////////////////////
#define WEIGHT_CB(step)                                                                \
    src[0][step] =                                                                     \
            GiFloat32Type2FixLenType(Func::impl(ptr + base_offset + step * ptr_step)); \
    src[1][step] = GiFloat32Type2FixLenType(                                           \
            Func::impl(ptr + base_offset + step * ptr_step + oc_offset));

#define LOAD_HELPER(step)                                                            \
    template <int base_offset, int ptr_step, typename Func, typename T, typename T2> \
    struct LoadHelper<step, base_offset, ptr_step, 2, Func, T, T2> {                 \
        static GI_FORCEINLINE void impl(T& src, T2 ptr, int oc_offset) {             \
            UNROLL_CALL_RAW(step, WEIGHT_CB);                                        \
        }                                                                            \
    }

LOAD_HELPER(1);
LOAD_HELPER(2);
LOAD_HELPER(3);
LOAD_HELPER(4);
LOAD_HELPER(5);
LOAD_HELPER(6);
LOAD_HELPER(7);
LOAD_HELPER(8);

#undef LOAD_HELPER
#undef WEIGHT_CB

#define WEIGHT_PTR_CB(step)                             \
    src[0][step] = ptr + base_offset + step * ptr_step; \
    src[1][step] = ptr + base_offset + step * ptr_step + oc_offset;

#define LOAD_PTR_HELPER(step)                                            \
    template <int base_offset, int ptr_step, typename T, typename T2>    \
    struct LoadPtrHelper<step, base_offset, ptr_step, 2, T, T2> {        \
        static GI_FORCEINLINE void impl(T& src, T2 ptr, int oc_offset) { \
            UNROLL_CALL_RAW(step, WEIGHT_PTR_CB);                        \
        }                                                                \
    }

LOAD_PTR_HELPER(1);
LOAD_PTR_HELPER(2);
LOAD_PTR_HELPER(3);
LOAD_PTR_HELPER(4);
LOAD_PTR_HELPER(5);
LOAD_PTR_HELPER(6);
LOAD_PTR_HELPER(7);
LOAD_PTR_HELPER(8);

#undef LOAD_HELPER
#undef WEIGHT_PTR_CB

template <
        int weight_number, int base_offset, int ptr_step, int c_dim, typename Func,
        typename T, typename T2>
GI_FORCEINLINE void load_helper(T& weight, T2 ptr, int oc_offset) {
    LoadHelper<weight_number, base_offset, ptr_step, c_dim, Func, T, T2>::impl(
            weight, ptr, oc_offset);
}

template <
        int weight_number, int base_offset, int ptr_step, int c_dim, typename T,
        typename T2>
GI_FORCEINLINE void load_ptr_helper(T& weight, T2 ptr, int oc_offset) {
    LoadPtrHelper<weight_number, base_offset, ptr_step, c_dim, T, T2>::impl(
            weight, ptr, oc_offset);
}

////////////////////Store_OCX_OW8_Remain/////////////////////////
template <int c_dim, int ow_remain, typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc);
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 0, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
        op(vis(c[0][6]), reinterpret_cast<T3>(dst_ptr + 24));
        op(vis(c[0][7]), reinterpret_cast<T3>(dst_ptr + 28));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(vis(c[1][3]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 12));
        op(vis(c[1][4]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op(vis(c[1][5]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 20));
        op(vis(c[1][6]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 24));
        op(vis(c[1][7]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 28));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 8, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
        op(vis(c[0][6]), reinterpret_cast<T3>(dst_ptr + 24));
        op(vis(c[0][7]), reinterpret_cast<T3>(dst_ptr + 28));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(vis(c[1][3]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 12));
        op(vis(c[1][4]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op(vis(c[1][5]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 20));
        op(vis(c[1][6]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 24));
        op(vis(c[1][7]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 28));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 7, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
        op(vis(c[0][6]), reinterpret_cast<T3>(dst_ptr + 24));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(vis(c[1][3]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 12));
        op(vis(c[1][4]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op(vis(c[1][5]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 20));
        op(vis(c[1][6]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 6, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(vis(c[1][3]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 12));
        op(vis(c[1][4]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op(vis(c[1][5]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 20));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 5, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(vis(c[1][3]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 12));
        op(vis(c[1][4]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 4, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(vis(c[1][3]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 12));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 3, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
        op(vis(c[1][2]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 2, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));

        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(vis(c[1][1]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 4));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 1, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[1][0]), reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
    }
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 0, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
        op(vis(c[0][6]), reinterpret_cast<T3>(dst_ptr + 24));
        op(vis(c[0][7]), reinterpret_cast<T3>(dst_ptr + 28));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 8, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
        op(vis(c[0][6]), reinterpret_cast<T3>(dst_ptr + 24));
        op(vis(c[0][7]), reinterpret_cast<T3>(dst_ptr + 28));
    }
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 7, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
        op(vis(c[0][6]), reinterpret_cast<T3>(dst_ptr + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 6, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
        op(vis(c[0][5]), reinterpret_cast<T3>(dst_ptr + 20));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 5, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
        op(vis(c[0][4]), reinterpret_cast<T3>(dst_ptr + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 4, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
        op(vis(c[0][3]), reinterpret_cast<T3>(dst_ptr + 12));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 3, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
        op(vis(c[0][2]), reinterpret_cast<T3>(dst_ptr + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 2, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis;
        op(vis(c[0][0]), reinterpret_cast<T3>(dst_ptr));
        op(vis(c[0][1]), reinterpret_cast<T3>(dst_ptr + 4));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 1, Op, T, T2, T3> {
    static GI_FORCEINLINE void impl(T& c, const Op& op, T2 dst_ptr, int) {
        ParamElemFixLenVisitor<typename Op::src_ctype> vis0;
        op(vis0(c[0][0]), reinterpret_cast<T3>(dst_ptr));
    }
};

template <int c_dim, int ow_remain, typename Op, typename T, typename T2>
GI_FORCEINLINE void store_ocx_ow8_remain_static(
        T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
    StoreOcxOw8Remain<c_dim, ow_remain, Op, T, T2, T2>::impl(c, op, dst_ptr, ld_dst_oc);
}

#undef cb
#undef cb2
#undef cb_case
#undef cb_case2

#pragma GCC diagnostic pop

/////////////////////////init_ocx_ow8////////////////////

template <typename T>
struct GiLdqSimd;
template <>
struct GiLdqSimd<float> {
    static constexpr int simd_len = 4;
};
template <int c_dim, BiasMode bias_mode, int ow_remain, typename T, typename T2>
struct InitOcxOw8 {
    static GI_FORCEINLINE void impl(T& c, const T2* bias_ptr, int oc_step);
};
template <int c_dim, BiasMode bias_mode, typename T, typename T2>
struct InitOcxOw8<c_dim, bias_mode, 0, T, T2> {
    static GI_FORCEINLINE void impl(T&, const T2*, int) {}
};

#define BAIS_INIT_NO_BIAS_C2(step)                                                 \
    c[0][step] = GiFloat32Type2FixLenType(GiBroadcastFloat32(static_cast<T2>(0))); \
    c[1][step] = GiFloat32Type2FixLenType(GiBroadcastFloat32(static_cast<T2>(0)));
#define BAIS_INIT_NO_BIAS_C1(step) \
    c[0][step] = GiFloat32Type2FixLenType(GiBroadcastFloat32(static_cast<T2>(0)));

#define BAIS_INIT_BROADCAST_C2(step)                                \
    c[0][step] = GiFloat32Type2FixLenType(GiLoadFloat32(bias_ptr)); \
    c[1][step] = GiFloat32Type2FixLenType(GiLoadFloat32(bias_ptr + oc_step));
#define BAIS_INIT_BROADCAST_C1(step) \
    c[0][step] = GiFloat32Type2FixLenType(GiLoadFloat32(bias_ptr));

#define BAIS_INIT_BIAS_C2(step)                                                       \
    c[0][step] = GiFloat32Type2FixLenType(GiLoadFloat32(bias_ptr + step * simd_len)); \
    c[1][step] = GiFloat32Type2FixLenType(                                            \
            GiLoadFloat32(bias_ptr + oc_step + step * simd_len));

#define BAIS_INIT_BIAS_C1(step) \
    c[0][step] = GiFloat32Type2FixLenType(GiLoadFloat32(bias_ptr + step * simd_len));

#define INSTANCE_InitOcxOw8(ow_remain, cdim)                                      \
    template <typename T, typename T2>                                            \
    struct InitOcxOw8<cdim, BiasMode::NO_BIAS, ow_remain, T, T2> {                \
        static GI_FORCEINLINE void impl(T& c, const T2*, int) {                   \
            UNROLL_CALL_RAW(ow_remain, BAIS_INIT_NO_BIAS_C##cdim);                \
        }                                                                         \
    };                                                                            \
    template <typename T, typename T2>                                            \
    struct InitOcxOw8<cdim, BiasMode::BROADCAST_CHANNEL_BIAS, ow_remain, T, T2> { \
        static GI_FORCEINLINE void impl(T& c, const T2* bias_ptr, int oc_step) {  \
            (void)oc_step;                                                        \
            UNROLL_CALL_RAW(ow_remain, BAIS_INIT_BROADCAST_C##cdim);              \
        }                                                                         \
    };                                                                            \
    template <typename T, typename T2>                                            \
    struct InitOcxOw8<cdim, BiasMode::BIAS, ow_remain, T, T2> {                   \
        static GI_FORCEINLINE void impl(T& c, const T2* bias_ptr, int oc_step) {  \
            constexpr int simd_len = GiLdqSimd<T2>::simd_len;                     \
            (void)oc_step;                                                        \
            UNROLL_CALL_RAW(ow_remain, BAIS_INIT_BIAS_C##cdim);                   \
        }                                                                         \
    };
#define INSTANCE_InitOcxOw8_C(ow_remain) \
    INSTANCE_InitOcxOw8(ow_remain, 2);   \
    INSTANCE_InitOcxOw8(ow_remain, 1);

INSTANCE_InitOcxOw8_C(1);
INSTANCE_InitOcxOw8_C(2);
INSTANCE_InitOcxOw8_C(3);
INSTANCE_InitOcxOw8_C(4);
INSTANCE_InitOcxOw8_C(5);
INSTANCE_InitOcxOw8_C(6);
INSTANCE_InitOcxOw8_C(7);
INSTANCE_InitOcxOw8_C(8);

#undef INSTANCE_InitOcxOw8
#undef INSTANCE_InitOcxOw8_C
#undef BAIS_INIT_BIAS_C1
#undef BAIS_INIT_BIAS_C2
#undef BAIS_INIT_BROADCAST_C1
#undef BAIS_INIT_BROADCAST_C2
#undef BAIS_INIT_NO_BIAS_C1
#undef BAIS_INIT_NO_BIAS_C2

template <int c_dim, BiasMode bias_mode, int ow_remain, typename T, typename T2>
GI_FORCEINLINE void init_ocx_ow8(T& c, const T2* bias_ptr, int oc_step) {
    InitOcxOw8<c_dim, bias_mode, ow_remain, T, T2>::impl(c, bias_ptr, oc_step);
}

}  // namespace
}  // namespace megdnn
#undef GI_FORCEINLINE
// vim: syntax=cpp.doxygen
