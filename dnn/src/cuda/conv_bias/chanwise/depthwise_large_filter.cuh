#pragma once
namespace {
#define DIVUP(x, y) (((x) + (y)-1) / (y))
enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

template <typename ThreadConfig_, int oh_, int ow_>
struct OutTileConfig {
    using ThreadConfig = ThreadConfig_;
    static int const unroll_h = oh_;
    static int const unroll_w = ThreadConfig::thread_x * ow_;
    static int const unroll_size = unroll_h * unroll_w;
    static int const block_h = unroll_h * ThreadConfig::thread_y;
    static int const block_w = unroll_w;
};

template <int fh_, int fw_>
struct FilterTileConfig {
    static int const unroll_h = fh_;
    static int const unroll_w = fw_;
    static int const unroll_size = unroll_h * unroll_w;
};

template <int x_, int y_>
struct ThreadConfig {
    static int const thread_x = x_;
    static_assert((thread_x & (thread_x - 1)) == 0, "thread_x must be pow of 2!");
    static int const thread_y = y_;
    static int const nr_threads = x_ * y_;
};

template <
        typename ldg_dtype, typename ThreadConfig_, typename OutTileConfig_,
        typename FilterTileConfig_, int stride_w, int stride_h>
struct ConvTraitInner {
    using ThreadConfig = ThreadConfig_;
    using OutTileConfig = OutTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    using CompType = ldg_dtype;

    struct SrcTileConfig {
        static int const unroll_h =
                OutTileConfig::unroll_h + FilterTileConfig::unroll_h - 1;
        static int const unroll_w =
                (OutTileConfig::unroll_w - 1) * stride_w + FilterTileConfig::unroll_w;
        static int const unroll_size = unroll_h * unroll_w;
    };

    struct SrcTileCount {
        static int const smem_src_h =
                (OutTileConfig::block_h - 1) * stride_h + FilterTileConfig::unroll_h;
        static int const smem_buff_h = FilterTileConfig::unroll_h;
        static int const smem_load_h = smem_src_h + smem_buff_h *
                                                            FilterTileConfig::unroll_w *
                                                            ThreadConfig::thread_x;
        static int const smem_h = smem_load_h + smem_buff_h;
        static int const smem_w =
                DIVUP((OutTileConfig::block_w - 1) * stride_w +
                              FilterTileConfig::unroll_w * ThreadConfig::thread_x,
                      2) *
                2;
        static int const load_w =
                smem_w > ThreadConfig::nr_threads ? ThreadConfig::nr_threads : smem_w;
        static int const load_h = 1;
        static int const reg_h = 1;
        static int const reg_w = DIVUP(smem_w, load_w);
        static bool constexpr check_bounds_h = smem_h % load_h != 0;
        static bool constexpr check_bounds_w = smem_w % load_w != 0;
        // to avoid bank confilct, every bank_offset_line in 8 lines, add one offset
        static int const bank_w = smem_w / (4 / sizeof(CompType));
        static int const bank_offset_line =
                (bank_w % 32 == 0 || bank_w % FilterTileConfig::unroll_w == 0)
                        ? 1
                        : (bank_w % 16 == 0 ? 2 : 4);
        static int const smem_size = smem_h * smem_w + DIVUP(smem_h, bank_offset_line) *
                                                               (4 / sizeof(CompType));
    };

    struct FilterTileCount {
        static int const smem_flt_h = FilterTileConfig::unroll_h;
        static int const smem_buff_h = FilterTileConfig::unroll_h;
        static int const smem_w = FilterTileConfig::unroll_w * ThreadConfig::thread_x;
        static int const smem_load_h = smem_flt_h + smem_buff_h * smem_w;
        static int const smem_h = smem_load_h + smem_buff_h;
        static int const load_w = smem_w > 32 ? 32 : smem_w;
        static int const load_h = ThreadConfig::nr_threads / load_w;
        static int const reg_h = 1;
        static int const reg_w = DIVUP(smem_w, load_w);
        static bool constexpr check_bounds_h = smem_h % load_h != 0;
        static bool constexpr check_bounds_w = smem_w % load_w != 0;
        // to avoid bank confilct, every bank_offset_line in 8 lines, add one offset
        static int const bank_w = smem_w / (4 / sizeof(CompType));
        static int const bank_offset_line =
                (bank_w % 32 == 0 || bank_w % FilterTileConfig::unroll_w == 0)
                        ? 1
                        : (bank_w % 16 == 0 ? 2 : 4);
        static int const smem_size = smem_h * smem_w + DIVUP(smem_h, bank_offset_line) *
                                                               (4 / sizeof(CompType));
    };
};

#define CHECK_AB_FWD(a, b)                                                             \
    if (param.out_w > b * 4) {                                                         \
        if (param.stride_h == 1 && param.stride_w == 1) {                              \
            using FilterTileConfig_ = FilterTileConfig<unroll_fh, a + 2>;              \
            using ThreadConfig_ = ThreadConfig<4, 32>;                                 \
            using OutTileConfig_ = OutTileConfig<ThreadConfig_, unroll_oh, b + 1>;     \
            using IConvTrait = ConvTraitInner<                                         \
                    float, ThreadConfig_, OutTileConfig_, FilterTileConfig_, 1, 1>;    \
            using SrcTileConfig = typename IConvTrait::SrcTileConfig;                  \
            using SrcTileCount = typename IConvTrait::SrcTileCount;                    \
            using FilterTileCount = typename IConvTrait::FilterTileCount;              \
                                                                                       \
            if (device_prop.regsPerBlock <                                             \
                        4 * 32 *                                                       \
                                (FilterTileConfig_::unroll_h *                         \
                                         FilterTileConfig_::unroll_w * 2 +             \
                                 SrcTileConfig::unroll_h * SrcTileConfig::unroll_w) || \
                device_prop.sharedMemPerBlock <                                        \
                        static_cast<size_t>(                                           \
                                (SrcTileCount::smem_size +                             \
                                 FilterTileCount::smem_size))) {                       \
                return false;                                                          \
            }                                                                          \
            return true;                                                               \
        } else if (param.stride_h == 2 && param.stride_w == 2) {                       \
            using FilterTileConfig_ = FilterTileConfig<unroll_fh, a + 2>;              \
            using ThreadConfig_ = ThreadConfig<4, 32>;                                 \
            using OutTileConfig_ = OutTileConfig<ThreadConfig_, unroll_oh, b + 1>;     \
            using IConvTrait = ConvTraitInner<                                         \
                    float, ThreadConfig_, OutTileConfig_, FilterTileConfig_, 2, 2>;    \
            using SrcTileConfig = typename IConvTrait::SrcTileConfig;                  \
            using SrcTileCount = typename IConvTrait::SrcTileCount;                    \
            using FilterTileCount = typename IConvTrait::FilterTileCount;              \
                                                                                       \
            if (device_prop.regsPerBlock <                                             \
                        4 * 32 *                                                       \
                                (FilterTileConfig_::unroll_h *                         \
                                         FilterTileConfig_::unroll_w * 2 +             \
                                 SrcTileConfig::unroll_h * SrcTileConfig::unroll_w) || \
                device_prop.sharedMemPerBlock <                                        \
                        static_cast<size_t>(                                           \
                                (SrcTileCount::smem_size +                             \
                                 FilterTileCount::smem_size))) {                       \
                return false;                                                          \
            }                                                                          \
            return true;                                                               \
        }                                                                              \
    }

#define CHECK_AB_BWD(a, b)                                                             \
    if (param.out_w > b * 4 || b == 3) {                                               \
        using FilterTileConfig_ = FilterTileConfig<unroll_fh, a + 2>;                  \
        using ThreadConfig_ = ThreadConfig<4, 32>;                                     \
        using OutTileConfig_ = OutTileConfig<ThreadConfig_, unroll_oh, b + 1>;         \
        using IConvTrait = ConvTraitInner<                                             \
                float, ThreadConfig_, OutTileConfig_, FilterTileConfig_, 1, 1>;        \
        using SrcTileConfig = typename IConvTrait::SrcTileConfig;                      \
        using SrcTileCount = typename IConvTrait::SrcTileCount;                        \
        using FilterTileCount = typename IConvTrait::FilterTileCount;                  \
                                                                                       \
        if (device_prop.regsPerBlock <                                                 \
                    4 * 32 *                                                           \
                            (FilterTileConfig_::unroll_h *                             \
                                     FilterTileConfig_::unroll_w * 2 +                 \
                             SrcTileConfig::unroll_h * SrcTileConfig::unroll_w) ||     \
            device_prop.sharedMemPerBlock <                                            \
                    static_cast<size_t>(                                               \
                            (SrcTileCount::smem_size + FilterTileCount::smem_size))) { \
            return false;                                                              \
        }                                                                              \
        return true;                                                                   \
    }

#define CHECK_A(a, cb)                                                         \
    if (param.flt_w > a * 4) {                                                 \
        CHECK_AB_##cb(a, 15) else CHECK_AB_##cb(a, 7) else CHECK_AB_##cb(a, 3) \
    }

#define CHECK(cb)  \
    CHECK_A(6, cb) \
    else CHECK_A(4, cb) else CHECK_A(2, cb) else CHECK_A(0, cb)

}  // namespace
