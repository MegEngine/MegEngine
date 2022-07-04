#pragma once
namespace {
#define DIVUP(x, y) (((x) + (y)-1) / (y))
enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

template <typename ThreadConfig_, int oh_, int ow_>
struct OutTileConfig {
    using ThreadConfig = ThreadConfig_;
    static int constexpr unroll_h = oh_;
    static int constexpr unroll_w = ThreadConfig::thread_x * ow_;
    static int constexpr unroll_size = unroll_h * unroll_w;
    static int constexpr block_h = unroll_h * ThreadConfig::thread_y;
    static int constexpr block_w = unroll_w;
};

template <int fh_, int fw_>
struct FilterTileConfig {
    static int constexpr unroll_h = fh_;
    static int constexpr unroll_w = fw_;
    static int constexpr unroll_size = unroll_h * unroll_w;
};

template <int x_, int y_>
struct ThreadConfig {
    static int constexpr thread_x = x_;
    static_assert((thread_x & (thread_x - 1)) == 0, "thread_x must be pow of 2!");
    static int constexpr thread_y = y_;
    static int constexpr nr_threads = x_ * y_;
};

template <
        typename ldg_dtype, typename Rldg_dtype, typename Rcmp_dtype,
        typename ThreadConfig_, typename OutTileConfig_, typename FilterTileConfig_,
        int stride_w, int stride_h>
struct ConvTraitInner {
    using ThreadConfig = ThreadConfig_;
    using OutTileConfig = OutTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    using CompType = ldg_dtype;

    struct SrcTileConfig {
        static int constexpr unroll_h =
                OutTileConfig::unroll_h + FilterTileConfig::unroll_h - 1;
        static int constexpr unroll_w =
                (OutTileConfig::unroll_w - 1) * stride_w + FilterTileConfig::unroll_w;
        static int constexpr unroll_size = unroll_h * unroll_w;
    };

    struct SrcTileCount {
        static int constexpr smem_src_h =
                (OutTileConfig::block_h - 1) * stride_h + FilterTileConfig::unroll_h;
        static int constexpr smem_delta_h = 2;
        static int constexpr smem_buff_h =
                FilterTileConfig::unroll_h * smem_delta_h * 2;
        static int constexpr smem_load_h = smem_src_h + smem_buff_h;
        static int constexpr smem_h = smem_load_h;
        static int constexpr smem_w =
                DIVUP((OutTileConfig::block_w - 1) * stride_w +
                              FilterTileConfig::unroll_w * ThreadConfig::thread_x,
                      2) *
                2;
        static int constexpr load_w = smem_w > 32 ? 32 : smem_w;
        static int constexpr load_h = ThreadConfig::nr_threads / load_w;
        static int constexpr reg_h = DIVUP(smem_delta_h, load_h);
        static int constexpr reg_w = DIVUP(smem_w, load_w);
        static bool constexpr check_bounds_h = smem_delta_h % load_h != 0;
        static bool constexpr check_bounds_w = smem_w % load_w != 0;
        // to avoid bank confilct, every bank_offset_line in 8 lines, add one offset
        static int constexpr bank_w = smem_w / (4 / sizeof(CompType));
        static int constexpr bank_offset_line =
                (bank_w % 32 == 0 || bank_w % FilterTileConfig::unroll_w == 0)
                        ? 1
                        : (bank_w % 16 == 0 ? 2 : 4);
        static int constexpr smem_size =
                smem_h * smem_w +
                DIVUP(smem_h, bank_offset_line) * (4 / sizeof(CompType));
    };

    struct FilterTileCount {
        static int constexpr smem_flt_h = FilterTileConfig::unroll_h;
        static int constexpr smem_buff_h = FilterTileConfig::unroll_h;
        static int constexpr smem_w =
                FilterTileConfig::unroll_w * ThreadConfig::thread_x;
        static int constexpr smem_delta_h = 2;
        static int constexpr smem_load_h = smem_flt_h + smem_buff_h * smem_w;
        static int constexpr smem_h = smem_load_h + smem_buff_h;
        static int constexpr load_w = smem_w > 32 ? 32 : smem_w;
        static int constexpr load_h = ThreadConfig::nr_threads / load_w;
        static int constexpr reg_h = 1;
        static int constexpr reg_w = DIVUP(smem_w, load_w);
        static bool constexpr check_bounds_h = smem_h % load_h != 0;
        static bool constexpr check_bounds_w = smem_w % load_w != 0;
        // to avoid bank confilct, every bank_offset_line in 8 lines, add one offset
        static int constexpr bank_w = smem_w / (4 / sizeof(CompType));
        static int constexpr bank_offset_line =
                (bank_w % 32 == 0 || bank_w % FilterTileConfig::unroll_w == 0)
                        ? 1
                        : (bank_w % 16 == 0 ? 2 : 4);
        static int constexpr smem_size =
                smem_h * smem_w +
                DIVUP(smem_h, bank_offset_line) * (4 / sizeof(CompType));
    };

    struct RinTileCount {
        static int constexpr smem_src_h =
                (OutTileConfig::block_h - 1) * stride_h + FilterTileConfig::unroll_h;
        static int constexpr smem_delta_h = 2;
        static int constexpr smem_buff_h =
                FilterTileConfig::unroll_h * smem_delta_h * 2;
        static int constexpr smem_load_h = smem_src_h + smem_buff_h;
        static int constexpr smem_h = smem_load_h;
        static int constexpr factor = sizeof(Rldg_dtype) / sizeof(Rcmp_dtype);
        static int constexpr smem_w =
                DIVUP(DIVUP((OutTileConfig::block_w - 1) * stride_w +
                                    FilterTileConfig::unroll_w * ThreadConfig::thread_x,
                            factor),
                      2) *
                2;
        static int constexpr load_w = smem_w > 32 ? 32 : smem_w;
        static int constexpr load_h = ThreadConfig::nr_threads / load_w;
        static int constexpr reg_h = DIVUP(smem_delta_h, load_h);
        static int constexpr reg_w = DIVUP(smem_w, load_w);
        static bool constexpr check_bounds_h = smem_delta_h % load_h != 0;
        static bool constexpr check_bounds_w = smem_w % load_w != 0;
        // to avoid bank confilct, every bank_offset_line in 8 lines, add one offset
        static int constexpr bank_w = smem_w;
        static int constexpr bank_offset_line =
                (bank_w % 32 == 0 || bank_w % FilterTileConfig::unroll_w == 0)
                        ? 1
                        : (bank_w % 16 == 0 ? 2 : 4);
        static int constexpr smem_size =
                smem_h * smem_w + DIVUP(smem_h, bank_offset_line);
    };
};

}  // namespace
