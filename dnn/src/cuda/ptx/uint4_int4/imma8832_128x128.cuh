#pragma once

#include "./base.cuh"

#define TX             128
#define TY             1
#define BM             128
#define BN             128
#define BK             128
#define mma_m          16
#define mma_n          8
#define mma_k          64
#define reg_m          8
#define reg_n          8
#define packed_channel 64
#define BKd32          (BK / 32)
#define BKd64          (BK / 64)
#define reg_md4        (reg_m >> 2)
#define WARPS          (TX / 32)
#define cache_per_warp 128
#define reg_nd4        (reg_n >> 2)
#define ldg_src        (BN * BK / (16 * TX))
#define ldg_filter     (BM * BK / (16 * TX))
#define ldg_width      16

// vim: syntax=cpp.doxygen
