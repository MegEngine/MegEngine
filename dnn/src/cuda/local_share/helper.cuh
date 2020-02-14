/**
 * \file dnn/src/cuda/local_share/helper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace local_share {

struct Param {
    int n, co, ci, hi, wi, ph, pw, grp_ho, grp_wo, sgh, sgw;
};

struct LaunchConfig {
    int nr_threads_x;
    int nr_threads_y;
    int nr_threads_z;
    int nr_blocks_x;
    int nr_blocks_y;
    int nr_blocks_z;
    int smem_size_in_bytes;
    LaunchConfig()
            : nr_threads_x{1},
              nr_threads_y{1},
              nr_threads_z{1},
              nr_blocks_x{1},
              nr_blocks_y{1},
              nr_blocks_z{1},
              smem_size_in_bytes{1} {}
};

template <int fh_, int fw_, int sh_, int sw_>
struct LocalShareConfig {
    static int const fh = fh_;
    static int const fw = fw_;
    static int const sh = sh_;
    static int const sw = sw_;
};

void _check_launch_config(const LaunchConfig& launch_config);

uint32_t _get_kern_block_size(const void* kern);

}  // namespace local_share
}  // namespace cuda
}  // namespace megdnn

#define unpack_local_share_params(_src, _filter, _dst, _param)            \
    size_t n = _src[0], ci = _src[1], hi = _src[2], wi = _src[3];         \
    size_t weight_spatial_pos;                                            \
    if (_param.sparse == LocalShare::Param::Sparse::DENSE) {              \
        weight_spatial_pos = 3;                                           \
    } else {                                                              \
        megdnn_assert(_param.sparse == LocalShare::Param::Sparse::GROUP); \
        weight_spatial_pos = 4;                                           \
    }                                                                     \
    size_t fh = _filter[weight_spatial_pos],                              \
           fw = _filter[weight_spatial_pos + 1];                          \
    size_t co = _dst[1], ho = _dst[2], wo = _dst[3];                      \
    size_t ph = _param.pad_h, pw = _param.pad_w;                          \
    size_t sh = _param.stride_h, sw = _param.stride_w;                    \
    size_t dh = _param.dilate_h, dw = _param.dilate_w;                    \
    size_t sgh = _param.spatial_groups_h, sgw = _param.spatial_groups_w;  \
    MEGDNN_MARK_USED_VAR(n);                                              \
    MEGDNN_MARK_USED_VAR(ci);                                             \
    MEGDNN_MARK_USED_VAR(hi);                                             \
    MEGDNN_MARK_USED_VAR(wi);                                             \
    MEGDNN_MARK_USED_VAR(co);                                             \
    MEGDNN_MARK_USED_VAR(fh);                                             \
    MEGDNN_MARK_USED_VAR(fw);                                             \
    MEGDNN_MARK_USED_VAR(ho);                                             \
    MEGDNN_MARK_USED_VAR(wo);                                             \
    MEGDNN_MARK_USED_VAR(ph);                                             \
    MEGDNN_MARK_USED_VAR(pw);                                             \
    MEGDNN_MARK_USED_VAR(sh);                                             \
    MEGDNN_MARK_USED_VAR(sw);                                             \
    MEGDNN_MARK_USED_VAR(dh);                                             \
    MEGDNN_MARK_USED_VAR(dw);                                             \
    MEGDNN_MARK_USED_VAR(sgh);                                            \
    MEGDNN_MARK_USED_VAR(sgw);

// vim: syntax=cuda.doxygen
