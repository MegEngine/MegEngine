#include <cuda_runtime.h>

#include "./imma8832_128x256.cuh"
#include "./macro.cuh"
#include "./tools.cuh"

using namespace convolution;

extern "C" __global__ void __launch_bounds__(256)
        ampere_conv_bias_uint4_int4_imma8832_ldg16_128x256_relu(
                const int8_t* __restrict__ src, int8_t* __restrict__ filter,
                const float* __restrict__ bias, int8_t* __restrict__ dst, float alpha,
                float beta, uint32_t pk_src_zero_point, float dst_zero_point,
                uint32_t relu, Conv2dInt4Param param,
                Conv2dConstantOffset conv2d_constant) {
#ifdef SM80_ENABLED
    const uint32_t tid = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    __shared__ int32_t smem[12288];  // (128+256)*128/8*2
    int2 reg_acc[reg_m][reg_n];
    int4 reg_src[2][reg_nd4];
    int4 reg_flt[2][reg_md4];
    // use in other way, maybe use reg_ser/flt
    int4 reg_src_cache[2];
    int4 reg_filter_cache[4];

    uint32_t gtid = (tid >> 7);
    uint32_t tid127 = (tid & 127);
    uint32_t section = (tid127 >> 1);
    uint32_t residue = ((tid127 << 5) & 63);
    uint32_t nhw = bidx * BN + section;

    uint32_t tn, hw, toh, tow;
    int tih, tiw;
    int h_start[2];
    int h_end[2];
    int w_start[2];
    int w_end[2];
    bool g[2];
    const int8_t* __restrict__ g_src_ptr[2];
    for (int i = 0; i < 2; i++) {
        if (i != 0) {
            nhw += 64;
        }
        tn = nhw / param.div_ohow;
        hw = nhw % param.div_ohow;
        toh = hw / param.div_ow;
        tow = hw % param.div_ow;
        tih = toh * param.sh - param.ph;
        tiw = tow * param.sw - param.pw;
        g[i] = tn < param.n;
        h_start[i] = -tih;
        h_end[i] = param.ih - tih;
        w_start[i] = -tiw;
        w_end[i] = param.iw - tiw;
        // param's members have been converted to byte offset and int4 offset need to
        // div 2
        int src_offset = tn * param.ibs + tih * param.ihs +
                         ((int)(tiw * packed_channel + residue) >> 1);
        g_src_ptr[i] = src + src_offset;
    }

    const uint32_t section_section = (section >> 2);
    const uint32_t section_residue = (section & 3);
    const uint32_t section_factor = ((section & 15) >> 2);
    const uint32_t crosswise_offset =
            ((section_residue >> 1) << 4) +
            (((section_residue & 1) ^ (section_factor >> 1)) << 3);
    const uint32_t residue_offset = ((residue >> 5) ^ (section_factor & 1)) << 2;

    // next + 64 * BK / 8
    int32_t* write_src_s = smem + section_section * BK / 2 + crosswise_offset +
                           residue_offset + (gtid << 5);

    int iter = (param.icfhfw >> 6);

    uint32_t tid31 = (tid & 31);
    uint32_t warp_idx = (tid >> 5);
    uint32_t warp_strided = (warp_idx << 2);
    uint32_t htid = (tid31 >> 4);
    const uint32_t flt_strided = bidy * BM / 8 + warp_strided;
    bool guard = flt_strided * 8 < param.oc && iter > htid;
    // icfhfw * 8/2 is a stride
    int8_t* __restrict__ g_filter_ptr0 =
            filter + flt_strided * (param.icfhfw * 4) + (tid31 << 4);
    int8_t* __restrict__ g_filter_ptr1 = g_filter_ptr0 + (param.icfhfw * 4);
    int8_t* __restrict__ g_filter_ptr2 = g_filter_ptr0 + (param.icfhfw * 8);
    int8_t* __restrict__ g_filter_ptr3 = g_filter_ptr0 + (param.icfhfw * 12);

    // next + BK * 8 / (INT32/INT4)
    uint32_t q = (tid31 >> 3);
    uint32_t r = (tid31 & 7);
    int32_t* write_flt_s = smem + BN * BK / 8 + warp_strided * BK + ((q & 1) << 6) +
                           ((q >> 1) << 5) + (r << 2);
    uint32_t quad_idx = (tid31 >> 2);
    uint32_t idx_in_quad = (tid & 3);
    uint32_t quad_factor = ((tid & 15) >> 2);
    uint32_t crosswise =
            ((idx_in_quad >> 1) << 4) + (((idx_in_quad & 1) ^ (quad_factor >> 1)) << 3);
    uint32_t warp_x = (warp_idx >> 2);
    uint32_t warp_y = (warp_idx & 3);

    int32_t* read_src_s_0 = smem + (warp_x * 8 * BK) + (quad_idx * BK / 2) + crosswise +
                            ((0 ^ (quad_factor & 1)) << 2);
    int32_t* read_src_s_1 = smem + (warp_x * 8 * BK) + (quad_idx * BK / 2) + crosswise +
                            ((1 ^ (quad_factor & 1)) << 2);
    int32_t* read_flt_s_0 = smem + BN * BK / 8 + (warp_y * 8 * BK) +
                            (quad_idx * BK / 2) + crosswise +
                            ((0 ^ (quad_factor & 1)) << 2);
    int32_t* read_flt_s_1 = smem + BN * BK / 8 + (warp_y * 8 * BK) +
                            (quad_idx * BK / 2) + crosswise +
                            ((1 ^ (quad_factor & 1)) << 2);

#pragma unroll
    for (int i = 0; i < reg_m; i++) {
#pragma unroll
        for (int j = 0; j < reg_n; j++) {
            reg_acc[i][j] = make_int2(0, 0);
        }
    }

    int32_t smem_switch = 6144;
    uint32_t offset = gtid * 2;
    int src_step = conv2d_constant.c_offset[offset];
    uint32_t spatial = *(
            reinterpret_cast<const uint32_t*>(&(conv2d_constant.c_offset[offset + 1])));
    int x = (spatial & 0xff);
    int y = ((spatial >> 8) & 0xff);
    if (offset < conv2d_constant.c_offset_param.max) {
        offset += 4;
    } else {
        offset += conv2d_constant.c_offset_param.rewind;
    }

    bool guard0 = g[0] && x >= h_start[0] && x < h_end[0] && y >= w_start[0] &&
                  y < w_end[0] && iter > gtid;
    bool guard1 = g[1] && x >= h_start[1] && x < h_end[1] && y >= w_start[1] &&
                  y < w_end[1] && iter > gtid;
    g_src_ptr[0] += src_step;
    g_src_ptr[1] += src_step;

    if (guard0) {
        reg_src_cache[0] = *(reinterpret_cast<const int4*>(g_src_ptr[0]));
    } else {
        reg_src_cache[0] = make_int4(
                pk_src_zero_point, pk_src_zero_point, pk_src_zero_point,
                pk_src_zero_point);
    }
    if (guard1) {
        reg_src_cache[1] = *(reinterpret_cast<const int4*>(g_src_ptr[1]));
    } else {
        reg_src_cache[1] = make_int4(
                pk_src_zero_point, pk_src_zero_point, pk_src_zero_point,
                pk_src_zero_point);
    }

    if (guard) {
        reg_filter_cache[0] = *(reinterpret_cast<const int4*>(g_filter_ptr0));
        reg_filter_cache[1] = *(reinterpret_cast<const int4*>(g_filter_ptr1));
        reg_filter_cache[2] = *(reinterpret_cast<const int4*>(g_filter_ptr2));
        reg_filter_cache[3] = *(reinterpret_cast<const int4*>(g_filter_ptr3));
    } else {
        reg_filter_cache[0] = make_int4(0, 0, 0, 0);
        reg_filter_cache[1] = make_int4(0, 0, 0, 0);
        reg_filter_cache[2] = make_int4(0, 0, 0, 0);
        reg_filter_cache[3] = make_int4(0, 0, 0, 0);
    }

    *(reinterpret_cast<int4*>(write_src_s)) = reg_src_cache[0];
    *(reinterpret_cast<int4*>(write_src_s + 8 * BK)) = reg_src_cache[1];
    *(reinterpret_cast<int4*>(write_flt_s)) = reg_filter_cache[0];
    *(reinterpret_cast<int4*>(write_flt_s + BK)) = reg_filter_cache[1];
    *(reinterpret_cast<int4*>(write_flt_s + 2 * BK)) = reg_filter_cache[2];
    *(reinterpret_cast<int4*>(write_flt_s + 3 * BK)) = reg_filter_cache[3];

    __syncthreads();

    iter -= 2;

    src_step = conv2d_constant.c_offset[offset];
    spatial = *(
            reinterpret_cast<const uint32_t*>(&(conv2d_constant.c_offset[offset + 1])));
    x = (spatial & 0xff);
    y = ((spatial >> 8) & 0xff);
    if (offset < conv2d_constant.c_offset_param.max) {
        offset += 4;
    } else {
        offset += conv2d_constant.c_offset_param.rewind;
    }

    guard0 = g[0] && x >= h_start[0] && x < h_end[0] && y >= w_start[0] && y < w_end[0];
    guard1 = g[1] && x >= h_start[1] && x < h_end[1] && y >= w_start[1] && y < w_end[1];

    g_src_ptr[0] += src_step;
    g_src_ptr[1] += src_step;
    g_filter_ptr0 += 8 * 64;
    g_filter_ptr1 += 8 * 64;
    g_filter_ptr2 += 8 * 64;
    g_filter_ptr3 += 8 * 64;

    write_src_s += smem_switch;
    write_flt_s += smem_switch;

    for (; iter >= 2; iter -= 2) {
        if (guard0) {
            reg_src_cache[0] = *(reinterpret_cast<const int4*>(g_src_ptr[0]));
        } else {
            reg_src_cache[0] = make_int4(
                    pk_src_zero_point, pk_src_zero_point, pk_src_zero_point,
                    pk_src_zero_point);
        }
        if (guard1) {
            reg_src_cache[1] = *(reinterpret_cast<const int4*>(g_src_ptr[1]));
        } else {
            reg_src_cache[1] = make_int4(
                    pk_src_zero_point, pk_src_zero_point, pk_src_zero_point,
                    pk_src_zero_point);
        }

        if (guard) {
            reg_filter_cache[0] = *(reinterpret_cast<const int4*>(g_filter_ptr0));
            reg_filter_cache[1] = *(reinterpret_cast<const int4*>(g_filter_ptr1));
            reg_filter_cache[2] = *(reinterpret_cast<const int4*>(g_filter_ptr2));
            reg_filter_cache[3] = *(reinterpret_cast<const int4*>(g_filter_ptr3));
        } else {
            reg_filter_cache[0] = make_int4(0, 0, 0, 0);
            reg_filter_cache[1] = make_int4(0, 0, 0, 0);
            reg_filter_cache[2] = make_int4(0, 0, 0, 0);
            reg_filter_cache[3] = make_int4(0, 0, 0, 0);
        }

#pragma unroll
        for (int i = 0; i < reg_nd4; ++i) {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_src_s_0 + i * 4 * BK);  // BK*32/8
            asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                    "[%4];"
                    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                    : "r"(addr));
            reg_src[0][i] = make_int4(x, y, z, w);
        }

#pragma unroll
        for (int j = 0; j < reg_md4; ++j) {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_flt_s_0 + 4 * j * BK);
            asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                    "[%4];"
                    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                    : "r"(addr));
            reg_flt[0][j] = make_int4(x, y, z, w);
        }

        smem_switch = -smem_switch;
#pragma unroll
        for (int k_inner = 0; k_inner < BKd32; k_inner++) {
            int comp = (k_inner & 0x1);
            int load = 1 - comp;
            if (k_inner < BKd32 - 1) {
                int32_t* read_src_s = (k_inner & 1) ? read_src_s_0 : read_src_s_1;
                int32_t* read_flt_s = (k_inner & 1) ? read_flt_s_0 : read_flt_s_1;
                read_src_s += 32 * ((k_inner + 1) >> 1);
                read_flt_s += 32 * ((k_inner + 1) >> 1);

#pragma unroll
                for (int i = 0; i < reg_nd4; ++i) {
                    int x, y, z, w;
                    unsigned addr =
                            get_smem_pointer(read_src_s + i * 4 * BK);  // BK*32/8
                    asm volatile(
                            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                            "%3}, "
                            "[%4];"
                            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                            : "r"(addr));
                    reg_src[load][i] = make_int4(x, y, z, w);
                }

#pragma unroll
                for (int j = 0; j < reg_md4; ++j) {
                    int x, y, z, w;
                    unsigned addr = get_smem_pointer(read_flt_s + 4 * j * BK);
                    asm volatile(
                            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                            "%3}, "
                            "[%4];"
                            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                            : "r"(addr));
                    reg_flt[load][j] = make_int4(x, y, z, w);
                }
            }

            int* A = reinterpret_cast<int*>(&reg_flt[comp][0]);
            int* B = reinterpret_cast<int*>(&reg_src[comp][0]);
#pragma unroll
            for (int x = 0; x < reg_n; x++) {
#pragma unroll
                for (int y = 0; y < reg_m; y++) {
                    int* D = reinterpret_cast<int*>(&reg_acc[y][x]);
                    int* C = reinterpret_cast<int*>(&reg_acc[y][x]);
                    asm volatile(
                            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4."
                            "s32 "
                            "{%0,%1}, {%2}, {%3}, "
                            "{%4,%5};\n"
                            : "=r"(D[0]), "=r"(D[1])
                            : "r"(B[y]), "r"(A[x]), "r"(C[0]), "r"(C[1]));
                }
            }
        }

        *(reinterpret_cast<int4*>(write_src_s)) = reg_src_cache[0];
        *(reinterpret_cast<int4*>(write_src_s + 8 * BK)) = reg_src_cache[1];
        *(reinterpret_cast<int4*>(write_flt_s)) = reg_filter_cache[0];
        *(reinterpret_cast<int4*>(write_flt_s + BK)) = reg_filter_cache[1];
        *(reinterpret_cast<int4*>(write_flt_s + 2 * BK)) = reg_filter_cache[2];
        *(reinterpret_cast<int4*>(write_flt_s + 3 * BK)) = reg_filter_cache[3];

        write_src_s += smem_switch;
        write_flt_s += smem_switch;
        read_src_s_0 -= smem_switch;
        read_src_s_1 -= smem_switch;
        read_flt_s_0 -= smem_switch;
        read_flt_s_1 -= smem_switch;
        __syncthreads();

        int src_step = conv2d_constant.c_offset[offset];
        uint32_t spatial = *(reinterpret_cast<const uint32_t*>(
                &(conv2d_constant.c_offset[offset + 1])));
        x = (spatial & 0xff);
        y = ((spatial >> 8) & 0xff);
        if (offset < conv2d_constant.c_offset_param.max) {
            offset += 4;
        } else {
            offset += conv2d_constant.c_offset_param.rewind;
        }

        guard0 = g[0] && x >= h_start[0] && x < h_end[0] && y >= w_start[0] &&
                 y < w_end[0];
        guard1 = g[1] && x >= h_start[1] && x < h_end[1] && y >= w_start[1] &&
                 y < w_end[1];

        g_src_ptr[0] += src_step;
        g_src_ptr[1] += src_step;

        g_filter_ptr0 += 8 * 64;
        g_filter_ptr1 += 8 * 64;
        g_filter_ptr2 += 8 * 64;
        g_filter_ptr3 += 8 * 64;
    }

    if (iter > 0) {
        if (guard0 && iter > gtid) {
            reg_src_cache[0] = *(reinterpret_cast<const int4*>(g_src_ptr[0]));
        } else {
            reg_src_cache[0] = make_int4(
                    pk_src_zero_point, pk_src_zero_point, pk_src_zero_point,
                    pk_src_zero_point);
        }
        if (guard1 && iter > gtid) {
            reg_src_cache[1] = *(reinterpret_cast<const int4*>(g_src_ptr[1]));
        } else {
            reg_src_cache[1] = make_int4(
                    pk_src_zero_point, pk_src_zero_point, pk_src_zero_point,
                    pk_src_zero_point);
        }

        if (guard && iter > htid) {
            reg_filter_cache[0] = *(reinterpret_cast<const int4*>(g_filter_ptr0));
            reg_filter_cache[1] = *(reinterpret_cast<const int4*>(g_filter_ptr1));
            reg_filter_cache[2] = *(reinterpret_cast<const int4*>(g_filter_ptr2));
            reg_filter_cache[3] = *(reinterpret_cast<const int4*>(g_filter_ptr3));
        } else {
            reg_filter_cache[0] = make_int4(0, 0, 0, 0);
            reg_filter_cache[1] = make_int4(0, 0, 0, 0);
            reg_filter_cache[2] = make_int4(0, 0, 0, 0);
            reg_filter_cache[3] = make_int4(0, 0, 0, 0);
        }

#pragma unroll
        for (int i = 0; i < reg_nd4; ++i) {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_src_s_0 + i * 4 * BK);  // BK*32/8
            asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                    "[%4];"
                    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                    : "r"(addr));
            reg_src[0][i] = make_int4(x, y, z, w);
        }

#pragma unroll
        for (int j = 0; j < reg_md4; ++j) {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_flt_s_0 + 4 * j * BK);
            asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                    "[%4];"
                    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                    : "r"(addr));
            reg_flt[0][j] = make_int4(x, y, z, w);
        }

        smem_switch = -smem_switch;
#pragma unroll
        for (int k_inner = 0; k_inner < BKd32; k_inner++) {
            int comp = (k_inner & 0x1);
            int load = 1 - comp;
            if (k_inner < BKd32 - 1) {
                int32_t* read_src_s = (k_inner & 1) ? read_src_s_0 : read_src_s_1;
                int32_t* read_flt_s = (k_inner & 1) ? read_flt_s_0 : read_flt_s_1;
                read_src_s += 32 * ((k_inner + 1) >> 1);
                read_flt_s += 32 * ((k_inner + 1) >> 1);

#pragma unroll
                for (int i = 0; i < reg_nd4; ++i) {
                    int x, y, z, w;
                    unsigned addr =
                            get_smem_pointer(read_src_s + i * 4 * BK);  // BK*32/8
                    asm volatile(
                            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                            "%3}, "
                            "[%4];"
                            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                            : "r"(addr));
                    reg_src[load][i] = make_int4(x, y, z, w);
                }
#pragma unroll
                for (int j = 0; j < reg_md4; ++j) {
                    int x, y, z, w;
                    unsigned addr = get_smem_pointer(read_flt_s + 4 * j * BK);
                    asm volatile(
                            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                            "%3}, "
                            "[%4];"
                            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                            : "r"(addr));
                    reg_flt[load][j] = make_int4(x, y, z, w);
                }
            }

            int* A = reinterpret_cast<int*>(&reg_flt[comp][0]);
            int* B = reinterpret_cast<int*>(&reg_src[comp][0]);
#pragma unroll
            for (int x = 0; x < reg_n; x++) {
#pragma unroll
                for (int y = 0; y < reg_m; y++) {
                    int* D = reinterpret_cast<int*>(&reg_acc[y][x]);
                    int* C = reinterpret_cast<int*>(&reg_acc[y][x]);
                    asm volatile(
                            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4."
                            "s32 "
                            "{%0,%1}, {%2}, {%3}, "
                            "{%4,%5};\n"
                            : "=r"(D[0]), "=r"(D[1])
                            : "r"(B[y]), "r"(A[x]), "r"(C[0]), "r"(C[1]));
                }
            }
        }
        *(reinterpret_cast<int4*>(write_src_s)) = reg_src_cache[0];
        *(reinterpret_cast<int4*>(write_src_s + 8 * BK)) = reg_src_cache[1];
        *(reinterpret_cast<int4*>(write_flt_s)) = reg_filter_cache[0];
        *(reinterpret_cast<int4*>(write_flt_s + BK)) = reg_filter_cache[1];
        *(reinterpret_cast<int4*>(write_flt_s + 2 * BK)) = reg_filter_cache[2];
        *(reinterpret_cast<int4*>(write_flt_s + 3 * BK)) = reg_filter_cache[3];

        read_src_s_0 -= smem_switch;
        read_src_s_1 -= smem_switch;
        read_flt_s_0 -= smem_switch;
        read_flt_s_1 -= smem_switch;

        __syncthreads();
    }

    size_t oc = bidy * BM + (warp_y << 6) + 16 * idx_in_quad;
    const float* bias_ptr = bias + oc;

    int4 load_bias0 = make_int4(0, 0, 0, 0);
    int4 load_bias1 = make_int4(0, 0, 0, 0);
    int4 load_bias2 = make_int4(0, 0, 0, 0);
    int4 load_bias3 = make_int4(0, 0, 0, 0);
    if (oc < param.oc) {
        load_bias0 = *(reinterpret_cast<const int4*>(bias_ptr));
        load_bias1 = *(reinterpret_cast<const int4*>(bias_ptr + 4));
        load_bias2 = *(reinterpret_cast<const int4*>(bias_ptr + 8));
        load_bias3 = *(reinterpret_cast<const int4*>(bias_ptr + 12));
    }

    guard = iter < 0;
#pragma unroll
    for (int i = 0; i < reg_nd4; ++i) {
        int x, y, z, w;
        unsigned addr = get_smem_pointer(read_src_s_0 + i * 4 * BK);  // BK*32/8
        asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                "%3}, "
                "[%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "r"(addr));
        reg_src[0][i] = make_int4(x, y, z, w);
    }
#pragma unroll
    for (int j = 0; j < reg_md4; ++j) {
        int x, y, z, w;
        unsigned addr = get_smem_pointer(read_flt_s_0 + 4 * j * BK);
        asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                "%3}, "
                "[%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "r"(addr));
        reg_flt[0][j] = make_int4(x, y, z, w);
    }

// compute
#pragma unroll
    for (int k_inner = 0; k_inner < BKd32; k_inner++) {
        int comp = (k_inner & 0x1);
        int load = 1 - comp;
        if (k_inner < BKd32 - 1 && !(k_inner == 1 && guard)) {
            int32_t* read_src_s = (k_inner & 1) ? read_src_s_0 : read_src_s_1;
            int32_t* read_flt_s = (k_inner & 1) ? read_flt_s_0 : read_flt_s_1;
            read_src_s += 32 * ((k_inner + 1) >> 1);
            read_flt_s += 32 * ((k_inner + 1) >> 1);

#pragma unroll
            for (int i = 0; i < reg_nd4; ++i) {
                int x, y, z, w;
                unsigned addr = get_smem_pointer(read_src_s + i * 4 * BK);  // BK*32/8
                asm volatile(
                        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                        "%3}, "
                        "[%4];"
                        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                        : "r"(addr));
                reg_src[load][i] = make_int4(x, y, z, w);
            }

#pragma unroll
            for (int j = 0; j < reg_md4; ++j) {
                int x, y, z, w;
                unsigned addr = get_smem_pointer(read_flt_s + 4 * j * BK);
                asm volatile(
                        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                        "%3}, "
                        "[%4];"
                        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                        : "r"(addr));
                reg_flt[load][j] = make_int4(x, y, z, w);
            }
        }

        int* A = reinterpret_cast<int*>(&reg_flt[comp][0]);
        int* B = reinterpret_cast<int*>(&reg_src[comp][0]);
#pragma unroll
        for (int x = 0; x < reg_n; x++) {
#pragma unroll
            for (int y = 0; y < reg_m; y++) {
                int* D = reinterpret_cast<int*>(&reg_acc[y][x]);
                int* C = reinterpret_cast<int*>(&reg_acc[y][x]);
                asm volatile(
                        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4."
                        "s32 "
                        "{%0,%1}, {%2}, {%3}, "
                        "{%4,%5};\n"
                        : "=r"(D[0]), "=r"(D[1])
                        : "r"(B[y]), "r"(A[x]), "r"(C[0]), "r"(C[1]));
            }
        }
        if (k_inner == 1 && guard) {
            break;
        }
    }

    __syncthreads();

    /// output
    tid31 = (tid & 31);
    idx_in_quad = tid & 3;

    section = tid31 >> 2;
    size_t nhw_post0 = bidx * BN + warp_x * 64 + section;
    size_t nhw_post1 = nhw_post0 + 8;
    size_t nhw_post2 = nhw_post0 + 16;
    size_t nhw_post3 = nhw_post0 + 24;

    size_t stg_oc = bidy * BM + (warp_y << 6);

    if (oc < param.oc) {
        mul_v4(load_bias0, load_bias0, beta);
        mul_v4(load_bias1, load_bias1, beta);
        mul_v4(load_bias2, load_bias2, beta);
        mul_v4(load_bias3, load_bias3, beta);
    }

    bool stg_guard[8];
    int8_t* __restrict__ g_dst_ptr =
            dst + ((bidy * (BM >> 6) + warp_y) * param.ocs + (idx_in_quad << 3));
    int8_t** stg_ptr = ((int8_t**)&reg_filter_cache);

#pragma unroll
    for (int y = 0; y < reg_m; y += 4) {
        FMA_4x8(reg_acc, y, 0, alpha, load_bias0, load_bias1, load_bias2, load_bias3);
        PACK_F2I_WITH_RELU_4x8(reg_acc, y, 0, relu, dst_zero_point);
        STG_4x1(stg_ptr, reg_acc, y, 0);

        nhw_post0 += 32;
        nhw_post1 += 32;
        nhw_post2 += 32;
        nhw_post3 += 32;
    }
#endif
}

namespace megdnn {
namespace cuda {
namespace ptx {
void run_ampere_conv_bias_uint4_int4_imma8832_ldg16_128x256_relu(
        const dim3 grid, const dim3 block, cudaStream_t stream, void** params) {
#ifdef SM80_SUPPORTED
    ampere_conv_bias_uint4_int4_imma8832_ldg16_128x256_relu<<<grid, block, 0, stream>>>(
            *((int8_t**)params[0]), *((int8_t**)params[1]), *((float**)params[2]),
            *((int8_t**)params[3]), *((float*)params[4]), *((float*)params[5]),
            *((uint32_t*)params[6]), *((float*)params[7]), *((uint32_t*)params[8]),
            *((Conv2dInt4Param*)params[9]), *((Conv2dConstantOffset*)params[10]));
#endif
}
}  // namespace ptx
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
