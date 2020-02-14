/**
 * \file dnn/src/cuda/elemwise_multi_type/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/elemwise_multi_type/kern.cuh"
#include "src/cuda/elemwise_multi_type/kern_ops.cuh"

using namespace megdnn;
using namespace cuda;
using namespace elemwise_multi_type;
using namespace elemwise_intl;
using namespace kern_ops;

void elemwise_multi_type::fma3_int16x32x32x32_1c1(
        const ElemwiseOpParamN<3>& param, dt_int32* dst, cudaStream_t stream) {
    typedef Fma3Int16x32x32x32Bcast101Op Caller;
    void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
    int grid_size, block_size;
    get_launch_spec(reinterpret_cast<const void*>(fptr), param.size, &grid_size,
                    &block_size);

    Caller caller;
    caller.a.host_init(param[0], grid_size, block_size);
    caller.b.host_init(param[1], grid_size, block_size);
    caller.c.host_init(param[2], grid_size, block_size);
    caller.dst = dst;

    (*fptr)<<<grid_size, block_size, 0, stream>>>(caller, param.size);
    after_kernel_launch();
}

template <typename stype, typename dst_type>
void elemwise_multi_type::round_shr_saturate_iXxi8xiX_scalar(
        const ElemwiseOpParamN<2>& param, dst_type* dst, cudaStream_t stream) {
    typedef RoundShrSaturateIXxBcastScalarOp<stype, dst_type> Caller;
    void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
    int grid_size, block_size;
    get_launch_spec(reinterpret_cast<const void*>(fptr), param.size, &grid_size,
                    &block_size);

    Caller caller;
    caller.a.host_init(param[0], grid_size, block_size);
    caller.b.host_init(param[1], grid_size, block_size);
    caller.dst = dst;

    (*fptr)<<<grid_size, block_size, 0, stream>>>(caller, param.size);
    after_kernel_launch();
}

#define INST(stype)                                                 \
    template void                                                   \
    elemwise_multi_type::round_shr_saturate_iXxi8xiX_scalar<stype>( \
            const ElemwiseOpParamN<2>& param, dt_int8*, cudaStream_t)
INST(int32_t);
INST(int16_t);
INST(int8_t);
#undef INST

#define INST(stype)                                                 \
    template void                                                   \
    elemwise_multi_type::round_shr_saturate_iXxi8xiX_scalar<stype>( \
            const ElemwiseOpParamN<2>& param, dt_int16*, cudaStream_t)
INST(int32_t);
INST(int16_t);
#undef INST

template <typename stype>
void elemwise_multi_type::fuse_add_rmulh_round_shr_saturate_bcast_1c11(
        const ElemwiseOpParamN<6>& param, dt_int8* dst, cudaStream_t stream) {
    typedef FuseAddRmulhRoundingShrBcastScalarOp<stype> Caller;
    void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
    int grid_size, block_size;
    get_launch_spec(reinterpret_cast<const void*>(fptr), param.size, &grid_size,
                    &block_size);

    Caller caller;
    caller.x.host_init(param[0], grid_size, block_size);
    caller.b.host_init(param[1], grid_size, block_size);
    caller.M.host_init(param[2], grid_size, block_size);
    caller.k.host_init(param[3], grid_size, block_size);
    caller.minv.host_init(param[4], grid_size, block_size);
    caller.maxv.host_init(param[5], grid_size, block_size);
    caller.dst = dst;

    (*fptr)<<<grid_size, block_size, 0, stream>>>(caller, param.size);
    after_kernel_launch();
}

#define INST(stype)                                                           \
    template void                                                             \
    elemwise_multi_type::fuse_add_rmulh_round_shr_saturate_bcast_1c11<stype>( \
            const ElemwiseOpParamN<6>& param, dt_int8*, cudaStream_t)
INST(int32_t);
INST(int16_t);
#undef INST

// vim: ft=cuda syntax=cuda.doxygen
