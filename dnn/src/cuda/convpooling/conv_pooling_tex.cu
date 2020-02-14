/**
 * \file dnn/src/cuda/convpooling/conv_pooling_tex.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./conv_pooling.cuh"
//#include "./kernel_impl/kernel_impl.h"
#include "./conv_pooling_utils.cuh"

namespace megdnn {
namespace cuda {
namespace conv_pool {

#define NR_PXL_PER_THREAD   4
#define NR_THREAD_PER_BLOCK 192
#define MAX_SHARED_MEM_SIZE 32768 //32 * 1024
#define MAX_TEX_OBJ_SIZE    134217728 //2^27
#define HEIGHT_EQUALS_WITH_WEIGHT


 __host__  void create_cuda_tex(float *input, cudaTextureObject_t& tex,
        size_t N, size_t IC, size_t IH, size_t IW) {

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = (void*)input;
    res_desc.res.linear.sizeInBytes = N * IC * IH * IW * sizeof(float);
    res_desc.res.linear.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.readMode = cudaReadModeElementType;
    CUDA_CHKERR(cudaCreateTextureObject(
        &tex, &res_desc, &tex_desc, NULL));

}

void start_gpu_xcorr_pool_with_texture_obj(
        cudaStream_t stream,
        float *input,
        const float *kernel,
        float *output,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t /*PH*/, size_t /*PW*/,
        size_t /*SH*/, size_t /*SW*/,
        size_t pool_shape_h,
        size_t pool_shape_w,
        PoolModeCu poolMode,
        ConvModeCu convMode,
        NonlineModeCu nonlineMode,
        const float *bias) {

    int nr_batch = N, nr_oc = OC,
        output_area2d = OH * OW,
        kern_h = FH, kern_w = FW,
        nr_thread_per_block = std::min(NR_THREAD_PER_BLOCK,
                align_to_warp(output_area2d)),
        oplane_nr_split = std::max(1,
                output_area2d / (nr_thread_per_block * NR_PXL_PER_THREAD)),
        share_size = kern_h * kern_w * IC * sizeof(float);
    megdnn_assert(share_size < MAX_SHARED_MEM_SIZE, "kernel too large: "
            "total %d bytes per output channel allowed, got %d",
            MAX_SHARED_MEM_SIZE, share_size);

    void (*f) (float *input,
        const float *filter,
        float *output,
        const float *output_bias,
        cudaTextureObject_t m_tex,
        int IC, int IH, int IW,
        int OH, int OW) = NULL;

#define DISPATCH_POOLMODE(nonlin, kh, kw, ph, pw, convMode) \
    do { \
        switch (poolMode) { \
        case AVERAGE: \
            f = kern_xcorr_smallkern_pool<kh, kw, ph, pw, \
                    nonlin, MeanPooler, convMode>; \
        break; \
        case MAX: \
            f = kern_xcorr_smallkern_pool<kh, kw, ph, pw, \
                    nonlin, MaxPooler, convMode>; \
        break; \
        } \
    } while(0)

#define DISPATCH_CONVMODE(nonlin, kh, kw, ph, pw) \
    do { \
        switch (convMode) { \
            case CONVOLUTION: DISPATCH_POOLMODE \
                (nonlin, kh, kw, ph, pw, IdxGetterConvolution); break; \
            case CROSS_CORRELATION: DISPATCH_POOLMODE\
                (nonlin, kh, kw, ph, pw, IdxGetterCorrRel); break; \
        } \
    } while(0)

#ifdef HEIGHT_EQUALS_WITH_WEIGHT

#define DISPATCH_POOLSHAPE(nonlin, kh, kw) \
    do { \
        switch (pool_shape_h) { \
            case 1: DISPATCH_CONVMODE(nonlin, kh, kw, 1, 1); break; \
            case 2: DISPATCH_CONVMODE(nonlin, kh, kw, 2, 2); break; \
            case 3: DISPATCH_CONVMODE(nonlin, kh, kw, 3, 3); break; \
            case 4: DISPATCH_CONVMODE(nonlin, kh, kw, 4, 4); break; \
        } \
    } while(0)

#define DISPATCH_KERN_H(nonlin) \
    do { \
        switch(kern_h) { \
        case 1: DISPATCH_POOLSHAPE(nonlin, 1, 1); break;\
        case 2: DISPATCH_POOLSHAPE(nonlin, 2, 2); break;\
        case 3: DISPATCH_POOLSHAPE(nonlin, 3, 3); break;\
        case 4: DISPATCH_POOLSHAPE(nonlin, 4, 4); break;\
        case 5: DISPATCH_POOLSHAPE(nonlin, 5, 5); break;\
        case 6: DISPATCH_POOLSHAPE(nonlin, 6, 6); break;\
        case 7: DISPATCH_POOLSHAPE(nonlin, 7, 7); break;\
        } \
    } while(0)

#else //HEIGHT_EQUALS_WITH_WEIGHT

#define DISPATCH_POOLSHAPE_W(nonlin, kh, kw, ph) \
    do { \
        switch (pool_shape_w) { \
            case 1: DISPATCH_CONVMODE(nonlin, kh, kw, ph, 1); break; \
            case 2: DISPATCH_CONVMODE(nonlin, kh, kw, ph, 2); break; \
            case 3: DISPATCH_CONVMODE(nonlin, kh, kw, ph, 3); break; \
            case 4: DISPATCH_CONVMODE(nonlin, kh, kw, ph, 4); break; \
        } \
    } while(0)

#define DISPATCH_POOLSHAPE_H(nonlin, kern_h, kern_w) \
    do { \
        switch (pool_shape_h) { \
            case 1: DISPATCH_POOLSHAPE_W(nonlin, kern_h, kern_w, 1); break; \
            case 2: DISPATCH_POOLSHAPE_W(nonlin, kern_h, kern_w, 2); break; \
            case 3: DISPATCH_POOLSHAPE_W(nonlin, kern_h, kern_w, 3); break; \
            case 4: DISPATCH_POOLSHAPE_W(nonlin, kern_h, kern_w, 4); break; \
        } \
    } while(0)

#define DISPATCH_KERN_W(nonlin, kern_h) \
    do { \
        switch(kern_w) { \
        case 1: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 1); break;\
        case 2: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 2); break;\
        case 3: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 3); break;\
        case 4: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 4); break;\
        case 5: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 5); break;\
        case 6: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 6); break;\
        case 7: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 7); break;\
        case 8: DISPATCH_POOLSHAPE_H(nonlin, kern_h, 8); break;\
        } \
    } while(0)

#define DISPATCH_KERN_H(nonlin) \
    do { \
        switch(kern_h) { \
        case 1: DISPATCH_KERN_W(nonlin, 1); break;\
        case 2: DISPATCH_KERN_W(nonlin, 2); break;\
        case 3: DISPATCH_KERN_W(nonlin, 3); break;\
        case 4: DISPATCH_KERN_W(nonlin, 4); break;\
        case 5: DISPATCH_KERN_W(nonlin, 5); break;\
        case 6: DISPATCH_KERN_W(nonlin, 6); break;\
        case 7: DISPATCH_KERN_W(nonlin, 7); break;\
        case 8: DISPATCH_KERN_W(nonlin, 8); break;\
        } \
    } while(0)

#endif //HEIGHT_EQUALS_WITH_WEIGHT
    switch(nonlineMode) {
        case IDENTITY:
            DISPATCH_KERN_H(Identity);
        break;
        case RELU:
            DISPATCH_KERN_H(Relu);
        break;

        case SIGMOID:
            DISPATCH_KERN_H(Sigmoid);
        break;
    }

    megdnn_assert(f, "Start_gpu_xcorr_pool: unsupported conv-pooling configuration. \
        pool_shape_h %zu, pool_shape_w %zu, kern_h %d, kern_w %d\n",
        pool_shape_h, pool_shape_w, kern_h, kern_w);

    cudaTextureObject_t m_tex = 0;
    size_t input_size = N * IC * IH * IW;

    // Case 1: Size of input data is less than
    // the limit of cudaTextureObject_t.
    if(input_size < MAX_TEX_OBJ_SIZE) {
        dim3 grid_dim(nr_batch, nr_oc, oplane_nr_split),
             block_dim(nr_thread_per_block);
        create_cuda_tex(input, m_tex, N, IC,  IH,  IW);
        f<<<grid_dim, block_dim, share_size, stream>>>(
                input, kernel, output, bias, m_tex,
                IC,  IH,  IW, OH,  OW);
    }
    // Case 2: Size of input data reached
    // the limit of cudaTextureObject_t (2^27 Bytes).
    else {
        size_t  input_stride = IC * IH * IW,
                output_stride = OC * OH * OW;
        int batch_size = MAX_TEX_OBJ_SIZE / input_stride;
        float *input_base = input;
        float *output_base = output;
        for(; nr_batch > 0; nr_batch -= batch_size) {
            int cur_batch = nr_batch < batch_size ? nr_batch : batch_size;
            dim3 grid_dim(cur_batch, nr_oc, oplane_nr_split),
            block_dim(nr_thread_per_block);
            create_cuda_tex(input_base, m_tex, N, IC,  IH,  IW);
            f<<<grid_dim, block_dim, share_size, stream>>>(
                input_base, kernel, output_base, bias, m_tex,
                IC, IH,  IW, OH,  OW);

            input_base += batch_size * input_stride;
            output_base += batch_size * output_stride;
        }
    }
    CUDA_CHKERR(cudaPeekAtLastError());
    CUDA_CHK_KERN_ERR;

    CUDA_CHKERR(cudaDestroyTextureObject(m_tex));
    m_tex = 0;
    //texinput.destory();
}

} // namespace conv_pool
} // namespace cuda
} // namespace megdnn
#undef CUDA_CHKERR
#undef CUDA_CHK_KERN_ERR
#undef NR_PXL_PER_THREAD
#undef NR_THREAD_PER_BLOCK
#undef MAX_SHARED_MEM_SIZE
#undef MAX_TEX_OBJ_SIZE
// vim: syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
