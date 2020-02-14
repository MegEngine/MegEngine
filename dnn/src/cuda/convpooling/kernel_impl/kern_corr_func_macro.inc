/*
 * Format the definition of cuda kernel function "kern_xcorr_smallkern_pool" as
 * a macro in order to generate a batch of definition-files. 
 * The original version of function "kern_xcorr_smallkern_pool" is in the file
 * "src/cuda/convpooling/conv_pooling_tex.cu.bak"
 */

#ifdef _MSC_VER
#define _Pragma __pragma
#endif

#define KERN_CORR_DEFINE(Nonlin, kern_h, kern_w, pool_shape_h, pool_shape_w, \
                        IdxGetter, Pooler) template<> \
        __global__ void kern_xcorr_smallkern_pool \
            <kern_h, kern_w, pool_shape_h, pool_shape_w, Nonlin, Pooler, IdxGetter>( \
                float *input, \
                const float *filter, \
                float *output, \
                const float *output_bias, \
                cudaTextureObject_t m_tex, \
                int IC, int IH, int IW, \
                int OH, int OW) { \
    const int\
        batch = blockIdx.x,\
        out_chl = blockIdx.y,\
        out_area2d = OH * OW,\
        out_pxl_start = (long long)blockIdx.z * out_area2d / gridDim.z,\
        out_pxl_end = (long long)(blockIdx.z + 1) * out_area2d / gridDim.z,\
        kern_volume = IC * (kern_h * kern_w),\
        thread_id = threadIdx.x,\
        nr_thread = blockDim.x,\
        pool_area = pool_shape_h * pool_shape_w;\
    const float bias = output_bias ? output_bias[out_chl] : 0; \
    const float* kernel_global = filter + out_chl * kern_volume;\
    extern __shared__ float kern[];\
    \
\
    for (int i = thread_id; i < kern_volume; i += nr_thread)\
        kern[i] = kernel_global[i];\
    __syncthreads();\
\
    float *output_ptr = output + (batch * gridDim.y + out_chl) \
                        * out_area2d; \
\
    Tex1DReader tex_reader;\
    for (int cur_out_pxl = out_pxl_start + thread_id;\
            cur_out_pxl < out_pxl_end;\
            cur_out_pxl += nr_thread) {\
        int ir_base = cur_out_pxl / OW * pool_shape_h,\
            ic_base = cur_out_pxl % OW * pool_shape_w;\
        tex_reader.set_pos(m_tex, IC, IH, IW, batch, 0, ir_base, ic_base);\
        float conv_sum[pool_area];\
\
_Pragma("unroll")\
        for (int i = 0; i < pool_area; i ++)\
            conv_sum[i] = bias;\
\
        const float *kern_ptr = kern;\
        for (int ichl = 0; ichl < IC; ichl ++) {\
            tex_reader.reset_row();\
_Pragma("unroll")\
            for (int ir = 0; ir < kern_h + pool_shape_h - 1; ir ++) {\
_Pragma("unroll")\
                for (int ic = 0; ic < kern_w + pool_shape_w - 1; ic ++) {\
                    float cur_input = tex_reader.get(ir, ic);\
_Pragma("unroll")\
                    for (int pr = 0; pr < pool_shape_h; pr ++) {\
_Pragma("unroll")\
                        for (int pc = 0; pc < pool_shape_w; pc ++) {       \
                            int kr = IdxGetter::apply(kern_h, ir, pr);\
                            int kc = IdxGetter::apply(kern_w, ic, pc);\
\
                            if (kr >= 0 && kr < kern_h &&\
                                    kc >= 0 && kc < kern_w)\
                                conv_sum[pr * pool_shape_w + pc] += \
                                    cur_input * kern_ptr[kr * kern_w + kc];\
\
                        } \
                    }\
                }\
                tex_reader.next_row();\
            }\
            kern_ptr += kern_h * kern_w;\
            tex_reader.next_channel();\
        }\
        \
_Pragma("unroll")\
        for (int i = 0; i < pool_area; i ++) {\
            conv_sum[i] = Nonlin::apply(conv_sum[i]);\
        }\
        output_ptr[cur_out_pxl] = Pooler::apply<pool_shape_h, pool_shape_w>(conv_sum);\
    } \
}
