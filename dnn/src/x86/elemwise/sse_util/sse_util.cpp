/**
 * \file dnn/src/x86/elemwise/sse_util/sse_util.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/x86/elemwise/sse_util/sse_util.h"
#include "src/x86/elemwise/sse_util/sse_mathfun.h"

#include <cmath>

namespace megdnn {
namespace x86 {
namespace detail {

void sse_element_add_by_channels(
    size_t batch_size, size_t channel_size, size_t channel_stride,
    float *src1_ptr, float *src2_ptr, float *dst_ptr) {

    size_t cur_pos = 0, src2_pos = 0, channel_offset = 0;
    __m128 src11, src12, src2;

    for(size_t batch = 0; batch < batch_size; ++batch) {
        src2_pos = 0;
        for(size_t chan = 0; chan < channel_size; ++chan, ++src2_pos) {
            src2 = _mm_set1_ps(src2_ptr[src2_pos]);
            channel_offset += channel_stride;
            for(; cur_pos + 7 < channel_offset;
                  cur_pos += 8, src1_ptr += 8, dst_ptr += 8) {
                src11 = _mm_loadu_ps(src1_ptr);
                src12 = _mm_loadu_ps(src1_ptr + 4);

                src11 = _mm_add_ps(src11, src2);
                src12 = _mm_add_ps(src12, src2);

                _mm_storeu_ps(dst_ptr, src11);
                _mm_storeu_ps(dst_ptr + 4, src12);

            }
            for(; cur_pos + 3 < channel_offset;
                  cur_pos += 4, src1_ptr += 4, dst_ptr += 4) {
                src11 = _mm_loadu_ps(src1_ptr);

                src11 = _mm_add_ps(src11, src2);

                _mm_storeu_ps(dst_ptr, src11);
            }

            float bias = src2_ptr[src2_pos];
            for(; cur_pos < channel_offset; ++cur_pos, ++dst_ptr, ++src1_ptr) {
                *dst_ptr = *src1_ptr + bias;
            }
        }
    }
}

void sse_element_set(float *dst_ptr, size_t dst_size, const float val) {
    size_t i = 0;
    __m128 vec = _mm_set1_ps(val);

    for(; i + 3 < dst_size; i += 4) {
        _mm_storeu_ps(dst_ptr + i, vec);
    }
    for(; i < dst_size; ++i) {
        dst_ptr[i] = val;
    }
}

void sse_element_add_by_channels(const TensorND &src1_tensor,
    const TensorND &src2_tensor, const TensorND &dst_tensor) {
    size_t batch_size = src1_tensor.layout.shape[0];
    size_t channel_size = src1_tensor.layout.shape[1];
    size_t channel_stride = src1_tensor.layout.stride[1];

    float* dst_ptr = dst_tensor.ptr<float>();
    float* src1_ptr = src1_tensor.ptr<float>();
    float* src2_ptr = src2_tensor.ptr<float>();

    sse_element_add_by_channels(
        batch_size, channel_size, channel_stride,
        src1_ptr, src2_ptr, dst_ptr);
}

void sse_element_add_single_val(const size_t tsize,
    float *src_ptr, float *dst_ptr, const float bias) {
    size_t i = 0;
    __m128 val1, val2, vbias = _mm_set1_ps(bias);

    for(; i + 7 < tsize; i += 8, src_ptr += 8, dst_ptr += 8) {
        val1 = _mm_loadu_ps(src_ptr);
        val2 = _mm_loadu_ps(src_ptr + 4);

        val1 = _mm_add_ps(val1, vbias);
        val2 = _mm_add_ps(val2, vbias);

        _mm_storeu_ps(dst_ptr, val1);
        _mm_storeu_ps(dst_ptr + 4, val2);
    }

    for(; i < tsize; ++i, ++src_ptr, ++dst_ptr) {
        *dst_ptr = *src_ptr + bias;
    }
}

void sse_element_add(size_t tsize, float *src_ptr,
    float *src1_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m128 val1, val2, val3, val4;

    if(tsize > 7) {
        for(; cur_pos + 7 < tsize;
            cur_pos += 8, src_ptr += 8, src1_ptr += 8, dst_ptr += 8) {
            val1 = _mm_loadu_ps(src_ptr);
            val2 = _mm_loadu_ps(src_ptr + 4);
            val3 = _mm_loadu_ps(src1_ptr);
            val4 = _mm_loadu_ps(src1_ptr + 4);

            val1 = _mm_add_ps(val1, val3);
            val2 = _mm_add_ps(val2, val4);

            _mm_storeu_ps(dst_ptr, val1);
            _mm_storeu_ps(dst_ptr + 4, val2);
        }
    }
    for(; cur_pos < tsize; ++cur_pos,
            ++src_ptr, ++src1_ptr, ++dst_ptr) {
        *dst_ptr = *src_ptr + *src1_ptr;
    }
}

void sse_element_add(const TensorND &src1_tensor,
    const TensorND &src2_tensor, const TensorND &dst_tensor) {
    size_t tsize = src1_tensor.layout.total_nr_elems();

    float* dst_ptr = dst_tensor.ptr<float>();
    float* src1_ptr = src1_tensor.ptr<float>();
    float* src2_ptr = src2_tensor.ptr<float>();

    sse_element_add( tsize,
        src1_ptr, src2_ptr, dst_ptr);
}

void sse_element_bias_relu_by_channels(const TensorND &dst_tensor, const TensorND &bias_tensor) {
    size_t batch_size = dst_tensor.layout.shape[0];
    size_t channel_size = dst_tensor.layout.shape[1];
    size_t channel_stride = dst_tensor.layout.stride[1];

    float* dst_ptr = dst_tensor.ptr<float>();
    float* bias_ptr = bias_tensor.ptr<float>();
    size_t dst_pos = 0, bias_pos = 0, channel_offset = 0;
    __m128 bias, dst1, dst2, zero_val;
    zero_val = _mm_setzero_ps();
    float tmpf;
    for(size_t batch = 0; batch < batch_size; ++ batch) {
        bias_pos = 0;
        for(size_t chan = 0; chan < channel_size; ++chan, ++bias_pos) {
            bias = _mm_set1_ps(bias_ptr[bias_pos]);
            channel_offset += channel_stride;
            if(channel_stride > 7) {
                for(; dst_pos + 7 < channel_offset;
                      dst_pos += 8, dst_ptr += 8) {
                    dst1 = _mm_loadu_ps(dst_ptr);
                    dst2 = _mm_loadu_ps(dst_ptr + 4);

                    dst1 = _mm_add_ps(dst1, bias);
                    dst2 = _mm_add_ps(dst2, bias);

                    dst1 = _mm_max_ps(dst1, zero_val);
                    dst2 = _mm_max_ps(dst2, zero_val);

                    _mm_storeu_ps(dst_ptr, dst1);
                    _mm_storeu_ps(dst_ptr + 4, dst2);
                }
            }
            for(; dst_pos < channel_offset; ++dst_pos, ++dst_ptr) {
                tmpf = *dst_ptr + bias_ptr[bias_pos];
                if(tmpf > 0) {
                    *dst_ptr  = tmpf;
                } else {
                    *dst_ptr = 0;
                }
            }
        }
    }
}

void sse_element_bias_sigmoid_by_channels(const TensorND &dst_tensor, const TensorND &bias_tensor) {
    size_t batch_size = dst_tensor.layout.shape[0];
    size_t channel_size = dst_tensor.layout.shape[1];
    size_t channel_stride = dst_tensor.layout.stride[1];

    float* dst_ptr = dst_tensor.ptr<float>();
    float* bias_ptr = bias_tensor.ptr<float>();
    size_t dst_pos = 0, bias_pos = 0, channel_offset = 0;
    __m128 bias, dst1, dst2;
    __m128 zero_val = _mm_setzero_ps();
    __m128 one_val = _mm_set1_ps(1.f);
    float tmpf;
    for(size_t batch = 0; batch < batch_size; ++ batch) {
        bias_pos = 0;
        for(size_t chan = 0; chan < channel_size; ++chan, ++bias_pos) {
            bias = _mm_set1_ps(bias_ptr[bias_pos]);
            channel_offset += channel_stride;
            if(channel_stride > 7) {
                for(; dst_pos + 7 < channel_offset;
                      dst_pos += 8, dst_ptr += 8) {
                    dst1 = _mm_loadu_ps(dst_ptr);
                    dst2 = _mm_loadu_ps(dst_ptr + 4);

                    dst1 = _mm_add_ps(dst1, bias);
                    dst2 = _mm_add_ps(dst2, bias);

                    dst1 = _mm_sub_ps(zero_val, dst1);
                    dst2 = _mm_sub_ps(zero_val, dst2);

                    dst1 = exp_ps(dst1);
                    dst2 = exp_ps(dst2);

                    dst1 = _mm_add_ps(one_val, dst1);
                    dst2 = _mm_add_ps(one_val, dst2);

                    dst1 = _mm_div_ps(one_val,dst1);
                    dst2 = _mm_div_ps(one_val,dst2);

                    _mm_storeu_ps(dst_ptr, dst1);
                    _mm_storeu_ps(dst_ptr + 4, dst2);
                }
            }
            for(; dst_pos < channel_offset; ++dst_pos, ++dst_ptr) {
                tmpf = *dst_ptr + bias_ptr[bias_pos];
                tmpf = exp(- tmpf);
                tmpf = 1.f / (1.f + tmpf);
                *dst_ptr = tmpf;
            }
        }
    }
}

void sse_element_relu(size_t tsize, float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m128 val1, val2;
    __m128 zero_val = _mm_setzero_ps();

    if(tsize > 7) {
        for(; cur_pos + 7 < tsize; cur_pos += 8, src_ptr += 8, dst_ptr += 8) {
            val1 = _mm_loadu_ps(src_ptr);
            val2 = _mm_loadu_ps(src_ptr + 4);

            val1 = _mm_max_ps(val1, zero_val);
            val2 = _mm_max_ps(val2, zero_val);

            _mm_storeu_ps(dst_ptr, val1);
            _mm_storeu_ps(dst_ptr + 4, val2);
        }
    }
    for(; cur_pos < tsize; ++cur_pos, ++src_ptr, ++dst_ptr) {
        float tmpf = *src_ptr;
        //*dst_ptr = tmpf > 0 ? tmpf : 0;
        if(tmpf > 0) {
            *dst_ptr = tmpf;
        } else {
            *dst_ptr = 0;
        }
    }
}

void sse_element_relu(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    sse_element_relu(tsize, src_ptr, dst_ptr);
}

void sse_element_sigmoid(size_t tsize, float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m128 val1, val2;
    __m128 zero_val = _mm_setzero_ps();
    __m128 one_val = _mm_set1_ps(1.f);

    for(; cur_pos + 7 < tsize; cur_pos += 8, src_ptr += 8, dst_ptr += 8) {
        val1 = _mm_loadu_ps(src_ptr);
        val2 = _mm_loadu_ps(src_ptr + 4);

        val1 = _mm_sub_ps(zero_val, val1);
        val2 = _mm_sub_ps(zero_val, val2);

        val1 = exp_ps(val1);
        val2 = exp_ps(val2);

        val1 = _mm_add_ps(one_val, val1);
        val2 = _mm_add_ps(one_val, val2);

        val1 = _mm_div_ps(one_val,val1);
        val2 = _mm_div_ps(one_val,val2);

        _mm_storeu_ps(dst_ptr, val1);
        _mm_storeu_ps(dst_ptr + 4, val2);
    }

    for(; cur_pos < tsize; ++cur_pos, ++src_ptr, ++dst_ptr) {
        float tmpf = *src_ptr;
        tmpf = exp(-tmpf);
        tmpf = 1.f / (1.f + tmpf);
        *dst_ptr = tmpf;
    }
}

void sse_element_sigmoid(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    sse_element_sigmoid(tsize, src_ptr, dst_ptr);
}

void sse_element_exp(size_t tsize, float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m128 val1, val2;

    for(; cur_pos + 7 < tsize; cur_pos += 8, src_ptr += 8, dst_ptr += 8) {
        val1 = _mm_loadu_ps(src_ptr);
        val2 = _mm_loadu_ps(src_ptr + 4);

        val1 = exp_ps(val1);
        val2 = exp_ps(val2);

        _mm_storeu_ps(dst_ptr, val1);
        _mm_storeu_ps(dst_ptr + 4, val2);
    }

    for(; cur_pos < tsize; ++cur_pos, ++src_ptr, ++dst_ptr) {
        float tmpf = *src_ptr;
        tmpf = exp(tmpf);
        *dst_ptr = tmpf;
    }
}

void sse_element_exp(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    sse_element_exp(tsize, src_ptr, dst_ptr);
}

void sse_element_pre_exp(size_t tsize, float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m128 val1, val2;
    __m128 h_val = _mm_set1_ps(88.3762626647949f);
    __m128 l_val = _mm_set1_ps(-88.3762626647949f);

    for(; cur_pos + 7 < tsize; cur_pos += 8, src_ptr += 8, dst_ptr += 8) {
        val1 = _mm_loadu_ps(src_ptr);
        val2 = _mm_loadu_ps(src_ptr + 4);

        val1 = _mm_min_ps(val1, h_val);
        val1 = _mm_max_ps(val1, l_val);
        val2 = _mm_min_ps(val2, h_val);
        val2 = _mm_max_ps(val2, l_val);

        _mm_storeu_ps(dst_ptr, val1);
        _mm_storeu_ps(dst_ptr + 4, val2);
    }

    for(; cur_pos < tsize; ++cur_pos, ++src_ptr, ++dst_ptr) {
        float tmpf = *src_ptr;
        if (tmpf > 88.3762626647949f) {
            *dst_ptr = 88.3762626647949f;
        } else if (tmpf < -88.3762626647949f) {
            *dst_ptr = -88.3762626647949f;
        }
    }
}

void sse_element_pre_exp(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    sse_element_exp(tsize, src_ptr, dst_ptr);
}

void sse_element_tanh(size_t tsize, float *src_ptr, float *dst_ptr) {
    size_t cur_pos = 0;
    __m128 val1, val2, exp1, exp2, rexp1, rexp2;
    //__m128 zero_val = _mm_setzero_ps();
    __m128 one_val = _mm_set1_ps(1.f);

    for(; cur_pos + 7 < tsize; cur_pos += 8, src_ptr += 8, dst_ptr += 8) {
        val1 = _mm_loadu_ps(src_ptr);
        val2 = _mm_loadu_ps(src_ptr + 4);

        exp1 = exp_ps(val1);
        exp2 = exp_ps(val2);
        rexp1 = _mm_div_ps(one_val, exp1);
        rexp2 = _mm_div_ps(one_val, exp2);

        val1 = _mm_sub_ps(exp1, rexp1);
        val2 = _mm_sub_ps(exp2, rexp2);
        exp1 = _mm_add_ps(exp1, rexp1);
        exp2 = _mm_add_ps(exp2, rexp2);

        val1 = _mm_div_ps(val1, exp1);
        val2 = _mm_div_ps(val2, exp2);

        _mm_storeu_ps(dst_ptr, val1);
        _mm_storeu_ps(dst_ptr + 4, val2);
    }

    for(; cur_pos < tsize; ++cur_pos, ++src_ptr, ++dst_ptr) {
        float tmpf = exp(*src_ptr);
        float tmpf2 = 1 / tmpf;
        *dst_ptr = (tmpf - tmpf2) / (tmpf + tmpf2);
    }
}

void sse_element_tanh(const TensorND &src_tensor, const TensorND &dst_tensor) {
    size_t tsize = src_tensor.layout.total_nr_elems();
    float* src_ptr = src_tensor.ptr<float>();
    float* dst_ptr = dst_tensor.ptr<float>();
    sse_element_tanh(tsize, src_ptr, dst_ptr);
}

} // namespace detail
} // namespace x86
} // namespace megdnn
