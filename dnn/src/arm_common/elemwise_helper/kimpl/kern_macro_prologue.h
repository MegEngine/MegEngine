/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/kern_macro_prologue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#define H_SWISH_KERN(_func_suffix, _val1, _val2)                             \
    do {                                                                     \
        auto val_zero = vdupq_n_##_func_suffix(0.f);                         \
        auto val_six = vdupq_n_##_func_suffix(6.f);                          \
        auto val_three = vdupq_n_##_func_suffix(3.f);                        \
        auto val_rec_six = vdupq_n_##_func_suffix(1.f / 6.f);                \
        auto clip1 = vmaxq_##_func_suffix(                                   \
                vminq_##_func_suffix(vaddq_##_func_suffix(_val1, val_three), \
                                     val_six),                               \
                val_zero);                                                   \
        auto clip2 = vmaxq_##_func_suffix(                                   \
                vminq_##_func_suffix(vaddq_##_func_suffix(_val2, val_three), \
                                     val_six),                               \
                val_zero);                                                   \
        _val1 = vmulq_##_func_suffix(vmulq_##_func_suffix(_val1, clip1),     \
                                     val_rec_six);                           \
        _val2 = vmulq_##_func_suffix(vmulq_##_func_suffix(_val2, clip2),     \
                                     val_rec_six);                           \
    } while (0);

#define H_SWISH_KERN_N1(_func_suffix, _val1)                                 \
    do {                                                                     \
        auto val_zero = vdupq_n_##_func_suffix(0.f);                         \
        auto val_six = vdupq_n_##_func_suffix(6.f);                          \
        auto val_three = vdupq_n_##_func_suffix(3.f);                        \
        auto val_rec_six = vdupq_n_##_func_suffix(1.f / 6.f);                \
        auto clip1 = vmaxq_##_func_suffix(                                   \
                vminq_##_func_suffix(vaddq_##_func_suffix(_val1, val_three), \
                                     val_six),                               \
                val_zero);                                                   \
        _val1 = vmulq_##_func_suffix(vmulq_##_func_suffix(_val1, clip1),     \
                                     val_rec_six);                           \
    } while (0);

// vim: syntax=cpp.doxygen
