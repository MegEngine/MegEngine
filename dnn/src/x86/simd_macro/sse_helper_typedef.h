//created by Victoria Zhislina, the Senior Application Engineer, Intel Corporation,  victoria.zhislina@intel.com

//*** Copyright (C) 2012-2019 Intel Corporation.  All rights reserved.

//IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

//By downloading, copying, installing or using the software you agree to this license.
//If you do not agree to this license, do not download, install, copy or use the software.

//                              License Agreement
//Redistribution and use in source and binary forms, with or without modification,
//are permitted provided that the following conditions are met:

//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.

//  * The name of the copyright holders may not be used to endorse or promote products
//    derived from this software without specific prior written permission.

//This software is provided by the copyright holders and contributors "as is" and
//any express or implied warranties, including, but not limited to, the implied
//warranties of merchantability and fitness for a particular purpose are disclaimed.
//In no event shall the Intel Corporation or contributors be liable for any direct,
//indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
//loss of use, data, or profits; or business interruption) however caused
//and on any theory of liability, whether in contract, strict liability,
//or tort (including negligence or otherwise) arising in any way out of
//the use of this software, even if advised of the possibility of such damage.
/* --------------------------------------------------------------------------
 * \file dnn/src/x86/simd_macro/sse_helper_typedef.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 * ------------------------------------------------------------------------------
 */
#pragma once

#include <xmmintrin.h> // SSE
#include <stdint.h>
// The code is from
// [NEON_2_SSE.h](https://github.com/intel/ARM_NEON_2_x86_SSE/blob/master/NEON_2_SSE.h)
// Note that the performance of tranforming neon to sse is not very efficient.
struct float32x4x2_t {
    __m128 val[2];
};

typedef union __m64_128 {
    uint64_t m64_u64[1];
    float m64_f32[2];
    int8_t m64_i8[8];
    int16_t m64_i16[4];
    int32_t m64_i32[2];
    int64_t m64_i64[1];
    uint8_t m64_u8[8];
    uint16_t m64_u16[4];
    uint32_t m64_u32[2];
} __m64_128;
typedef __m64_128 float32x2_t;


