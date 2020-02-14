/**
 * \file dnn/src/common/simd_macro/epilogue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#undef MEGDNN_SIMD_NAME
#undef MEGDNN_SIMD_TARGET
#undef MEGDNN_SIMD_ATTRIBUTE_TARGET
#undef MEGDNN_SIMD_WIDTH
#undef MEGDNN_SIMD_TYPE
#undef MEGDNN_SIMD_LOADU
#undef MEGDNN_SIMD_STOREU
#undef MEGDNN_SIMD_SETZERO
#undef MEGDNN_SIMD_SET1
#undef MEGDNN_SIMD_FMADD
#undef MEGDNN_SIMD_MAX

#ifdef MEGDNN_SIMD_UZP
#undef MEGDNN_SIMD_UZP
#endif

#ifdef _INSERTPS_NDX
#undef _INSERTPS_NDX
#endif

#ifdef _M64
#undef _M64
#endif

#ifdef _M64f
#undef _M64f
#endif

#ifdef _pM128i
#undef _pM128i
#endif

#ifdef _pM128
#undef _pM128
#endif

#ifdef _M128i
#undef _M128i
#endif

#ifdef _M128
#undef _M128
#endif

#undef MEGDNN_SIMD_LOAD2
#undef MEGDNN_SIMD_EXT
#undef MEGDNN_SIMD_MUL
#undef MEGDNN_SIMD_FMA_LANE
#undef MEGDNN_SIMD_VEC
#undef MEGDNN_SIMD_SET_LANE
