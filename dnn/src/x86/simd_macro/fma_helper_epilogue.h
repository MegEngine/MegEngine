/**
 * \file dnn/src/x86/simd_macro/fma_helper_epilogue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/simd_macro/epilogue.h"

#undef MEGDNN_SIMD_ADD
#undef MEGDNN_SIMD_SUB
#undef MEGDNN_SIMD_MUL
#undef MEGDNN_SIMD_FNMADD
#undef MEGDNN_SIMD_UNPACKLO
#undef MEGDNN_SIMD_UNPACKHI
#undef MEGDNN_SIMD_SHUFFLE
#undef MEGDNN_SIMD_BLEND
#undef MEGDNN_SIMD_PERMUTE2F128
