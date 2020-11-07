/**
 * \file dnn/src/x86/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/x86/matrix_mul/opr_impl.h"
#include "src/common/metahelper.h"
#include "src/x86/matrix_mul/algos.h"
using namespace megdnn;
using namespace x86;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32Blas f32blas;

#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
    AlgoF32MKLPackA f32mkl_packa;
#endif
#if MEGDNN_X86_WITH_VNNI
    AlgoInt8x8x32Vnni algoint8x8x32vnni;
#endif
#if MEGDNN_X86_WITH_MKL_DNN
    AlgoInt8x8x32Mkldnn algoint8x8x32mkldnn;
#endif
    AlgoInt8x8x32AVX2M4N16K2 algoint8x8x32avx2_m4n16k2;
    AlgoInt8x8x32AVX2M2N4K16 algoint8x8x32avx2_m2n4k16;
    AlgoInt8x8x32SSEM4N8K2 algoint8x8x32sse_m4n8k2;
    AlgoInt8x8x16AVX2 algoint8x8x16avx2_m4n16k2;
    AlgoInt8x8x16SSE algoint8x8x16sse_m4n8k2;
    AlgoF32MK8_8x8 algof32mk8_8x8;

public:
    AlgoPack() {
        if (is_supported(SIMDType::VNNI)) {
#if MEGDNN_X86_WITH_VNNI
            all_algos.emplace_back(&algoint8x8x32vnni);
#endif
        }
        all_algos.emplace_back(&algoint8x8x32avx2_m4n16k2);
        all_algos.emplace_back(&algoint8x8x16avx2_m4n16k2);
        all_algos.emplace_back(&algoint8x8x32avx2_m2n4k16);
        all_algos.emplace_back(&algoint8x8x32sse_m4n8k2);
        all_algos.emplace_back(&algoint8x8x16sse_m4n8k2);
        all_algos.emplace_back(&algof32mk8_8x8);
#if MEGDNN_X86_WITH_MKL_DNN
        all_algos.emplace_back(&algoint8x8x32mkldnn);
#endif
        all_algos.emplace_back(&f32blas);
#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
        all_algos.emplace_back(&f32mkl_packa);
#endif
    }
    SmallVector<fallback::MatrixMulImpl::AlgoBase*> all_algos;
};

SmallVector<fallback::MatrixMulImpl::AlgoBase*> MatrixMulImpl::algo_pack() {
    static AlgoPack s_algo_pack;
    auto&& algos = fallback::MatrixMulImpl::algo_pack();
    algos.insert(algos.begin(), s_algo_pack.all_algos.begin(),
                 s_algo_pack.all_algos.end());
    return std::move(algos);
}

// vim: syntax=cpp.doxygen
