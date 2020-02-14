/**
 * \file dnn/src/x86/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/x86/matrix_mul/opr_impl.h"
#include "src/common/metahelper.h"
#include "src/x86/matrix_mul/algos.h"
#include "src/x86/utils.h"
using namespace megdnn;
using namespace x86;

namespace {
uint8_t x86_algo_type_storage;
}  // anonymous namespace

void* const MatrixMulImpl::sm_x86_algo_type = &x86_algo_type_storage;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32Blas f32blas;

#if defined(MEGDNN_X86_WITH_MKL)
    AlgoF32MKLPackA f32mkl_packa;
#endif
#if MEGDNN_X86_WITH_VNNI
    AlgoInt8x8x32Vnni algoint8x8x32vnni;
#endif
#if defined(MEGDNN_X86_WITH_MKL_DNN)
    AlgoInt8x8x32Mkldnn algoint8x8x32mkldnn;
#endif
    AlgoInt8x8x32AVX2M4N16K2 algoint8x8x32avx2_m4n16k2;
    AlgoInt8x8x32AVX2M2N4K16 algoint8x8x32avx2_m2n4k16;
    AlgoInt8x8x32SSEM4N8K2 algoint8x8x32sse_m4n8k2;
    AlgoF32MK8_8x8 algof32mk8_8x8;

public:
    AlgoPack() {
        if (is_supported(SIMDType::VNNI)) {
#if defined(MEGDNN_X86_WITH_MKL_DNN)
            all_algos.emplace_back(&algoint8x8x32mkldnn);
#endif
#if MEGDNN_X86_WITH_VNNI
            all_algos.emplace_back(&algoint8x8x32vnni);
#endif
        }
        all_algos.emplace_back(&algoint8x8x32avx2_m4n16k2);
        all_algos.emplace_back(&algoint8x8x32avx2_m2n4k16);
        all_algos.emplace_back(&algoint8x8x32sse_m4n8k2);
        all_algos.emplace_back(&algof32mk8_8x8);
#if defined(MEGDNN_X86_WITH_MKL_DNN)
        all_algos.emplace_back(&algoint8x8x32mkldnn);
#endif
        all_algos.emplace_back(&f32blas);
#if defined(MEGDNN_X86_WITH_MKL)
        all_algos.emplace_back(&f32mkl_packa);
#endif
    }
    SmallVector<AlgoBase*> all_algos;
};

SmallVector<MatrixMulImpl::AlgoBase*> MatrixMulImpl::algo_pack() {
    static AlgoPack s_algo_pack;
    auto&& algos = fallback::MatrixMulImpl::algo_pack();
    algos.insert(algos.begin(), s_algo_pack.all_algos.begin(),
                 s_algo_pack.all_algos.end());
    return std::move(algos);
}

// vim: syntax=cpp.doxygen
