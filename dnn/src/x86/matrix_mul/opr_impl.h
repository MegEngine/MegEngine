/**
 * \file dnn/src/x86/matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "src/common/utils.h"
#include "src/fallback/matrix_mul/opr_impl.h"
#include "src/x86/utils.h"

#if MEGDNN_X86_WITH_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#elif MEGDNN_X86_WITH_OPENBLAS
#include <cblas.h>
#else
#endif

#if MEGDNN_X86_WITH_MKL_DNN
#include <mkldnn.h>
#endif

namespace megdnn {
namespace x86 {

class MatrixMulImpl : public fallback::MatrixMulImpl {
public:
    using fallback::MatrixMulImpl::MatrixMulImpl;
    class AlgoBase : public fallback::MatrixMulImpl::AlgoBase {
    public:
        AlgoBase() : fallback::MatrixMulImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::X86;
        }
    };

    bool is_thread_safe() const override { return true; }

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> get_all_packed_algo()
            override;

    static fallback::MatrixMulImpl::AlgoBase* get_algo_from_desc(
            const AlgorithmDesc& desc);

private:
    class AlgoF32Blas;
#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
    class AlgoF32MKLPackA;
#endif
#if MEGDNN_X86_WITH_VNNI
    class AlgoInt8x8x32Vnni;
#endif

#if MEGDNN_X86_WITH_MKL_DNN
    class AlgoInt8x8x32Mkldnn;
#endif

    class AlgoInt8x8x32AVX2M2N4K16;
    class AlgoInt8x8x32AVX2M4N16K2;
    class AlgoInt8x8x32SSEM4N8K2;
    class AlgoInt8x8x16AVX2;
    class AlgoInt8x8x16SSE;
    class AlgoPack;
    class AlgoF32MK8_8x8;

public:
    static const AlgoPack& algo_pack();
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
