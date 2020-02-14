/**
 * \file dnn/src/x86/matrix_mul/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/x86/matrix_mul/opr_impl.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace x86 {

class MatrixMulImpl::AlgoF32Blas : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_F32_BLAS"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};

#if defined(MEGDNN_X86_WITH_MKL)
class MatrixMulImpl::AlgoF32MKLPackA : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_F32_MKL_PACKA"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    PackMode packmode() const override { return PackMode::ONLY_PACKA; }
    kern_naked_t get_kern_naked(const KernSizeParam&) const override;
    void pack_A(const KernParam& kern_param, void* out, size_t index,
                size_t stride) const override;
    void pack_B(const KernParam&, void*, size_t, size_t) const override {
        megdnn_assert(0);
    };
    WorkspaceBundle get_bundle(const KernSizeParam& param) const override;
    InnerBlockSize get_inner_block_size() const override { return {8, 16, 1}; };
};
#endif

class MatrixMulImpl::AlgoInt8x8x32AVX2M2N4K16 : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_INT8X8X32_AVX2_2X4X16"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
};

class MatrixMulImpl::AlgoInt8x8x32AVX2M4N16K2 : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_INT8X8X32_AVX2_4X16X2"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};

class MatrixMulImpl::AlgoInt8x8x32SSEM4N8K2 : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_INT8X8X32_SSE_4X8X2"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};

class MatrixMulImpl::AlgoF32MK8_8x8 : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_F32MK8_8X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};

#if MEGDNN_X86_WITH_VNNI
class MatrixMulImpl::AlgoInt8x8x32Vnni : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_INT8X8X32_VNNI"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
};
#endif

#if defined(MEGDNN_X86_WITH_MKL_DNN)
class MatrixMulImpl::AlgoInt8x8x32Mkldnn : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_INT8X8X32_MKLDNN"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_x86_algo_type; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};
#endif
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
