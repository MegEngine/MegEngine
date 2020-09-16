/**
 * \file dnn/src/fallback/matrix_mul/opr_impl.h
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
#include <unordered_map>
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/base.h"
#include "src/common/algo_base.h"
#include "src/common/utils.h"
#include "src/naive/matrix_mul/opr_impl.h"

namespace megdnn {

struct AlgoTypePack {
    detail::AlgoDataType data_type : 32;
    param::MatrixMul::Format format : 32;
};

namespace fallback {
class MatrixMulImpl : public naive::MatrixMulForwardImpl {
public:
    using naive::MatrixMulForwardImpl::MatrixMulForwardImpl;
    using AlgoDataType = detail::AlgoDataType;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&) override;

    void exec(_megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
              _megdnn_workspace workspace) override;

    struct KernSizeParam {
        DType A_type, B_type, C_type;
        size_t M, N, K;
        size_t LDA, LDB, LDC;
        bool trA, trB;
        Param::ComputeMode compute_mode;
        Param::Format format;
        //! get the data type category of the param for select the algo
        AlgoDataType deduce_algo_data_type() const;
    };

    struct KernParam : public KernSizeParam {
        const void* A_ptr;
        const void* B_ptr;
        void* C_ptr;
        void* workspace_ptr;
        size_t workspace_size;

        template <typename T>
        inline const T* A() const {
            // A_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(A_ptr);
        }

        template <typename T>
        inline const T* B() const {
            // B_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(B_ptr);
        }

        template <typename T>
        inline T* C() const {
            // C_type.assert_is_compatible_ctype<T>();
            return static_cast<T*>(C_ptr);
        }
        template <typename T>
        inline T* workspace() const {
            return static_cast<T*>(workspace_ptr);
        }
    };

    typedef void (*kern_t)(const KernParam&);
    typedef void (*kern_naked_t)(const KernParam&, const void* a_panel,
                                 const void* b_panel);
    class AlgoBase : public Algorithm {
    protected:
        virtual ~AlgoBase() = default;

        bool can_be_treated_as_int8x8x32(const KernSizeParam& param) const {
            return param.A_type.enumv() == param.B_type.enumv() &&
                   (param.A_type.enumv() == DTypeEnum::Int8 ||
                    param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
                   (param.C_type.enumv() == DTypeEnum::Int32 ||
                    param.C_type.enumv() == DTypeEnum::QuantizedS32) &&
                   param.compute_mode == Param::ComputeMode::DEFAULT &&
                   param.format == param::MatrixMul::Format::DEFAULT;
        }

        bool can_be_treated_as_int8x8x16(const KernSizeParam& param) const {
            return param.A_type.enumv() == param.B_type.enumv() &&
                   (param.A_type.enumv() == DTypeEnum::Int8 ||
                    param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
                   (param.C_type.enumv() == DTypeEnum::Int16 ||
                    param.C_type.enumv() == DTypeEnum::QuantizedS16);
        }

    public:
        AlgoBase() { m_handle_type = Handle::HandleType::FALLBACK; }
        enum class AlgoType : uint32_t {
            //! fallback
            FB_F32K8x12x1 = 1 << 0,
            FB_GEMV,

#if MEGDNN_X86
            //! x86
            X86_F32_BLAS = 1 << 8,
            X86_F32_MKL_PACKA,
            X86_INT8X8X32_AVX2_2X4X16,
            X86_INT8X8X32_AVX2_4X16X2,
            X86_INT8X8X16_AVX2,
            X86_INT8X8X16_SSE,
            X86_INT8X8X32_SSE_4X8X2,
            X86_F32_MK8_8X8,
            X86_INT8X8X32_VNNI,
            X86_INT8X8X32_MKLDNN,
#elif MEGDNN_AARCH64 || MEGDNN_ARMV7
            ARM_COMMON_INT8X8X16 = 1 << 8,
            ARM_COMMON_INT8X8X32_GEMV,
            ARM_COMMON_INT8X8X32_GEMV_MK4,
            ARM_COMMON_INT8X8X32_GEMV_MK4_DOT,
            ARM_COMMON_F32_GEMV_MK4,
            ARM_COMMON_F16_GEMV,
            ARM_COMMON_GEVM,
#if MEGDNN_AARCH64
            AARCH64_F32_K8X12X1 = 1 << 16,
            AARCH64_F32_MK4_K8X12X1,
            AARCH64_F32_K4X16X1,
            AARCH64_F32_MK4_4x16,
            AARCH64_F32_GEMV,
            AARCH64_F16_K8X24X1,
            AARCH64_F16_MK8_8X8,
            AARCH64_INT8X8X32_K8X12X4_DOTPROD,
            AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD,
            AARCH64_INT8X8X32_MK4_4X4X16,
            AARCH64_INT8X8X32_K4X4X16,
            AARCH64_INT8X8X32_K8X8X8,
            AARCH64_INT8X8X16_K8X8X8,
            AARCH64_INT8X8X16_K4X4X16,
            AARCH64_INT8X8X16_MK4_16X12X4,
            AARCH64_INT8X8X16_MK4_K8X8X8,
            AARCH64_INT8X8X16_MK4_4X4X8,
            AARCH64_INT16X16X32_K12X8X1,
            AARCH64_INT16X16X32_MK8_8X8,
            AARCH64_QUINT8_K8X8X4_DOTPROD,
            AARCH64_QUINT8_GEMV_DOTPROD,
            AARCH64_QUINT8_K8X8X8,
            AARCH64_INT4X4X16_K8X8X8,
#else
            ARMV7_F32 = 1 << 16,
            ARMV7_F32_MK4_PACK_4X12,
            ARMV7_F32_MK4_4x8,
            ARMV7_F16_K4X16X1,
            ARMV7_F16_MK8_4X8,
            ARMV7_INT8_K6X8X4,
            ARMV7_QUINT8_K4X8X4,
            ARMV7_INT8_MK4_8X4X4_DOTPROD,
            ARMV7_F32_GEMV,
            ARMV7_INT8X8X32_K4X2X16,
            ARMV7_INT8X8X32_K4X8X8,
            ARMV7_QUINT8_K4X8X8,
            ARMV7_INT8X8X16_K4X2X16,
            ARMV7_INT8X8X16_K4X8X8,
            ARMV7_INT8X8X16_MK4_K8X8X4,
            ARMV7_INT16X16X32_K12X4X1,
            ARMV7_INT16X16X32_MK8_4X8,
            ARMV7_INT8X8X32_MK4_4X2X16
#endif
#endif
        };

        enum class AlgoSet : uint32_t {
            ALGO_TYPE_GEMM = 0,
            ALGO_TYPE_GEMV = 1,
        };

        enum class PackMode : uint32_t {
            DEFAULT = 0,
            NO_PACK = 1,
            ONLY_PACKA = 2,
        };

        struct InnerBlockSize {
            size_t m, n, k;
        };

        struct MatmulDescription {
            PackMode packmode;
            InnerBlockSize innerblocksize;
            AlgoTypePack algo_type;
            size_t packa_type_size;
        };

        virtual bool usable(const KernSizeParam&) const = 0;
        virtual bool preferred(const KernSizeParam&) const { return true; }
        virtual size_t get_workspace(const KernSizeParam&) const = 0;
        virtual kern_t get_kern(const KernSizeParam&) const = 0;
        virtual kern_naked_t get_kern_naked(const KernSizeParam&) const {
            megdnn_assert(0);
        };
        virtual AlgoSet algoset() const { return AlgoSet::ALGO_TYPE_GEMM; }
        virtual PackMode packmode() const { return PackMode::DEFAULT; }
        virtual void pack_A(const KernParam&, void*, size_t, size_t) const {
            megdnn_assert(0);
        };
        virtual void pack_B(const KernParam&, void*, size_t, size_t) const {
            megdnn_assert(0);
        };
        virtual WorkspaceBundle get_bundle(const KernSizeParam&) const {
            megdnn_assert(0);
        };
        virtual InnerBlockSize get_inner_block_size() const {
            megdnn_assert(0);
        };
        bool preferred_reproducible(const KernSizeParam& param,
                                    bool reproducible = true) {
            return (!reproducible || is_reproducible()) && preferred(param);
        };
        virtual MatmulDescription matmul_description() const = 0;

        using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
    };

private:
    class AlgoF32K8x12x1;  // Fallback F32 Kernel 8x12x1
    class AlgoGemv;
    class AlgoPack;
    //! maintain all the algos of in the opr of fallback
    static const AlgoPack& algo_pack();
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);
public:

    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> get_all_packed_algo();

    /**
     * \brief select algo according to input algo type
     */
    SmallVector<AlgoBase*> select_algo_type(AlgoTypePack algo_type);

protected:
    KernSizeParam make_kern_size_param(const TensorLayout& A,
                                       const TensorLayout& B,
                                       const TensorLayout& C);

    KernParam make_kern_param(_megdnn_tensor_in A, _megdnn_tensor_in B,
                              _megdnn_tensor_out C,
                              _megdnn_workspace workspace);

    std::vector<Algorithm*> get_all_algorithms(const TensorLayout& A,
                                               const TensorLayout& B,
                                               const TensorLayout& C) override;

    Algorithm* get_algorithm_heuristic(const TensorLayout& A,
                                       const TensorLayout& B,
                                       const TensorLayout& C,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
