/**
 * \file dnn/src/fallback/matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/matrix_mul/opr_impl.h"
#include "src/common/utils.h"
namespace megdnn {
namespace fallback {

class MatrixMulImpl : public naive::MatrixMulForwardImpl {
public:
    using naive::MatrixMulForwardImpl::MatrixMulForwardImpl;

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
    typedef void (*kern_naked_t)(const KernParam& , const void* a_panel, const void *b_panel);
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
                   param.A_type.enumv() == DTypeEnum::Int8 &&
                   param.C_type.enumv() == DTypeEnum::Int16 &&
                   param.format == param::MatrixMul::Format::DEFAULT &&
                   param.compute_mode == Param::ComputeMode::DEFAULT;
        }
    public:
        enum class AlgoSet:uint32_t {
            ALGO_TYPE_GEMM = 0,
            ALGO_TYPE_GEMV = 1,
        };

        enum class PackMode:uint32_t {
            DEFAULT = 0,
            NO_PACK = 1,
            ONLY_PACKA = 2,
        };

        struct InnerBlockSize {
            size_t m, n, k;
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
        virtual size_t get_packA_type_size() const { megdnn_assert(0); };
        bool preferred_reproducible(const KernSizeParam& param,
                                    bool reproducible = true) {
            return (!reproducible || is_reproducible()) && preferred(param);
        };
    };

    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> algo_pack();

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

private:
    class AlgoF32K8x12x1;  // Fallback F32 Kernel 8x12x1
    class AlgoGemv;
    class AlgoPack;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
