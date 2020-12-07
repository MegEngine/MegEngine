/**
 * \file dnn/src/cuda/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megcore_cuda.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/common/utils.h"
#include "src/common/handle_impl.h"
#include "src/cuda/cudnn_with_check.h"

#include <atomic>
#include <mutex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cuda.h>
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

namespace megdnn {
namespace cuda {

class HandleImpl: public HandleImplHelper {
    public:
        HandleImpl(megcoreComputingHandle_t computing_handle);
        ~HandleImpl() noexcept;

        size_t alignment_requirement() const override;

        bool check_cross_dev_copy_constraint(const TensorLayout &src) override;

        const cudaDeviceProp& device_prop() const {
            return *m_device_prop;
        }

        template <typename Opr>
        std::unique_ptr<Opr> create_operator();

        const megcore::CudaContext& megcore_context() const {
            return m_megcore_context;
        }

        int device_id() const { return m_device_id; }

        cudaStream_t stream() const {
            return megcore_context().stream;
        }
        cudnnHandle_t cudnn_handle() {
            return m_cudnn_handle;
        }
        cublasHandle_t cublas_handle() {
            return m_cublas_handle;
        }
#if CUDA_VERSION >= 10010
        cublasLtHandle_t cublasLt_handle() {
            return m_cublasLt_handle;
        }
#endif
        cusolverDnHandle_t cusolver_handle() {
            std::call_once(m_cusolver_initialized,
                           [this] { initialize_cusolver(); });
            return m_cusolver_handle;
        }
        dt_float32 *zero_device() {
            return &m_const_scalars->f32[0];
        }
        dt_float32 *one_device() {
            return &m_const_scalars->f32[1];
        }
        __half* zero_device_h() {
            return &m_const_scalars->f16[0].cuda_x;
        }
        __half* one_device_h() {
            return &m_const_scalars->f16[1].cuda_x;
        }
        dt_int32 *zero_device_i32() {
            return &m_const_scalars->i32[0];
        }
        dt_int32 *one_device_i32() {
            return &m_const_scalars->i32[1];
        }

        bool is_tegra_k1() const {
            return m_is_tegra_k1;
        }

        //! global matmul opr
        MatrixMul* matmul_opr() override final {
            return get_helper_opr<MatrixMul, 0>(this);
        }

        //! global matmul opr with first operand transposed
        MatrixMul* matmul_aT_opr() override final {
            return get_helper_opr<MatrixMul, 1>(this, {true, false});
        }

        //! global matmul opr with second operand transposed
        MatrixMul* matmul_bT_opr() override final {
            return get_helper_opr<MatrixMul, 2>(this, {false, true});
        }

        //! global relayout opr
        Relayout* relayout_opr() override final {
            return get_helper_opr<Relayout, 3>(this);
        }

        BatchedMatrixMulForward* batched_matrix_mul() {
            return get_helper_opr<BatchedMatrixMulForward, 4>(this);
        }

        TypeCvt* typecvt_opr() { return get_helper_opr<TypeCvt, 0>(this); }

        size_t image2d_pitch_alignment() const override;
    private:
        bool m_is_tegra_k1;
        int m_device_id;
        //! MegDNN handle does not manage the lifetime of CUDA stream.
        megcore::CudaContext m_megcore_context;

        cudnnHandle_t m_cudnn_handle;
        cublasHandle_t m_cublas_handle;
#if CUDA_VERSION >= 10010
        cublasLtHandle_t m_cublasLt_handle;
#endif
        cusolverDnHandle_t m_cusolver_handle;
        std::once_flag m_cusolver_initialized;

        const cudaDeviceProp* m_device_prop;

        struct ConstScalars {
            union FP16 {
                __half cuda_x;
                dt_float16 megdnn_x;
                FP16() {}
            };
            static_assert(sizeof(FP16) == 2, "bad FP16 size");
            FP16 f16[2];
            dt_float32 f32[2];
            dt_int32 i32[2];
            void init();
        };

        //! device ptr to const scalars
        ConstScalars* m_const_scalars;

        void initialize_cusolver();
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
