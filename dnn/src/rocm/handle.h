/**
 * \file dnn/src/rocm/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megcore_rocm.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/common/handle_impl.h"
#include "src/common/utils.h"
#include "src/rocm/miopen_with_check.h"

#include <rocblas-types.h>
#include <rocblas.h>
#include <atomic>
#include <mutex>

namespace megdnn {
namespace rocm {

class HandleImpl : public HandleImplHelper {
public:
    HandleImpl(megcoreComputingHandle_t computing_handle);
    ~HandleImpl() noexcept;

    size_t alignment_requirement() const override;

    bool check_cross_dev_copy_constraint(const TensorLayout& src) override;

    const hipDeviceProp_t& device_prop() const { return m_device_prop; }

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    const megcore::ROCMContext& megcore_context() const {
        return m_megcore_context;
    }

    bool enable_miopen_algo_search() const {
        return megcore::ROCMContext::enable_miopen_algo_search();
    }

    void enable_miopen_algo_search(bool enable_algo_search) {
        megcore::ROCMContext::enable_miopen_algo_search(enable_algo_search);
    }

    int device_id() const { return m_device_id; }

    hipStream_t stream() const { return megcore_context().stream; }
    miopenHandle_t miopen_handle() { return m_miopen_handle; }
    rocblas_handle get_rocblas_handle() { return m_rocblas_handle; }
    dt_float32* zero_device() { return &m_const_scalars->f32[0]; }
    dt_float32* one_device() { return &m_const_scalars->f32[1]; }
#if !MEGDNN_DISABLE_FLOAT16
    __half* zero_device_h() { return &m_const_scalars->f16[0].hip_x; }
    __half* one_device_h() { return &m_const_scalars->f16[1].hip_x; }
#endif
    dt_int32* zero_device_i32() { return &m_const_scalars->i32[0]; }
    dt_int32* one_device_i32() { return &m_const_scalars->i32[1]; }

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

private:
    int m_device_id;
    //! MegDNN handle does not manage the lifetime of HIP stream.
    megcore::ROCMContext m_megcore_context;

    miopenHandle_t m_miopen_handle;
    rocblas_handle m_rocblas_handle;

    hipDeviceProp_t m_device_prop;

    struct ConstScalars {
#if !MEGDNN_DISABLE_FLOAT16
        union FP16 {
            __half hip_x;
            dt_float16 megdnn_x;
            FP16() {}
        };
        static_assert(sizeof(FP16) == 2, "bad FP16 size");
        FP16 f16[2];
#endif
        dt_float32 f32[2];
        dt_int32 i32[2];
        void init();
    };

    //! device ptr to const scalars
    ConstScalars* m_const_scalars;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
