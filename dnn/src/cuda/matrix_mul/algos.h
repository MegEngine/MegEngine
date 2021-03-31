/**
 * \file dnn/src/cuda/matrix_mul/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"
#include "src/cuda/matrix_mul/opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

#include <unordered_map>
#include <cuda.h>
#include <memory>
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for matrix mul algos
 *
 */
class MatrixMulForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_CUBLAS,
        CUDA_WMMA_UINT4X4X32,
        CUDA_CUBLASLT,
        CUDA_NAIVE,
        CUDA_BFLOAT16, 
#if CUDA_VERSION >= 9020
        CUDA_FLOAT32_SIMT, 
        CUDA_FLOAT32_SIMT_SPLIT_K, 
        CUDA_FLOAT32_SIMT_GEMV_BATCHED_STRIDED, 
#endif
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        MatrixMulForwardImpl* opr;
        TensorLayout layout_a, layout_b, layout_c;

        std::string to_string() const;
        SizeArgs(MatrixMulForwardImpl* opr, const TensorLayout& A,
                 const TensorLayout& B, const TensorLayout& C);

        bool can_be_treated_as_int8x8x32() const {
            return layout_a.dtype.enumv() == layout_b.dtype.enumv() &&
                   (layout_a.dtype.enumv() == DTypeEnum::Int8 ||
                    layout_a.dtype.enumv() == DTypeEnum::QuantizedS8) &&
                   (layout_c.dtype.enumv() == DTypeEnum::Int32 ||
                    layout_c.dtype.enumv() == DTypeEnum::QuantizedS32) &&
                   opr->param().format == param::MatrixMul::Format::DEFAULT;
        }
    };
    struct ExecArgs : public SizeArgs {
        TensorND tensor_a, tensor_b, tensor_c;
        Workspace workspace;

        ExecArgs(MatrixMulForwardImpl* opr, _megdnn_tensor_in A,
                 _megdnn_tensor_in B, _megdnn_tensor_out C,
                 _megdnn_workspace workspace);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_wk(const SizeArgs& args, size_t limit) const {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }
    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT,
            size_t limit = std::numeric_limits<size_t>::max()) const {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) &&
               is_available_wk(args, limit);
    }
    AlgoBase& check_workspace(const SizeArgs& args,
                              const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "matrix mul fwd algo %s: required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }
};

class MatrixMulForwardImpl::AlgoCuBlas final : public AlgoBase {
public:
    AlgoCuBlas() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override {
        return 0_z;
    }
    const char* name() const override { return "CUBLAS"; }
    void exec(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_CUBLAS)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};

#if CUDA_VERSION >= 10000
class MatrixMulForwardImpl::AlgoUInt4x4x32WMMA final : public AlgoBase {
public:
    AlgoUInt4x4x32WMMA() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return "UINT4x4x32_WMMA"; }
    void exec(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_WMMA_UINT4X4X32)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};
#endif
#if CUDA_VERSION >= 10010
class MatrixMulForwardImpl::AlgoCuBlasLt final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return "CUBLAS_LT"; }
    void exec(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_CUBLASLT)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};
#endif

class MatrixMulForwardImpl::AlgoNaive final : public AlgoBase {
public:
    AlgoNaive() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override {
        return 0_z;
    }
    const char* name() const override { return "NAIVE"; }
    void exec(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_NAIVE)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
};

#if !MEGDNN_DISABLE_FLOAT16
class MatrixMulForwardImpl::AlgoBFloat16 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_BFLOAT16)

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts,
            const OperatorBase* opr) const override;

    const char* name() const override { return "MATMUL_BFLOAT16"; }

    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};
#endif

#if CUDA_VERSION >= 9020
class MatrixMulForwardImpl::AlgoFloat32SIMT final : public AlgoBase {
public:
    struct AlgoParam {
        int threadblock_m, threadblock_n, threadblock_k;
        int warp_m, warp_n, warp_k;
        std::string to_string() {
            return ssprintf("%dX%dX%d_%dX%dX%d", threadblock_m, threadblock_n,
                            threadblock_k, warp_m, warp_n, warp_k);
        }
    };
    AlgoFloat32SIMT(AlgoParam algo_param)
            : m_algo_param{algo_param},
              m_name{ssprintf("CUTLASS_FLOAT32_SIMT_%s",
                              m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    void exec(const ExecArgs& args) const override;
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT32_SIMT)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    AlgoParam m_algo_param;
    std::string m_name;
};

class MatrixMulForwardImpl::AlgoFloat32SIMTSplitK final : public AlgoBase {
public:
    using AlgoParam = MatrixMulForwardImpl::AlgoFloat32SIMT::AlgoParam;
    AlgoFloat32SIMTSplitK(AlgoParam algo_param)
            : m_algo_param{algo_param},
              m_name{ssprintf("CUTLASS_FLOAT32_SIMT_SPLIT_K_%s",
                              m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    void exec(const ExecArgs& args) const override;
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT32_SIMT_SPLIT_K)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    AlgoParam m_algo_param;
    std::string m_name;
};

class MatrixMulForwardImpl::AlgoFloat32SIMTGemvBatchedStrided final
        : public AlgoBase {
public:
    AlgoFloat32SIMTGemvBatchedStrided(int threadblock_n)
            : m_threadblock_n{threadblock_n},
              m_name{ssprintf("CUTLASS_FLOAT32_SIMT_GEMV_BATCHED_STRIDED_%d",
                              m_threadblock_n)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    void exec(const ExecArgs& args) const override;
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT32_SIMT_GEMV_BATCHED_STRIDED)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_threadblock_n, ret);
        return ret;
    }

private:
    int m_threadblock_n;
    std::string m_name;
};
#endif

class MatrixMulForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoCuBlas cublas;
    AlgoNaive naive;
#if CUDA_VERSION >= 10000
    AlgoUInt4x4x32WMMA wmma_uint4x4x32;
#endif
#if CUDA_VERSION >= 10010
    AlgoCuBlasLt cublas_lt;
#endif
#if !MEGDNN_DISABLE_FLOAT16
    AlgoBFloat16 bfloat16;
#endif
#if CUDA_VERSION >= 9020
    std::vector<AlgoFloat32SIMT> simt_float32;
    std::vector<AlgoFloat32SIMTSplitK> simt_float32_split_k;
    std::vector<AlgoFloat32SIMTGemvBatchedStrided>
            simt_float32_gemv_batched_strided;
#endif
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
    void fill_cutlass_algos();
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
