/**
 * \file dnn/src/cuda/matrix_mul/algos.h
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
        CUDA_BFLOAT16
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
    bool is_available_reproducible(
            const SizeArgs& args, bool reproducible = true,
            size_t limit = std::numeric_limits<size_t>::max()) const {
        return (!reproducible || is_reproducible()) &&
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
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CUBLAS)
};

#if CUDA_VERSION >= 10000
class MatrixMulForwardImpl::AlgoUInt4x4x32WMMA final : public AlgoBase {
public:
    AlgoUInt4x4x32WMMA() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return "UINT4x4x32_WMMA"; }
    void exec(const ExecArgs& args) const override;
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_WMMA_UINT4X4X32)
};
#endif
#if CUDA_VERSION >= 10010
class MatrixMulForwardImpl::AlgoCuBlasLt final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return "CUBLAS_LT"; }
    void exec(const ExecArgs& args) const override;
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CUBLASLT)
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
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_NAIVE)
};

#if !MEGDNN_DISABLE_FLOAT16
class MatrixMulForwardImpl::AlgoBFloat16 final : public AlgoBase {
public:
    AlgoBFloat16(MatrixMulForwardImpl::AlgoBase*);
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    void exec(const ExecArgs& args) const override;
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_NAIVE)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algorithm, ret);
        return ret;
    }

private:
    MatrixMulForwardImpl::AlgoBase* m_algorithm = nullptr;
    std::string m_name;
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    SizeArgs float_args(const SizeArgs& args) const;
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
    std::unique_ptr<AlgoBFloat16> cublas_bfloat16;
#endif
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
