/**
 * \file dnn/src/cuda/batched_matrix_mul/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include <cuda.h>
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/cuda/batched_matrix_mul/opr_impl.h"
#include "src/cuda/matrix_mul/cublasLt_wrapper.h"

#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

namespace megdnn {
namespace cuda {

class BatchedMatrixMulForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_BRUTE_FORCE,
        CUDA_CUBLAS,
        CUDA_CUBLASLT,
        CUDA_INT8X8X32,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        BatchedMatrixMulForwardImpl* opr;
        TensorLayout layout_a, layout_b, layout_c;
        std::string to_string() const;
        SizeArgs(
                BatchedMatrixMulForwardImpl* o, const TensorLayout& A,
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
        ExecArgs(
                BatchedMatrixMulForwardImpl* o, _megdnn_tensor_in A,
                _megdnn_tensor_in B, _megdnn_tensor_in C, _megdnn_workspace workspace);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;
    virtual const char* name() const = 0;
    bool is_available_wk(const SizeArgs& args, size_t limit) {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }
    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT,
            size_t limit = std::numeric_limits<size_t>::max()) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available_wk(args, limit);
    }
    AlgoBase& check_workspace(const SizeArgs& args, const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "batched matrix mul fwd algo %s: required workspace %zu "
                "bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }
};
class BatchedMatrixMulForwardImpl::AlgoBruteForce final
        : public BatchedMatrixMulForwardImpl::AlgoBase {
    using Param = MatrixMulForward::Param;

private:
    WorkspaceBundle get_workspace_bundle();

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /*args*/) const override;
    void exec(const ExecArgs& args) const final;
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "BRUTE_FORCE"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_BRUTE_FORCE)

    std::vector<SearchItem> get_subopr_list(
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;
};
class BatchedMatrixMulForwardImpl::AlgoCublas final
        : public BatchedMatrixMulForwardImpl::AlgoBase {
public:
    AlgoCublas() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /*args*/) const override;
    void exec(const ExecArgs& args) const final;
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }
    const char* name() const override { return "CUBLAS"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CUBLAS)
};
#if CUDA_VERSION >= 10010
class BatchedMatrixMulForwardImpl::AlgoCublasLt final : public AlgoBase {
public:
    AlgoCublasLt() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /*args*/) const override;
    void exec(const ExecArgs& args) const final;
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }
    const char* name() const override { return "CUBLAS_LT"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CUBLASLT)
};
#endif
class BatchedMatrixMulForwardImpl::AlgoInt8x8x32 final
        : public BatchedMatrixMulForwardImpl::AlgoBase {
public:
    AlgoInt8x8x32() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /*args*/) const override;
    void exec(const ExecArgs& args) const final;
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "INT8x8x32"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_INT8X8X32)
};

class BatchedMatrixMulForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;
    MatrixMulForwardImpl::AlgoPack mm_pack;

public:
    AlgoPack();

    AlgoCublas cublas;
#if CUDA_VERSION >= 10010
    AlgoCublasLt cublasLt;
#endif
    AlgoInt8x8x32 int8x8x32;
    std::vector<AlgoBase*> all_algos;
    AlgoBruteForce brute_force;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};
}  // namespace cuda
}  // namespace megdnn
