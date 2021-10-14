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
#include <cuda.h>
#include <memory>
#include <unordered_map>
#include "megdnn/oprs.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/matrix_mul/opr_impl.h"
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
        CUDA_CONV1X1_CUDNN,
#if CUDA_VERSION >= 9020
        CUDA_FLOAT32_SIMT,
        CUDA_FLOAT32_SIMT_SPLIT_K,
        CUDA_FLOAT32_SIMT_GEMV_BATCHED_STRIDED,
        CUDA_FLOAT16_TENSOR_OP,
        CUDA_FLOAT16_TENSOR_OP_SPLIT_K,
#endif
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        MatrixMulForwardImpl* opr;
        TensorLayout layout_a, layout_b, layout_c;

        std::string to_string() const;
        SizeArgs(
                MatrixMulForwardImpl* opr, const TensorLayout& A, const TensorLayout& B,
                const TensorLayout& C);

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
                MatrixMulForwardImpl* opr, _megdnn_tensor_in A, _megdnn_tensor_in B,
                _megdnn_tensor_out C, _megdnn_workspace workspace);
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
               !contain_attribute_any(negative_attr) && is_available_wk(args, limit);
    }
    AlgoBase& check_workspace(const SizeArgs& args, const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "matrix mul fwd algo %s: required workspace %zu bytes, got %zu", name(),
                req, workspace.size);
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
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::USABLE_DEPEND_ON_SHAPE |
               AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
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
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
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
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
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
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
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
            const TensorLayoutArray& layouts, const OperatorBase* opr) const override;

    const char* name() const override { return "MATMUL_BFLOAT16"; }

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};
#endif

class MatrixMulForwardImpl::AlgoConv1X1CUDNN final : public AlgoBase {
public:
    AlgoConv1X1CUDNN(cudnnConvolutionFwdAlgo_t algo_enum) {
        m_impl = std::make_unique<ConvBiasForwardImpl::AlgoCUDNNConv>(
                ConvBiasForwardImpl::AlgoCUDNNConv(algo_enum));
        std::string algoname(m_impl.get()->name());
        m_name = "MATMUL_CONV1X1:" + algoname;
    }
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    void exec(const ExecArgs& args) const override;
    AlgoAttribute attribute() const override {
        auto ret = AlgoAttribute::DEFAULT;
#define cb(attr)                                     \
    if (m_impl.get()->contain_attribute_all(attr)) { \
        ret |= attr;                                 \
    }
        MEGDNN_FOREACH_ALGO_ATTRIBUTE_INHERITABLE(cb)
#undef cb
        if (m_impl.get()->contain_attribute_all(AlgoAttribute::REPRODUCIBLE)) {
            ret |= AlgoAttribute::REPRODUCIBLE;
        }
        return ret;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CONV1X1_CUDNN)
private:
    std::unique_ptr<ConvBiasForwardImpl::AlgoCUDNNConv> m_impl;
    std::string m_name;
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

#if CUDA_VERSION >= 9020
class MatrixMulForwardImpl::AlgoCutlassMatrixMulBase : public AlgoBase {
public:
    struct AlgoParam {
        int threadblock_m, threadblock_n, threadblock_k;
        int warp_m, warp_n, warp_k;
        int instruction_m, instruction_n, instruction_k;
        AlgoParam(
                int threadblock_m_, int threadblock_n_, int threadblock_k_, int warp_m_,
                int warp_n_, int warp_k_, int instruction_m_ = 1,
                int instruction_n_ = 1, int instruction_k_ = 1)
                : threadblock_m{threadblock_m_},
                  threadblock_n{threadblock_n_},
                  threadblock_k{threadblock_k_},
                  warp_m{warp_m_},
                  warp_n{warp_n_},
                  warp_k{warp_k_},
                  instruction_m{instruction_m_},
                  instruction_n{instruction_n_},
                  instruction_k{instruction_k_} {}
        std::string to_string() const;
    };
    AlgoCutlassMatrixMulBase(AlgoParam algo_param) : m_algo_param{algo_param} {}
    void exec(const ExecArgs& args) const override;
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

protected:
    virtual int min_alignment_requirement() const = 0;
    virtual void do_exec(const ExecArgs& args) const = 0;
    std::pair<bool, TensorLayoutArray> construct_aligned_layouts(
            const SizeArgs& args) const;
    int max_alignment(const SizeArgs& args) const;
    AlgoParam m_algo_param;
};

class MatrixMulForwardImpl::AlgoFloat32SIMT final : public AlgoCutlassMatrixMulBase {
public:
    AlgoFloat32SIMT(AlgoParam algo_param)
            : AlgoCutlassMatrixMulBase{algo_param},
              m_name{ssprintf(
                      "CUTLASS_FLOAT32_SIMT_%s", m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT32_SIMT)
    std::string param() const override {
        std::string ret;
        // FIXME: algo param compatible with old version, to avoid fastrun cache
        // error
        struct AlgoParam_ {
            int threadblock_m, threadblock_n, threadblock_k;
            int warp_m, warp_n, warp_k;
        };
        AlgoParam_ algo_param{m_algo_param.threadblock_m, m_algo_param.threadblock_n,
                              m_algo_param.threadblock_k, m_algo_param.warp_m,
                              m_algo_param.warp_n,        m_algo_param.warp_k};
        serialize_write_pod(algo_param, ret);
        return ret;
    }

private:
    void do_exec(const ExecArgs& args) const override;
    int min_alignment_requirement() const override { return 1; }
    std::string m_name;
    const void* get_available_op(const SizeArgs& args) const;
};

class MatrixMulForwardImpl::AlgoFloat32SIMTSplitK final
        : public AlgoCutlassMatrixMulBase {
public:
    AlgoFloat32SIMTSplitK(AlgoParam algo_param)
            : AlgoCutlassMatrixMulBase{algo_param},
              m_name{ssprintf(
                      "CUTLASS_FLOAT32_SIMT_SPLIT_K_%s",
                      m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::USABLE_DEPEND_ON_SHAPE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT32_SIMT_SPLIT_K)
    std::string param() const override {
        std::string ret;
        // FIXME: algo param compatible with old version, to avoid fastrun cache
        // error
        struct AlgoParam_ {
            int threadblock_m, threadblock_n, threadblock_k;
            int warp_m, warp_n, warp_k;
        };
        AlgoParam_ algo_param{m_algo_param.threadblock_m, m_algo_param.threadblock_n,
                              m_algo_param.threadblock_k, m_algo_param.warp_m,
                              m_algo_param.warp_n,        m_algo_param.warp_k};
        serialize_write_pod(algo_param, ret);
        return ret;
    }

private:
    void do_exec(const ExecArgs& args) const override;
    int min_alignment_requirement() const override { return 1; }
    std::string m_name;
    const void* get_available_op(const SizeArgs& args) const;
};

class MatrixMulForwardImpl::AlgoFloat32SIMTGemvBatchedStrided final : public AlgoBase {
public:
    AlgoFloat32SIMTGemvBatchedStrided(int threadblock_n)
            : m_threadblock_n{threadblock_n},
              m_name{ssprintf(
                      "CUTLASS_FLOAT32_SIMT_GEMV_BATCHED_STRIDED_%d",
                      m_threadblock_n)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    void exec(const ExecArgs& args) const override;
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
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

#if CUDA_VERSION >= 10020
class MatrixMulForwardImpl::AlgoFloat16TensorOp final
        : public AlgoCutlassMatrixMulBase {
public:
    AlgoFloat16TensorOp(AlgoParam algo_param)
            : AlgoCutlassMatrixMulBase{algo_param},
              m_name{ssprintf(
                      "CUTLASS_FLOAT16_TENSOR_OP_h%d%d%d_%s",
                      m_algo_param.instruction_m, m_algo_param.instruction_n,
                      m_algo_param.instruction_k, m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT16_TENSOR_OP)

private:
    void do_exec(const ExecArgs& args) const override;
    int min_alignment_requirement() const override { return 2; }
    std::string m_name;
};

class MatrixMulForwardImpl::AlgoFloat16TensorOpSplitK final
        : public AlgoCutlassMatrixMulBase {
public:
    AlgoFloat16TensorOpSplitK(AlgoParam algo_param)
            : AlgoCutlassMatrixMulBase{algo_param},
              m_name{ssprintf(
                      "CUTLASS_FLOAT16_TENSOR_OP_SPLIT_K_h%d%d%d_%s",
                      m_algo_param.instruction_m, m_algo_param.instruction_n,
                      m_algo_param.instruction_k, m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::USABLE_DEPEND_ON_SHAPE;
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_FLOAT16_TENSOR_OP_SPLIT_K)

private:
    void do_exec(const ExecArgs& args) const override;
    int min_alignment_requirement() const override { return 2; }
    std::string m_name;
};
#endif
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
    std::vector<AlgoFloat32SIMTGemvBatchedStrided> simt_float32_gemv_batched_strided;
#if CUDA_VERSION >= 10020
    std::vector<AlgoFloat16TensorOp> tensorop_float16;
    std::vector<AlgoFloat16TensorOpSplitK> tensorop_float16_split_k;
#endif
#endif
    std::vector<AlgoConv1X1CUDNN> conv1x1;
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
    void fill_cutlass_algos();
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
