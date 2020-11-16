/**
 * \file dnn/src/cuda/conv_bias/algo.h
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

#include "src/common/algo_base.h"
#include "src/common/utils.h"
#include "src/common/metahelper.h"
#include "src/cuda/conv_bias/conv_bias_int8.cuh"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/cudnn_wrapper.h"

#include <cuda.h>
#include <memory>
#include <unordered_map>

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for conv bias algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class ConvBiasForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_CUDNN_CONVBIAS,
        CUDA_CHANWISE,
        CUDA_CHANWISE_SMALL,
        CUDA_CHANWISE_INT8X8X32,
        CUDA_CUDNN_CONV,
        CUDA_INPLACE_MATMUL,
        CUDA_MATMUL,
        CUDA_MATMUL_INT8X8X32,
        CUDA_1X1,
        CUDA_BATCHED_MATMUL,
        CUDA_GROUP_CONV_GENERAL,
        CUDA_WMMA_UINT4X4X32,
        CUDA_IMPLICIT_GEMM_CHWN4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_CHWN4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_NCHW4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_REORDER_FILTER_CHWN4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_UNROLL_WIDTH_CHWN4_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_IMMA_NCHW32_INT8,
        CUDA_BFLOAT16,
        CUDA_IMPLICIT_GEMM_SASS_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_1X1_SASS_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_SASS_NCHW32_IMMA_INT8,
        CUDA_IMPLICIT_GEMM_1X1_SASS_NCHW32_IMMA_INT8,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs : public conv_bias::BiasForwardSizeArgs {
        ConvBiasForwardImpl* opr;
        const PreprocessedFilter* preprocessed_filter;

        std::string to_string() const;
        SizeArgs(ConvBiasForwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& filter, const TensorLayout& bias,
                 const TensorLayout& z, const TensorLayout& dst,
                 const PreprocessedFilter* preprocessed_filter = nullptr);
        SizeArgs(ConvBiasForwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& filter,
                 const CanonizedFilterMeta& filter_meta,
                 const TensorLayout& bias, const TensorLayout& z,
                 const TensorLayout& dst,
                 const PreprocessedFilter* preprocessed_filter = nullptr);

        void init_conv_bias_desc(conv_bias::CUDNNForwardDescs& desc) const {
            desc.set_conv_bias(*src_layout, filter_meta, *dst_layout,
                               *bias_layout, *z_layout, opr->param());
        }

        void init_conv_desc(conv_bias::CUDNNForwardDescs& desc) const {
            desc.set_conv(*src_layout, filter_meta, *dst_layout, opr->param());
        }
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *filter_tensor, *bias_tensor, *z_tensor,
                *dst_tensor;
        Workspace workspace;

        ExecArgs(ConvBiasForwardImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_in filter, _megdnn_tensor_in bias,
                 _megdnn_tensor_in z, _megdnn_tensor_out dst,
                 _megdnn_workspace workspace,
                 const PreprocessedFilter* preprocessed_filter = nullptr);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;
    virtual size_t get_preprocess_workspace_in_bytes(
            const SizeArgs& args) const {
        MEGDNN_MARK_USED_VAR(args);
        return 0;
    }
    virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const {
        MEGDNN_MARK_USED_VAR(args);
        return {};
    }
    virtual void exec_preprocess(const ExecArgs& args) const {
        MEGDNN_MARK_USED_VAR(args);
    }

    bool is_available_wk(const SizeArgs& args, size_t limit) {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }

    bool is_available_reproducible(
            const SizeArgs& args, bool reproducible = true,
            size_t limit = std::numeric_limits<size_t>::max()) {
        return (!reproducible || is_reproducible()) &&
               is_available_wk(args, limit);
    }

    AlgoBase& check_workspace(const SizeArgs& args,
                              const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "conv bias fwd algo %s: required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }

    virtual bool is_cudnn() const { return false; }
};

class ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation final : public AlgoBase {
public:
    AlgoCUDNNConvBiasActivation(cudnnConvolutionFwdAlgo_t cudnn_enum)
            : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(CudnnAlgoPack::conv_fwd_algos().find(cudnn_enum) !=
                      CudnnAlgoPack::conv_fwd_algos().end());
        m_attr = CudnnAlgoPack::conv_fwd_algos().at(cudnn_enum);
        m_name = ConvBiasForward::algo_name<DefaultParam>(
                "CUDNN:ConvBiasActivation:" + m_attr.name, {});
    }

    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    param::Convolution get_param_convolution(const SizeArgs& args) const;
    bool is_available(const SizeArgs&) const override;

    const char* name() const override { return m_name.c_str(); }

    bool is_reproducible() const override { return m_attr.is_reproducible; }

    cudnnConvolutionFwdAlgo_t cudnn_enum() { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN_CONVBIAS)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }

private:
    std::string m_name;
    cudnnConvolutionFwdAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;
};

class ConvBiasForwardImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name =
                    ConvBiasForward::algo_name<DirectParam>("CHANNEL_WISE", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE)

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoChanwiseSmall final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>(
                    "CHANNEL_WISE_SMALL", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE_SMALL)

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoChanwise8x8x32 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<DirectParam>(
                    "CHANNEL_WISE_8X8X32", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_CHANWISE_INT8X8X32)

private:
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoCUDNNConv final : public AlgoBase {
public:
    AlgoCUDNNConv(cudnnConvolutionFwdAlgo_t cudnn_enum)
            : m_cudnn_enum(cudnn_enum) {
        megdnn_assert(CudnnAlgoPack::conv_fwd_algos().find(cudnn_enum) !=
                      CudnnAlgoPack::conv_fwd_algos().end());
        m_attr = CudnnAlgoPack::conv_fwd_algos().at(cudnn_enum);
        m_name = ConvBiasForward::algo_name<DefaultParam>(
                "CUDNN:Convolution:" + m_attr.name, {});
    }

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return m_attr.is_reproducible; }

    const char* name() const override { return m_name.c_str(); }

    cudnnConvolutionFwdAlgo_t cudnn_enum() const { return m_cudnn_enum; }

    bool is_cudnn() const override { return true; }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN_CONV)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_cudnn_enum, ret);
        return ret;
    }

private:
    std::string m_name;
    cudnnConvolutionFwdAlgo_t m_cudnn_enum;
    CudnnAlgoPack::Attr m_attr;

    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
};

//! compute small matmul in the kernel
class ConvBiasForwardImpl::AlgoInplaceMatmul final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBias::MatmulParam>(
                    "INPLACE_MATMUL", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_INPLACE_MATMUL)

private:
    mutable std::string m_name;
};

//! im2col and matmul, with dilation
class ConvBiasForwardImpl::AlgoMatmul final : public AlgoBase {
    template <typename T>
    static void exec_internal(const ExecArgs& args,
                              const WorkspaceBundle& bundle);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "MATMUL", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL)

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;

    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoMatmul8x8x32 final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "MATMUL8X8X32", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL_INT8X8X32)

private:
    bool need_src_unroll(const SizeArgs& args) const;
    bool need_filter_reshape(const SizeArgs& args) const;
    template <Param::Format>
    WorkspaceBundle get_bundle(const SizeArgs& args) const;
    template <Param::Format>
    void exec_internal(const ExecArgs& args) const;
    mutable std::string m_name;
};

//! optimized 1x1 conv
class ConvBiasForwardImpl::Algo1x1 final : public AlgoBase {
    static void extract_matmul_layouts(const SizeArgs& args, TensorLayout& A,
                                       TensorLayout& B, TensorLayout& C);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "MATMUL1X1", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_1X1)

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    mutable std::string m_name;
};

class ConvBiasForwardImpl::AlgoBatchedMatmul final : public AlgoBase {
    static void extract_matmul_layouts(const SizeArgs& args, TensorLayout& A,
                                       TensorLayout& B, TensorLayout& C);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "BATCHEDMATMUL", {});
        }
        return m_name.c_str();
    }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_BATCHED_MATMUL)

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    mutable std::string m_name;
};

//! implement group conv by another algo
class ConvBiasForwardImpl::AlgoGroupConvGeneral final : public AlgoBase {
public:
    AlgoGroupConvGeneral(AlgoBase* impl);

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_name.c_str(); }

    bool is_reproducible() const override { return m_impl->is_reproducible(); }

    static void modify_size_args(SizeArgs& args, TensorLayout& src_pg,
                                 TensorLayout& dst_pg, TensorLayout& bias_pg);
    MEGDNN_DECL_ALGO_TYPE(CUDA_GROUP_CONV_GENERAL)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_impl, ret);
        return ret;
    }

private:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    AlgoBase* m_impl;
    std::string m_name;
};

#if CUDA_VERSION >= 10000
class ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA final : public AlgoBase {
public:
    AlgoQUInt4x4x32WMMA() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return "QUINT4x4x32_WMMA"; }
    bool is_reproducible() const override { return true; }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
    bool use_kernel_fhxfw(const SizeArgs& args) const;
    size_t get_workspace_in_bytes_do_conv(const SizeArgs& args) const;
    MEGDNN_DECL_ALGO_TYPE(CUDA_WMMA_UINT4X4X32)
};
#endif

class ConvBiasForwardImpl::AlgoInt8CHWN4DotProdImplicitGemm final
        : public AlgoBase {
public:
    AlgoInt8CHWN4DotProdImplicitGemm() = default;
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override {
        return "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM";
    }
    bool is_reproducible() const override { return true; }
    template <typename BiasVisitor>
    static void dispatch_nonlinear_mode(
            const int8_t* d_src, const int8_t* d_filter,
            BiasVisitor bias_visitor, const int8_t* d_z, int8_t* d_dst,
            const convolution::ConvParam& param, float alpha, float beta,
            float gamma, float scale, cudaStream_t stream,
            param::ConvBias::NonlineMode nonlinear_mode);
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_CHWN4_DOTPROD_INT8)
};

class ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm final
        : public AlgoBase {
public:
    struct AlgoParam {
        int threadblock_m;
        int threadblock_n;
        int threadblock_k;
        int warp_m;
        int warp_n;
        int warp_k;
        int stage;
        std::string to_string() {
            /// default algorithm
            if (threadblock_m == 128 && threadblock_n == 128 &&
                threadblock_k == 32 && warp_m == 32 && warp_n == 64 &&
                warp_k == 32 && stage == 2) {
                return "";
            }
            return ssprintf("_%dX%dX%d_%dX%dX%d_%dstage", threadblock_m,
                            threadblock_n, threadblock_k, warp_m, warp_n,
                            warp_k, stage);
        }
    };
    AlgoInt8NCHW4DotProdImplicitGemm(AlgoParam algo_param)
            : m_algo_param{algo_param},
              m_name{ssprintf("INT8_NCHW4_DOTPROD_IMPLICIT_GEMM%s",
                              m_algo_param.to_string().c_str())} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    bool is_reproducible() const override { return true; }
    size_t get_preprocess_workspace_in_bytes(
            const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW4_DOTPROD_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
    AlgoParam m_algo_param;
    std::string m_name;
};

#if CUDA_VERSION >= 10000
class ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemm final
        : public AlgoBase {
public:
    enum class MMATileSize : uint32_t {
        IMMA16x16x16,
        IMMA32x8x16,
        IMMA8x32x16
    };
    AlgoInt8CHWN4IMMAImplicitGemm(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_CHWN4_IMMA_IMPLICIT_GEMM_" +
                     to_string(m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    bool is_reproducible() const override { return true; }
    template <typename BiasVisitor>
    static void dispatch_nonlinear_mode(
            const int8_t* d_src, const int8_t* d_filter,
            BiasVisitor bias_visitor, int8_t* d_z, int8_t* d_dst,
            const convolution::ConvParam& param, float alpha, float beta,
            float gamma, float scale, cudaStream_t stream,
            param::ConvBias::NonlineMode nonlinear_mode,
            MMATileSize mma_tile_size);
    static std::string to_string(MMATileSize mma_tile_size);

    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_CHWN4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }

private:
    MMATileSize m_mma_tile_size;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8NCHW4IMMAImplicitGemm final
        : public AlgoBase {
public:
    using MMATileSize = AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize;
    AlgoInt8NCHW4IMMAImplicitGemm(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_NCHW4_IMMA_IMPLICIT_GEMM_" +
                     AlgoInt8CHWN4IMMAImplicitGemm::to_string(
                             m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_NCHW4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
    MMATileSize m_mma_tile_size;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemmReorderFilter final
        : public AlgoBase {
public:
    using MMATileSize = AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize;
    AlgoInt8CHWN4IMMAImplicitGemmReorderFilter(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_" +
                     AlgoInt8CHWN4IMMAImplicitGemm::to_string(
                             m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_REORDER_FILTER_CHWN4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }

private:
    MMATileSize m_mma_tile_size;
    std::string m_name;
};

class ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth final
        : public AlgoBase {
public:
    using MMATileSize = AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize;
    AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth(MMATileSize mma_tile_size)
            : m_mma_tile_size{mma_tile_size},
              m_name{"INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_" +
                     AlgoInt8CHWN4IMMAImplicitGemm::to_string(
                             m_mma_tile_size)} {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_UNROLL_WIDTH_CHWN4_IMMA_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_mma_tile_size, ret);
        return ret;
    }

private:
    MMATileSize m_mma_tile_size;
    std::string m_name;
};
#endif

#if CUDA_VERSION >= 10020
class ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm final
        : public AlgoBase {
public:
    struct AlgoParam {
        int threadblock_m;
        int threadblock_n;
        int threadblock_k;
        int warp_m;
        int warp_n;
        int warp_k;
    };
    AlgoInt8NCHW32IMMAImplicitGemm(AlgoParam algo_param)
            : m_algo_param{algo_param} {
        m_name = ConvBias::algo_name<ConvBias::DirectParam>(
                ssprintf("INT8_NCHW32_IMMA_IMPLICIT_GEMM_%s",
                         to_string(m_algo_param).c_str()),
                ConvBias::DirectParam{});
    }
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;
    const char* name() const override { return m_name.c_str(); }
    bool is_reproducible() const override { return true; }
    static std::string to_string(AlgoParam algo_param);
    size_t get_preprocess_workspace_in_bytes(
            const SizeArgs& args) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const SizeArgs& args) const override;
    void exec_preprocess(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_IMMA_NCHW32_INT8)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_param, ret);
        return ret;
    }

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;

    AlgoParam m_algo_param;
    std::string m_name;
};
#endif

class ConvBiasForwardImpl::AlgoBFloat16 final : public AlgoBase {
public:
    AlgoBFloat16(AlgoBase* impl);

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_name.c_str(); }

    bool is_reproducible() const override { return m_impl->is_reproducible(); }

    MEGDNN_DECL_ALGO_TYPE(CUDA_BFLOAT16)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_impl, ret);
        return ret;
    }

private:
    SizeArgs float_args(const SizeArgs& args, ConvBiasForwardImpl* opr,
                        TensorLayout& fsrc, TensorLayout& ffilter,
                        TensorLayout& fbias, TensorLayout& fz,
                        TensorLayout& fdst) const;
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args) const;
    AlgoBase* m_impl;
    std::string m_name;
};


class ConvBiasForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    std::vector<AlgoBase*> all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos, bfloat16_algos;
    std::vector<AlgoCUDNNConvBiasActivation> cudnn_conv_bias_activations;
    std::vector<AlgoCUDNNConv> cudnn_convs;
    AlgoChanwise chanwise;
    AlgoChanwiseSmall chanwise_small;
    AlgoChanwise8x8x32 chanwise8x8x32;
    AlgoInplaceMatmul inplace_matmul;
    AlgoMatmul matmul;
    AlgoMatmul8x8x32 matmul8x8x32;
    AlgoBatchedMatmul batched_matmul;
    Algo1x1 a1x1;
    std::vector<AlgoInt8NCHW4DotProdImplicitGemm> int8_nchw4_dotprod;
    AlgoInt8CHWN4DotProdImplicitGemm int8_chwn4_dotprod;
#if CUDA_VERSION >= 10000
    AlgoQUInt4x4x32WMMA wmma_quint4x4x32;
    std::vector<AlgoInt8CHWN4IMMAImplicitGemm> int8_chwn4_imma;
    std::vector<AlgoInt8NCHW4IMMAImplicitGemm> int8_nchw4_imma;
    std::vector<AlgoInt8CHWN4IMMAImplicitGemmReorderFilter>
            int8_chwn4_imma_reorder_filter;
    std::vector<AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth>
            int8_chwn4_imma_unroll_width;
#endif
#if CUDA_VERSION >= 10020
    std::vector<AlgoInt8NCHW32IMMAImplicitGemm> int8_nchw32_imma;
#endif
    std::vector<std::unique_ptr<AlgoGroupConvGeneral>> gconv_refhold;
    std::vector<std::unique_ptr<AlgoBFloat16>> bfloat16_refhold;
    std::unordered_map<AlgoBase*, AlgoGroupConvGeneral*> algo2gconv;

    AlgoBase* cudnn_conv_bias_act_from_enum(cudnnConvolutionFwdAlgo_t algo);

    AlgoBase* cudnn_conv_from_enum(cudnnConvolutionFwdAlgo_t algo);

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }

private:
#if CUDA_VERSION >= 10000
    void fill_imma_algos();
#endif
    void fill_cudnn_algos();
    void fill_dp4a_algos();
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
