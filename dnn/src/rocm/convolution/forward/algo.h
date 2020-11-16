/**
 * \file dnn/src/rocm/convolution/forward/algo.h
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
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/rocm/convolution/helper.h"
#include "src/rocm/convolution/opr_impl.h"
#include "src/rocm/handle.h"

#include <unordered_map>

namespace megdnn {
namespace rocm {

/*!
 * \brief base class for convolution algos
 *
 */
class ConvolutionForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        ROCM_MIOPEN,
        ROCM_MATMUL,
        ROCM_INPLACE_MATMUL,
        ROCM_1X1,
        ROCM_1X1_LARGE_BATCH,
        ROCM_CHANWISE
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ROCM; }
    struct SizeArgs : public convolution::ForwardSizeArgs {
        ConvolutionForwardImpl* opr;

        std::string to_string() const;
        convolution::MIOpenCacheKey to_miopen_algo_cache_key() const;
        void init_desc(convolution::MIOpenForwardDescs& desc) const {
            desc.set(*src_layout, filter_meta, *dst_layout, opr->param());
        }
        SizeArgs(ConvolutionForwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& filter, const TensorLayout& dst);
        SizeArgs(ConvolutionForwardImpl* opr, const TensorLayout& src,
                 const CanonizedFilterMeta& filter, const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *filter_tensor, *dst_tensor;
        Workspace workspace;

        ExecArgs(ConvolutionForwardImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_in filter, _megdnn_tensor_out dst,
                 _megdnn_workspace workspace);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

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
        megdnn_assert(req <= workspace.size,
                      "conv fwd algo %s: required workspace %zu bytes, got %zu",
                      name(), req, workspace.size);
        return *this;
    }

    virtual bool is_miopen() const { return false; }
};

class ConvolutionForwardImpl::AlgoMIOpen final : public AlgoBase {
    bool m_is_reproducible;
    const char* m_name;

    miopenConvFwdAlgorithm_t find_best_algo(const ExecArgs& args);

public:
    AlgoMIOpen() = delete;
    AlgoMIOpen(bool is_reproducible) : m_is_reproducible(is_reproducible) {}

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return m_is_reproducible; }

    const char* name() const override { return "MIOpenConvolutionForward"; }

    bool is_miopen() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(ROCM_MIOPEN)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_is_reproducible, ret);
        return ret;
    }

    static convolution::MIOpenCache<SizeArgs, miopenConvFwdAlgorithm_t>
            sm_miopen_algo_cache;
    static convolution::MIOpenCache<SizeArgs, size_t> sm_miopen_ws_cache;
};

class ConvolutionForwardImpl::AlgoMatmul final : public AlgoBase {
    template <typename T>
    static void exec_internal(const ExecArgs& args);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "MATMUL"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(ROCM_MATMUL)
};

//! compute small matmul in the kernel
class ConvolutionForwardImpl::AlgoInplaceMatmul final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "INPLACE_MATMUL"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(ROCM_INPLACE_MATMUL)
};

//! optimized 1x1 conv
class ConvolutionForwardImpl::Algo1x1 final : public AlgoBase {
    static void extract_matmul_layouts(const SizeArgs& args, TensorLayout& A,
                                       TensorLayout& B, TensorLayout& C);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "1x1"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(ROCM_1X1)
};

//! optimized 1x1 conv when input data batchsize is larger than 32
class ConvolutionForwardImpl::Algo1x1LargeBatch final : public AlgoBase {
    static void extract_matmul_layouts(const SizeArgs& args, TensorLayout& A,
                                       TensorLayout& B, TensorLayout& C);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "LARGE_BATCH_1x1"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(ROCM_1X1_LARGE_BATCH)
};

class ConvolutionForwardImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE"; }
    bool is_reproducible() const override { return true; }
    MEGDNN_DECL_ALGO_TYPE(ROCM_CHANWISE)
};

class ConvolutionForwardImpl::AlgoPack : NonCopyableObj {
    // defined in miopen.cpp
    void fill_miopen_algos();

    AlgoBase::Mapper m_all_algos_map;
public:
    AlgoPack();

    AlgoMIOpen miopen{true};
    AlgoMatmul matmul;
    AlgoInplaceMatmul inplace_matmul;
    Algo1x1 a1x1;
    Algo1x1LargeBatch batched_matrix_mul;
    AlgoChanwise chanwise;

    std::vector<AlgoBase*>
            //! all algorithms
            all_algos, miopen_algos, non_miopen_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
