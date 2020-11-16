/**
 * \file dnn/src/cuda/batch_conv_bias/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <unordered_map>
#include "megdnn/oprs.h"

#include "src/common/utils.h"
#include "src/cuda/batch_conv_bias/opr_impl.h"
#include "src/cuda/handle.h"

#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

namespace megdnn {
namespace cuda {

class BatchConvBiasForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_GEMM_NCHW4_DOTPROD_INT8,
        CUDA_IMPLICIT_GEMM_PRECOMP_NCHW4_DOTPROD_INT8,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        BatchConvBiasForwardImpl* opr;
        TensorLayout src_layout, filter_layout, bias_layout, z_layout,
                dst_layout;

        std::string to_string() const;
        SizeArgs(BatchConvBiasForwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& filter, const TensorLayout& bias,
                 const TensorLayout& z, const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *filter_tensor, *bias_tensor, *z_tensor,
                *dst_tensor;
        Workspace workspace;

        ExecArgs(BatchConvBiasForwardImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_in filter, _megdnn_tensor_in bias,
                 _megdnn_tensor_in z, _megdnn_tensor_out dst,
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
                      "batch conv bias fwd algo %s: required workspace %zu "
                      "bytes, got %zu",
                      name(), req, workspace.size);
        return *this;
    }
};

class BatchConvBiasForwardImpl::AlgoInt8NCHW4DotProdGemm final
        : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return true; }

    const char* name() const override {
        return "BATCH_CONV_BIAS_INT8_NCHW4_GEMM_DOTPROD";
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_GEMM_NCHW4_DOTPROD_INT8)
};

class BatchConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemmPrecomp final
        : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return true; }

    const char* name() const override {
        return "BATCH_CONV_BIAS_INT8_NCHW4_IMPLICIT_GEMM_PRECOMP_DOTPROD";
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM_PRECOMP_NCHW4_DOTPROD_INT8)

private:
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
};

class BatchConvBiasForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    AlgoInt8NCHW4DotProdGemm int8_nchw4_gemm_dotprod;
    AlgoInt8NCHW4DotProdImplicitGemmPrecomp int8_nchw4_implicit_gemm_dotprod;

    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
