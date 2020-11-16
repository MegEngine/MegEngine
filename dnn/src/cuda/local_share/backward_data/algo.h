/**
 * \file dnn/src/cuda/local_share/backward_data/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs.h"

#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/local_share/opr_impl.h"

#include <unordered_map>

namespace megdnn {
namespace cuda {

class LocalShareBackwardDataImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_IMPLICIT_GEMM,
        CUDA_BATCHED_MATMUL,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;


    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        LocalShareBackwardDataImpl* opr;
        TensorLayout filter_layout, diff_layout, grad_layout;

        std::string to_string() const;
        SizeArgs(LocalShareBackwardDataImpl* opr, const TensorLayout& filter,
                 const TensorLayout& diff, const TensorLayout& grad);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *filter_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(LocalShareBackwardDataImpl* opr, _megdnn_tensor_in filter,
                 _megdnn_tensor_in diff, _megdnn_tensor_out grad,
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
                      "local share conv fwd algo %s: required workspace %zu "
                      "bytes, got %zu",
                      name(), req, workspace.size);
        return *this;
    }
};

class LocalShareBackwardDataImpl::AlgoImplicitGemm final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return true; }

    const char* name() const override {
        return "LOCAL_SHARE_IMPLICIT_GEMM";
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_IMPLICIT_GEMM)
};

class LocalShareBackwardDataImpl::AlgoBatchedMatMul final
        : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    WorkspaceBundle get_workspace_bundle(dt_byte* raw_ptr,
                                         const SizeArgs& args) const;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return true; }

    const char* name() const override {
        return "LOCAL_SHARE_BATCHED_MATMUL";
    }
    MEGDNN_DECL_ALGO_TYPE(CUDA_BATCHED_MATMUL)
};

class LocalShareBackwardDataImpl::AlgoPack : NonCopyableObj {
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();

    AlgoImplicitGemm implicit_gemm;
    AlgoBatchedMatMul batched_matmul;

    std::vector<AlgoBase*> all_algos;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
