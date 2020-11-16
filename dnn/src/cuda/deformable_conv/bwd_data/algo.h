/**
 * \file dnn/src/cuda/deformable_conv/bwd_data/algo.h
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

#include "src/cuda/deformable_conv/opr_impl.h"

#include <unordered_map>

namespace megdnn {
namespace cuda {

class DeformableConvBackwardDataImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        CUDA_MATMUL,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        DeformableConvBackwardDataImpl* opr;
        HandleImpl* handle;
        const TensorLayout& im_layout;
        CanonizedFilterMeta filter_meta;
        const TensorLayout& offset_layout;
        const TensorLayout& mask_layout;
        const TensorLayout& out_grad_layout;
        const TensorLayout& im_grad_layout;
        const TensorLayout& offset_grad_layout;
        const TensorLayout& mask_grad_layout;

        std::string to_string() const;

        SizeArgs(DeformableConvBackwardDataImpl* opr, const TensorLayout& im,
                 const TensorLayout& filter, const TensorLayout& offset,
                 const TensorLayout& mask, const TensorLayout& out_grad,
                 const TensorLayout& im_grad, const TensorLayout& offset_grad,
                 const TensorLayout& mask_grad);

        SizeArgs(DeformableConvBackwardDataImpl* opr, const TensorLayout& im,
                 const CanonizedFilterMeta& filter, const TensorLayout& offset,
                 const TensorLayout& mask, const TensorLayout& out_grad,
                 const TensorLayout& im_grad, const TensorLayout& offset_grad,
                 const TensorLayout& mask_grad);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND im_tensor, filter_tensor, offset_tensor, mask_tensor,
                out_grad_tensor;
        TensorND im_grad_tensor, offset_grad_tensor, mask_grad_tensor;
        Workspace workspace;

        ExecArgs(DeformableConvBackwardDataImpl* opr, _megdnn_tensor_in im,
                 _megdnn_tensor_in filter, _megdnn_tensor_in offset,
                 _megdnn_tensor_in mask, _megdnn_tensor_in out_grad,
                 _megdnn_tensor_out im_grad, _megdnn_tensor_out offset_grad,
                 _megdnn_tensor_out mask_grad, _megdnn_workspace workspace);
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
        megdnn_assert(
                req <= workspace.size,
                "deformable_conv bwd_data algo %s: required workspace %zu "
                "bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }
};

class DeformableConvBackwardDataImpl::AlgoMatmul final : public AlgoBase {
private:
    static WorkspaceBundle get_bundle(const SizeArgs& args);

    static void get_matmul_layout(const SizeArgs& args, TensorLayout& al,
                                  TensorLayout& bl, TensorLayout& cl);

public:
    AlgoMatmul() {}

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return true; }

    const char* name() const override { return "AlgoMatmul"; }
    MEGDNN_DECL_ALGO_TYPE(CUDA_MATMUL)
};

class DeformableConvBackwardDataImpl::AlgoPack : NonCopyableObj {
    AlgoBase::Mapper m_all_algos_map;
public:
    AlgoPack();
    AlgoMatmul algo_matmul;
    //! all algorithms
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
