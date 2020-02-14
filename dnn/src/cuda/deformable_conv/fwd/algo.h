/**
 * \file dnn/src/cuda/deformable_conv/fwd/algo.h
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

#include "src/cuda/deformable_conv/opr_impl.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class DeformableConvForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    struct SizeArgs {
        DeformableConvForwardImpl* opr;
        HandleImpl* handle;
        const TensorLayout& im_layout;
        CanonizedFilterMeta filter_meta;
        const TensorLayout& offset_layout;
        const TensorLayout& mask_layout;
        const TensorLayout& dst_layout;

        std::string to_string() const;
        SizeArgs(DeformableConvForwardImpl* opr, const TensorLayout& im,
                 const TensorLayout& filter, const TensorLayout& offset,
                 const TensorLayout& mask, const TensorLayout& dst);
        SizeArgs(DeformableConvForwardImpl* opr, const TensorLayout& im,
                 const CanonizedFilterMeta& filter, const TensorLayout& offset,
                 const TensorLayout& mask, const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND &im_tensor, filter_tensor, offset_tensor, mask_tensor,
                dst_tensor;
        Workspace workspace;

        ExecArgs(DeformableConvForwardImpl* opr, _megdnn_tensor_in im,
                 _megdnn_tensor_in filter, _megdnn_tensor_in offset,
                 _megdnn_tensor_in mask, _megdnn_tensor_out dst,
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
                      "deformable_conv fwd algo %s: required workspace %zu "
                      "bytes, got %zu",
                      name(), req, workspace.size);
        return *this;
    }
};

class DeformableConvForwardImpl::AlgoMatmul final : public AlgoBase {
private:
    static void get_matmul_layout(const SizeArgs& args, TensorLayout& al,
                                  TensorLayout& bl, TensorLayout& cl);
    static WorkspaceBundle get_bundle(const SizeArgs& args);

public:
    AlgoMatmul(){};

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return true; }

    const char* name() const override { return "AlgoMatmul"; }
};

class DeformableConvForwardImpl::AlgoPack {
    AlgoPack(const AlgoPack&) = delete;
    AlgoPack& operator=(const AlgoPack&) = delete;

public:
    AlgoPack();
    AlgoMatmul algo_matmul;
    //! all algorithms
    std::vector<AlgoBase*> all_algos;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
