/**
 * \file dnn/src/cuda/convolution/backward_data/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/cuda/convolution/helper.h"
#include <unordered_map>

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for convolution algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class ConvolutionBackwardDataImpl::AlgoBase: public Algorithm {
    protected:
        ~AlgoBase() = default;

    public:
        struct SizeArgs {
            HandleImpl *handle;
            CanonizedFilterMeta filter_meta;
            const TensorLayout *diff_layout, *grad_layout;
            ConvolutionBackwardDataImpl *opr;

            std::string to_string() const;
            void init_desc(convolution::CUDNNBwdDataDescs &desc) const {
                desc.set(filter_meta, *diff_layout, *grad_layout, opr->param());
            }
            SizeArgs(ConvolutionBackwardDataImpl *opr,
                    const TensorLayout &filter, const TensorLayout &diff,
                    const TensorLayout &grad);
            SizeArgs(ConvolutionBackwardDataImpl *opr,
                    const CanonizedFilterMeta &filter, const TensorLayout &diff,
                    const TensorLayout &grad);

            convolution::ForwardSizeArgs as_fwd_args() const {
                return {handle, grad_layout, filter_meta, diff_layout};
            }
        };
        struct ExecArgs: public SizeArgs {
            const TensorND *filter_tensor, *diff_tensor, *grad_tensor;
            Workspace workspace;

            ExecArgs(ConvolutionBackwardDataImpl *opr,
                    _megdnn_tensor_in filter,
                    _megdnn_tensor_in diff,
                    _megdnn_tensor_out grad,
                    _megdnn_workspace workspace);
        };
        virtual bool is_available(const SizeArgs &args) const = 0;
        virtual size_t get_workspace_in_bytes(const SizeArgs &args) const = 0;
        virtual void exec(const ExecArgs &args) const = 0;

        bool is_available_wk(const SizeArgs &args, size_t limit) {
            return is_available(args) && get_workspace_in_bytes(args) <= limit;
        }

        bool is_available_reproducible(
                const SizeArgs& args, bool reproducible = true,
                size_t limit = std::numeric_limits<size_t>::max()) {
            return (!reproducible || is_reproducible()) &&
                   is_available_wk(args, limit);
        }

        AlgoBase& check_workspace(
                const SizeArgs &args, const Workspace &workspace) {
            auto req = get_workspace_in_bytes(args);
            megdnn_assert(req <= workspace.size,
                    "conv bwd data algo %s: "
                    "required workspace %zu bytes, got %zu",
                    name(), req, workspace.size);
            return *this;
        }

        virtual bool is_cudnn() const {
            return false;
        }
};

class ConvolutionBackwardDataImpl::AlgoCUDNN final : public AlgoBase {
    bool m_is_reproducible;
    const char *m_name;
    cudnnConvolutionBwdDataAlgo_t m_cudnn_enum;

    public:

        AlgoCUDNN(bool is_reproducible, const char *name,
                cudnnConvolutionBwdDataAlgo_t cudnn_enum):
            m_is_reproducible(is_reproducible),
            m_name(name),
            m_cudnn_enum(cudnn_enum)
        {}

        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        bool is_reproducible() const override {
            return m_is_reproducible;
        }

        const char* name() const override {
            return m_name;
        }

        cudnnConvolutionBwdDataAlgo_t cudnn_enum() const {
            return m_cudnn_enum;
        }

        bool is_cudnn() const override {
            return true;
        }
};

//! im2col and matmul, with dilation
class ConvolutionBackwardDataImpl::AlgoMatmul final: public AlgoBase {
    template<typename T>
    static void exec_internal(const ExecArgs &args);

    public:
        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        const char* name() const override {
            return "MATMUL";
        }
        bool is_reproducible() const override {
            return true;
        }
};

class ConvolutionBackwardDataImpl::AlgoChanwise final: public AlgoBase {
    public:
        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        const char* name() const override {
            return "CHANNEL_WISE";
        }
        bool is_reproducible() const override {
            return true;
        }
};

class ConvolutionBackwardDataImpl::AlgoChanwiseSmall final: public AlgoBase {
    public:
        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        const char* name() const override {
            return "CHANNEL_WISE_SMALL";
        }
        bool is_reproducible() const override {
            return true;
        }
};

//! implement group conv by another algo
class ConvolutionBackwardDataImpl::AlgoGroupConvGeneral final: public AlgoBase {
    AlgoBase *m_impl;
    std::string m_name;

    public:
        AlgoGroupConvGeneral(AlgoBase *impl);

        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        const char* name() const override {
            return m_name.c_str();
        }

        bool is_reproducible() const override {
            return m_impl->is_reproducible();
        }

        static void modify_size_args(SizeArgs &args,
                TensorLayout &diff_pg, TensorLayout &grad_pg);
};

class ConvolutionBackwardDataImpl::AlgoPack {
    // defined in cudnn.cpp
    void fill_cudnn_algos();

    AlgoPack(const AlgoPack&) = delete;
    AlgoPack& operator = (const AlgoPack &) = delete;

    public:
        AlgoPack();

        std::vector<AlgoCUDNN> cudnn;
        AlgoMatmul matmul;
        AlgoChanwise chanwise;
        AlgoChanwiseSmall chanwise_small;
        std::vector<AlgoGroupConvGeneral> gconv;
        std::unordered_map<AlgoBase*, AlgoGroupConvGeneral*> algo2gconv;

        std::vector<AlgoBase*>
            //! all algorithms
            all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos;

        AlgoCUDNN* cudnn_from_enum(cudnnConvolutionBwdDataAlgo_t algo);
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
