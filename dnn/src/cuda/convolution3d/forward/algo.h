/**
 * \file dnn/src/cuda/convolution3d/forward/algo.h
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

#include "src/cuda/convolution3d/helper.h"
#include "src/cuda/handle.h"
#include "src/cuda/convolution3d/opr_impl.h"
#include "src/common/utils.h"

#include <unordered_map>

namespace megdnn {
namespace cuda {

/*!
 * \brief base class for convolution3d algos
 *
 * All the algo impls should try to support non-contiguous batch dim, for group
 * conv execution.
 */
class Convolution3DForwardImpl::AlgoBase: public Algorithm {
    protected:
        ~AlgoBase() = default;

    public:
        struct SizeArgs: public convolution3d::ForwardSizeArgs {
            Convolution3DForwardImpl *opr;

            std::string to_string() const;
            void init_desc(convolution3d::CUDNNForwardDescs &desc) const {
                desc.set(*src_layout, filter_meta, *dst_layout, opr->param());
            }
            SizeArgs(Convolution3DForwardImpl *opr,
                    const TensorLayout &src, 
                    const TensorLayout &filter,
                    const TensorLayout &dst);
            SizeArgs(Convolution3DForwardImpl *opr,
                    const TensorLayout &src, 
                    const CanonizedFilterMeta &filter,
                    const TensorLayout &dst);
        };
        struct ExecArgs: public SizeArgs {
            const TensorND *src_tensor, *filter_tensor, *dst_tensor;
            Workspace workspace;

            ExecArgs(Convolution3DForwardImpl *opr,
                    _megdnn_tensor_in src,
                    _megdnn_tensor_in filter,
                    _megdnn_tensor_out dst,
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
        AlgoBase& check_workspace(const SizeArgs& args,
                                  const Workspace& workspace) {
            auto req = get_workspace_in_bytes(args);
            megdnn_assert(req <= workspace.size,
                    "conv3d fwd algo %s: required workspace %zu bytes, got %zu",
                    name(), req, workspace.size);
            return *this;
        }

        virtual bool is_cudnn() const {
            return false;
        }
};
class Convolution3DForwardImpl::Algo1x1x1 final: public AlgoBase {
    static void extract_matmul_layouts(const SizeArgs &args,
            TensorLayout &A, TensorLayout &B, TensorLayout &C);
    public:
        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        const char* name() const override {
            return "1x1x1";
        }
        bool is_reproducible() const override {
            return true;
        }
};

//! implement group conv by another algo
class Convolution3DForwardImpl::AlgoGroupConvGeneral final: public AlgoBase {
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
                TensorLayout &src_pg, TensorLayout &dst_pg);
};

class Convolution3DForwardImpl::AlgoCUDNN final : public AlgoBase {
    bool m_is_reproducible;
    const char *m_name;
    cudnnConvolutionFwdAlgo_t m_cudnn_enum;

    public:

        AlgoCUDNN(bool is_reproducible, const char *name,
                cudnnConvolutionFwdAlgo_t cudnn_enum):
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

        cudnnConvolutionFwdAlgo_t cudnn_enum() const {
            return m_cudnn_enum;
        }

        bool is_cudnn() const override {
            return true;
        }
};

class Convolution3DForwardImpl::AlgoInplaceMatmul final: public AlgoBase {
    public:
        bool is_available(const SizeArgs &args) const override;
        size_t get_workspace_in_bytes(const SizeArgs &args) const override;
        void exec(const ExecArgs &args) const override;

        const char* name() const override {
            return "INPLACE_MATMUL";
        }
        bool is_reproducible() const override {
            return true;
        }
};


class Convolution3DForwardImpl::AlgoChanwise final: public AlgoBase {
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

class Convolution3DForwardImpl::AlgoPack {
    // defined in cudnn.cpp
    void fill_cudnn_algos();

    AlgoPack(const AlgoPack&) = delete;
    AlgoPack& operator = (const AlgoPack &) = delete;

    public:
        AlgoPack();

        std::vector<AlgoCUDNN> cudnn;
        Algo1x1x1 a1x1x1;
        AlgoInplaceMatmul inplace_matmul;
        AlgoChanwise chanwise;
        std::vector<AlgoGroupConvGeneral> gconv;
        std::unordered_map<AlgoBase*, AlgoGroupConvGeneral*> algo2gconv;

        std::vector<AlgoBase*>
            //! all algorithms
            all_algos,
            //! non-cudnn algos, used for heuristic if cudnn is not supported
            non_cudnn_algos;

        AlgoCUDNN* cudnn_from_enum(cudnnConvolutionFwdAlgo_t algo);
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
