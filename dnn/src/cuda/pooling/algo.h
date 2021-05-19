/**
 * \file dnn/src/cuda/pooling/algo.h
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

#include <unordered_map>
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/pooling/opr_impl.h"

namespace megdnn {
namespace cuda {

namespace {
#define V1(v) #v
#define V(v) V1(v)
#define DEF_NAME(NAME) \
#NAME "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)
}  // namespace

class PoolingForwardImpl::AlgoBase : public Algorithm {
public:
    enum class AlgoType : uint32_t {
        CUDA_CUDNN,
#if CUDNN_VERSION >= 6000
        CUDA_CUDNN_MAXDETERMINISTIC,
#endif
        CUDA_CHWN4,
        CUDA_NCHW4,
        CUDA_NCHW32,
        CUDA_NHWC,
        CUDA_NCHW64
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        HandleImpl* handle;
        PoolingForwardImpl* opr;
        const TensorLayout *layout_src, *layout_dst;

        std::string to_string() const;
        SizeArgs(PoolingForwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *dst_tensor;
        Workspace workspace;

        ExecArgs(PoolingForwardImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_out dst, _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    size_t get_workspace_in_bytes(const SizeArgs& args) const;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available(args);
    }

protected:
    ~AlgoBase() = default;
    virtual WorkspaceBundle get_workspace_bundle(void* ptr,
                                                 const SizeArgs& args) const;
};

class PoolingForwardImpl::AlgoCUDNN final : public AlgoBase {
    std::string m_algo_name;

public:
    AlgoCUDNN(std::string name) : m_algo_name(name) {}

    bool is_available(const SizeArgs& args) const override;
    void init_mode(const ExecArgs& args, cudnnPoolingMode_t& mode) const;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_algo_name.c_str(); }
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN)

    std::string param() const override { return m_algo_name; }
};

#if CUDNN_VERSION >= 6000
class PoolingForwardImpl::AlgoCUDNNMAXDETERMINISTIC final : public AlgoBase {
    std::string m_algo_name;

public:
    AlgoCUDNNMAXDETERMINISTIC(std::string name) : m_algo_name(name) {}

    bool is_available(const SizeArgs& args) const override;
    void init_mode(const ExecArgs& args, cudnnPoolingMode_t& mode) const;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_algo_name.c_str(); }
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN_MAXDETERMINISTIC)

    std::string param() const override { return m_algo_name; }
};
#endif

#define ALGO_LAYOUT_POOLING_IMPL(_layout)                                 \
    class PoolingForwardImpl::Algo##_layout final : public AlgoBase {     \
        std::string m_algo_name;                                          \
                                                                          \
    public:                                                               \
        Algo##_layout(                                                    \
                std::string name = std::string("CUDA_").append(#_layout)) \
                : m_algo_name(name) {}                                    \
        bool is_available(const SizeArgs& args) const override;           \
        void exec(const ExecArgs& args) const override;                   \
        const char* name() const override { return m_algo_name.c_str(); } \
        AlgoAttribute attribute() const override {                        \
            return AlgoAttribute::REPRODUCIBLE;                           \
        }                                                                 \
        MEGDNN_DECL_ALGO_TYPE(CUDA_##_layout)

ALGO_LAYOUT_POOLING_IMPL(CHWN4)};
ALGO_LAYOUT_POOLING_IMPL(NCHW4)};
ALGO_LAYOUT_POOLING_IMPL(NCHW32)};
ALGO_LAYOUT_POOLING_IMPL(NHWC)};
ALGO_LAYOUT_POOLING_IMPL(NCHW64) //{
protected:
    WorkspaceBundle get_workspace_bundle(void* ptr, const SizeArgs& args)
            const override;

private:
    inline void deduce_reformat_layout(
            std::unique_ptr<RelayoutFormat> & relayout,
            const TensorLayout& src_layout, TensorLayout& dst_layout,
            RelayoutFormat::Param::Mode mode, const int oc, const int group)
            const;
    void get_inner_layout(const TensorLayout& src, const TensorLayout& dst,
                          TensorLayout& inner_src, TensorLayout& inner_dst,
                          Handle* handle,
                          PoolingForwardImpl::Param::Format format) const;
};

#undef ALGO_LAYOUT_POOLING_IMPL

class PoolingForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoCUDNN algo_cudnn{DEF_NAME(cudnnForward)};
#if CUDNN_VERSION >= 6000
    AlgoCUDNNMAXDETERMINISTIC algo_cudnn_max_deterministic{
            DEF_NAME(cudnnForwardMaxDeterministic)};
#endif
    AlgoCHWN4 algo_chwn4;
    AlgoNCHW4 algo_nchw4;
    AlgoNCHW32 algo_nchw32;
    AlgoNHWC algo_nhwc;
    AlgoNCHW64 algo_nchw64;

    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

class PoolingBackwardImpl::AlgoBase : public Algorithm {
public:
    enum class AlgoType : uint32_t { CUDA_CUDNN };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CUDA; }
    struct SizeArgs {
        HandleImpl* handle;
        PoolingBackwardImpl* opr;
        const TensorLayout *layout_src, *layout_dst, *layout_diff, *layout_grad;

        std::string to_string() const;
        SizeArgs(PoolingBackwardImpl* opr, const TensorLayout& src,
                 const TensorLayout& dst, const TensorLayout& diff,
                 const TensorLayout& grad);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *dst_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(PoolingBackwardImpl* opr, _megdnn_tensor_in src,
                 _megdnn_tensor_in dst, _megdnn_tensor_in diff,
                 _megdnn_tensor_out grad, _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    size_t get_workspace_in_bytes(const SizeArgs& args) const;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available(args);
    }

protected:
    ~AlgoBase() = default;
    virtual WorkspaceBundle get_workspace_bundle(void* ptr,
                                                 const SizeArgs& args) const;
};

class PoolingBackwardImpl::AlgoCUDNN final : public AlgoBase {
    std::string m_algo_name;
    bool m_is_reproducible;

public:
    AlgoCUDNN(std::string name, bool is_reproducible)
            : m_algo_name(name), m_is_reproducible(is_reproducible) {}

    bool is_available(const SizeArgs& args) const override;
    void init_mode(const ExecArgs& args, cudnnPoolingMode_t& mode) const;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_algo_name.c_str(); }
    AlgoAttribute attribute() const override {
        auto ret = AlgoAttribute::DEFAULT;
        if (m_is_reproducible) {
            ret |= AlgoAttribute::REPRODUCIBLE;
        }
        return ret;
    }

    MEGDNN_DECL_ALGO_TYPE(CUDA_CUDNN)

    std::string param() const override { return m_algo_name; }
};

class PoolingBackwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    std::vector<AlgoCUDNN> algo_cudnn;
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
