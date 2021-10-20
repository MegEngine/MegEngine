/**
 * \file dnn/src/rocm/pooling/algo.h
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
#include "src/rocm/handle.h"
#include "src/rocm/miopen_wrapper.h"
#include "src/rocm/pooling/opr_impl.h"

namespace megdnn {
namespace rocm {

class PoolingForwardImpl::AlgoBase : public Algorithm {
public:
    enum class AlgoType : uint32_t { ROCM_MIOPEN };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ROCM; }
    struct SizeArgs {
        HandleImpl* handle;
        PoolingForwardImpl* opr;
        const TensorLayout *layout_src, *layout_dst;

        std::string to_string() const;
        void init_desc(TensorDesc& src_desc, TensorDesc& dst_desc) const {
            src_desc.set(*layout_src, opr->param().format);
            dst_desc.set(*layout_dst, opr->param().format);
        }

        SizeArgs(
                PoolingForwardImpl* opr, const TensorLayout& src,
                const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *dst_tensor;
        Workspace workspace;

        ExecArgs(
                PoolingForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_out dst,
                _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
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
};

class PoolingForwardImpl::AlgoMIOpen final : public AlgoBase {
    std::string m_algo_name;
    AlgoAttribute m_algo_attribute;

public:
    AlgoMIOpen(AlgoAttribute attr)
            : m_algo_name("MIOpenPoolingForward"), m_algo_attribute(attr) {}

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void init_mode(const ExecArgs& args, miopenPoolingMode_t& mode) const;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_algo_name.c_str(); }
    AlgoAttribute attribute() const override { return m_algo_attribute; }

    MEGDNN_DECL_ALGO_TYPE(ROCM_MIOPEN)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_attribute, ret);
        return ret;
    }
};

class PoolingForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoMIOpen algo_miopen{AlgoAttribute::REPRODUCIBLE};

    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

class PoolingBackwardImpl::AlgoBase : public Algorithm {
public:
    enum class AlgoType : uint32_t { ROCM_MIOPEN };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ROCM; }
    struct SizeArgs {
        HandleImpl* handle;
        PoolingBackwardImpl* opr;
        const TensorLayout *layout_src, *layout_dst, *layout_diff, *layout_grad;

        std::string to_string() const;
        void init_desc(
                TensorDesc& src_desc, TensorDesc& dst_desc, TensorDesc& diff_desc,
                TensorDesc& grad_desc) const {
            src_desc.set(*layout_src);
            dst_desc.set(*layout_dst);
            diff_desc.set(*layout_diff);
            grad_desc.set(*layout_grad);
        }
        SizeArgs(
                PoolingBackwardImpl* opr, const TensorLayout& src,
                const TensorLayout& dst, const TensorLayout& diff,
                const TensorLayout& grad);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *dst_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(
                PoolingBackwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in dst,
                _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
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
};

class PoolingBackwardImpl::AlgoMIOpen final : public AlgoBase {
    std::string m_algo_name;
    AlgoAttribute m_algo_attribute;

public:
    AlgoMIOpen(AlgoAttribute attr)
            : m_algo_name("MIOpenPoolingBackward"), m_algo_attribute(attr) {}

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void init_mode(const ExecArgs& args, miopenPoolingMode_t& mode) const;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return m_algo_name.c_str(); }
    AlgoAttribute attribute() const override { return m_algo_attribute; }

    MEGDNN_DECL_ALGO_TYPE(ROCM_MIOPEN)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_algo_attribute, ret);
        return ret;
    }
};

class PoolingBackwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoMIOpen algo_miopen{AlgoAttribute::REPRODUCIBLE};
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace rocm
}  // namespace megdnn
