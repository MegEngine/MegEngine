/**
 * \file dnn/src/fallback/batched_matrix_mul/algos.h
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
#include "megdnn/oprs.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/fallback/batched_matrix_mul/opr_impl.h"

#include <memory>
#include <unordered_map>

namespace megdnn {
namespace fallback {

/*!
 * \brief base class for matrix mul algos
 *
 */
class BatchedMatrixMulForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        fallback_BLAS,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::FALLBACK; }

    struct SizeArgs {
        BatchedMatrixMulForwardImpl* opr;
        TensorLayout layout_a, layout_b, layout_c;

        std::string to_string() const;
        SizeArgs(BatchedMatrixMulForwardImpl* opr, const TensorLayout& A,
                 const TensorLayout& B, const TensorLayout& C);
    };
    struct ExecArgs : public SizeArgs {
        TensorND tensor_a, tensor_b, tensor_c;
        Workspace workspace;

        ExecArgs(BatchedMatrixMulForwardImpl* opr, _megdnn_tensor_in A,
                 _megdnn_tensor_in B, _megdnn_tensor_out C,
                 _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs&) const = 0;

    bool is_available_wk(const SizeArgs& args, size_t limit) const {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }
    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT,
            size_t limit = std::numeric_limits<size_t>::max()) const {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) &&
               is_available_wk(args, limit);
    }
    AlgoBase& check_workspace(const SizeArgs& args,
                              const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "matrix mul fwd algo %s: required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }
};

class BatchedMatrixMulForwardImpl::AlgoDefault final : public AlgoBase {
public:
    AlgoDefault() = default;
    bool is_available(const SizeArgs&) const override { return true; }
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override;
    const char* name() const override { return "DEFAULT"; }
    virtual void exec(const ExecArgs&) const override;
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE;
    }
    MEGDNN_DECL_ALGO_TYPE(fallback_BLAS)
};

class BatchedMatrixMulForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoDefault algo_default;
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
