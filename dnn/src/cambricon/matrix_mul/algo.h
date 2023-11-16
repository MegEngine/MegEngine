#pragma once

#include "src/cambricon/matrix_mul/opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

#include <unordered_map>

namespace megdnn {
namespace cambricon {

/*!
 * \brief base class for matrix mul algos
 *
 */
class MatrixMulForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::CAMBRICON; }
    struct SizeArgs {
        MatrixMulForwardImpl* opr;
        TensorLayout layout_a, layout_b, layout_c;

        std::string to_string() const;
        SizeArgs(
                MatrixMulForwardImpl* opr, const TensorLayout& A, const TensorLayout& B,
                const TensorLayout& C);
    };
    struct ExecArgs : public SizeArgs {
        TensorND tensor_a, tensor_b, tensor_c;
        Workspace workspace;

        ExecArgs(
                MatrixMulForwardImpl* opr, _megdnn_tensor_in A, _megdnn_tensor_in B,
                _megdnn_tensor_out C, _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_wk(const SizeArgs& args, size_t limit) {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }

    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT,
            size_t limit = std::numeric_limits<size_t>::max()) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available_wk(args, limit);
    }

    AlgoBase& check_workspace(const SizeArgs& args, const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "matrixmul fwd algo %s: required workspace %zu bytes, got %zu", name(),
                req, workspace.size);
        return *this;
    }
};

class MatrixMulForwardImpl::AlgoDefault final : public MatrixMulForwardImpl::AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override { return 0; }
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "cambricon:MatrixMul_Default"; }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    uint32_t type() const { return 0; }
};

class MatrixMulForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    std::vector<AlgoBase*> all_algos;
    AlgoDefault default_matmul;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
