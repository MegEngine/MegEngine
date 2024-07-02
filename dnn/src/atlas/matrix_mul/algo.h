#pragma once

#include <unordered_map>
#include "src/atlas/handle.h"
#include "src/atlas/matrix_mul/opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

namespace megdnn {
namespace atlas {

class MatrixMulForwardImpl::AlgoBase : public Algorithm {
public:
    enum class AlgoType : uint32_t { ATLAS_ACL };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ATLAS; }
    struct SizeArgs {
        HandleImpl* handle;
        MatrixMulForwardImpl* opr;
        const TensorLayout &layout_a, &layout_b, &layout_c;

        SizeArgs(
                MatrixMulForwardImpl* opr, const TensorLayout& A, const TensorLayout& B,
                const TensorLayout& C);
        std::string to_string() const;
    };
    struct ExecArgs : public SizeArgs {
        const TensorND &tensor_a, &tensor_b, &tensor_c;
        Workspace workspace;

        ExecArgs(
                MatrixMulForwardImpl* opr, _megdnn_tensor_in A, _megdnn_tensor_in B,
                _megdnn_tensor_out C, _megdnn_workspace workspace);
    };

    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_attribute(
            const SizeArgs& args,
            const AlgoAttribute& positive_attr = AlgoAttribute::REPRODUCIBLE,
            const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT,
            size_t limit = std::numeric_limits<size_t>::max()) const {
        MEGDNN_MARK_USED_VAR(limit);
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available(args);
    }

protected:
    ~AlgoBase() = default;
};

class MatrixMulForwardImpl::AlgoACL final : public AlgoBase {
    std::string m_algo_name;

public:
    AlgoACL(const std::string& name) : m_algo_name(name) {}
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override;
    const char* name() const override;
    void exec(const ExecArgs& args) const override;
    MEGDNN_DECL_ALGO_TYPE(ATLAS_ACL)
    AlgoAttribute attribute() const override {
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::USABLE_DEPEND_ON_SHAPE |
               AlgoAttribute::ACCURACY_DEPEND_ON_BATCH;
    }
};

class MatrixMulForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoACL algo_acl{"AclMatrixMulForward"};
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
