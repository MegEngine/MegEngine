#pragma once

#include "src/atlas/convolution/opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

#include <unordered_map>
namespace megdnn {
namespace atlas {

/*!
 * \brief base class for convolution algos
 */
class ConvolutionBackwardDataImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        ATLAS_DEFAULT,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ATLAS; }
    struct SizeArgs {
        CanonizedFilterMeta filter_meta;
        const TensorLayout *diff_layout, *grad_layout, *filter_layout;
        const ConvolutionBackwardDataImpl* opr;

        std::string to_string() const;
        SizeArgs(
                const ConvolutionBackwardDataImpl* opr, const TensorLayout& filter,
                const TensorLayout& diff, const TensorLayout& grad);
        SizeArgs(
                const ConvolutionBackwardDataImpl* opr, const TensorLayout& filter,
                const CanonizedFilterMeta& filter_meta, const TensorLayout& diff,
                const TensorLayout& grad);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *filter_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(
                const ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
                _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                _megdnn_workspace workspace);
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
                "conv bwd data algo %s: "
                "required workspace %zu bytes, got %zu",
                name(), req, workspace.size);
        return *this;
    }
};

class ConvolutionBackwardDataImpl::AlgoDefault final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        return "atlas:ConvolutionBackwardData_Default";
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(ATLAS_DEFAULT)
};

class ConvolutionBackwardDataImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    std::vector<AlgoBase*> all_algos;
    AlgoDefault default_impl;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};
}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
