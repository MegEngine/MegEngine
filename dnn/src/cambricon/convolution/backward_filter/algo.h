#pragma once

#include "src/cambricon/convolution/opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

#include <unordered_map>

namespace megdnn {
namespace cambricon {

/*!
 * \brief base class for convolution algos
 */
class ConvolutionBackwardFilterImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    struct SizeArgs {
        ConvolutionBackwardFilterImpl* opr;
        const TensorLayout* src_layout;
        const TensorLayout* diff_layout;
        const TensorLayout* grad_layout;
        CanonizedFilterMeta filter_meta;
        std::string to_string() const;
        SizeArgs(
                ConvolutionBackwardFilterImpl* opr, const TensorLayout& src,
                const TensorLayout& diff, const TensorLayout& grad);
        SizeArgs(
                ConvolutionBackwardFilterImpl* opr, const TensorLayout& src,
                const TensorLayout& diff, const TensorLayout& grad,
                const CanonizedFilterMeta& filter_meta);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;
        ExecArgs(
                ConvolutionBackwardFilterImpl* opr, _megdnn_tensor_in src,
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
                "conv fwd algo %s: required workspace %zu bytes, got %zu", name(), req,
                workspace.size);
        return *this;
    }
};

class ConvolutionBackwardFilterImpl::AlgoDefault final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override {
        return "cambricon:ConvolutionBackwardFilter_Default";
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    uint32_t type() const { return 0; }
};

class ConvolutionBackwardFilterImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    std::vector<AlgoBase*> all_algos;
    AlgoDefault default_impl;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
