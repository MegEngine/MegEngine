#pragma once

#include "opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"

#include <unordered_map>

namespace megdnn {
namespace atlas {

/*!
 * \brief base class for conv_bias algos
 *
 */
class ConvBiasForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        ATLAS_DEFAULT,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ATLAS; }
    struct SizeArgs {
        const ConvBiasForwardImpl* opr;
        const TensorLayout* src_layout;
        const TensorLayout* filter_layout;
        CanonizedFilterMeta filter_meta;
        const TensorLayout* bias_layout;
        const TensorLayout* z_layout;
        const TensorLayout* dst_layout;
        std::string to_string() const;
        SizeArgs(
                ConvBiasForwardImpl* opr, const TensorLayout& src,
                const TensorLayout& filter, const TensorLayout& bias,
                const TensorLayout& z, const TensorLayout& dst);
        SizeArgs(
                ConvBiasForwardImpl* opr, const TensorLayout& src,
                const TensorLayout& filter, const CanonizedFilterMeta& filter_meta,
                const TensorLayout& bias, const TensorLayout& z,
                const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *src_tensor, *filter_tensor, *bias_tensor, *z_tensor,
                *dst_tensor;
        Workspace workspace;

        ExecArgs(
                ConvBiasForwardImpl* opr, _megdnn_tensor_in src,
                _megdnn_tensor_in filter, _megdnn_tensor_in bias, _megdnn_tensor_in z,
                _megdnn_tensor_out dst, _megdnn_workspace workspace);
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
                "conv bias fwd algo %s: required workspace %zu bytes, got %zu", name(),
                req, workspace.size);
        return *this;
    }
};

class ConvBiasForwardImpl::AlgoDefault final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "atlas:ConvBias_Default"; }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_DECL_ALGO_TYPE(ATLAS_DEFAULT)
};

class ConvBiasForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    std::vector<AlgoBase*> all_algos;
    AlgoDefault default_conv;
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
