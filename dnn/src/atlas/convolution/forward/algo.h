#pragma once
#include "megdnn/oprs.h"
#include "src/atlas/convolution/opr_impl.h"
#include "src/common/algo_base.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"

#include <unordered_map>

namespace megdnn {
namespace atlas {

/*!
 * \brief base class for convolutionForward algos
 *
 */
class ConvolutionForwardImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    enum class AlgoType : uint32_t {
        ATLAS_DEFAULT,
    };
    using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;

    AlgoBase() : Algorithm() { m_handle_type = Handle::HandleType::ATLAS; }

    struct SizeArgs {
        ConvolutionForwardImpl* opr;
        const TensorLayout *layout_src, *layout_filter, *layout_dst;

        std::string to_string() const;
        SizeArgs(
                ConvolutionForwardImpl* opr, const TensorLayout& src,
                const TensorLayout& filter, const TensorLayout& dst);
    };
    struct ExecArgs : public SizeArgs {
        TensorND tensor_src, tensor_filter, tensor_dst;
        Workspace workspace;

        ExecArgs(
                ConvolutionForwardImpl* opr, _megdnn_tensor_in src,
                _megdnn_tensor_in filter, _megdnn_tensor_out dst,
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
            size_t limit = std::numeric_limits<size_t>::max()) {
        return contain_attribute_all(positive_attr) &&
               !contain_attribute_any(negative_attr) && is_available_wk(args, limit);
    }

    AlgoBase& check_workspace(const SizeArgs& args, const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(
                req <= workspace.size,
                "convolution fwd algo %s: required workspace %zu bytes, "
                "got %zu",
                name(), req, workspace.size);
        return *this;
    }
};

class ConvolutionForwardImpl::AlgoDefault final : public AlgoBase {
public:
    AlgoDefault() = default;
    bool is_available(const SizeArgs&) const override;
    size_t get_workspace_in_bytes(const SizeArgs& /* args */) const override;
    const char* name() const override { return "DEFAULT"; }
    void exec(const ExecArgs&) const override;
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    MEGDNN_DECL_ALGO_TYPE(ATLAS_DEFAULT)
};

class ConvolutionForwardImpl::AlgoPack : NonCopyableObj {
private:
    AlgoBase::Mapper m_all_algos_map;

public:
    AlgoPack();
    AlgoDefault algo_default;
    std::vector<AlgoBase*> all_algos;

    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
