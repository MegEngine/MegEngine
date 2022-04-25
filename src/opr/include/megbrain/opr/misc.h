#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#if MGB_CUDA
#include "../../../impl/nvof/denseflownvidia.h"
#include "megbrain/opr/param_defs.h"
#endif
#include "megdnn/oprs.h"

#include <array>

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Argmax, intl::MegDNNOprWrapperFwd<megdnn::Argmax>) // {
public:
    MGE_WIN_DECLSPEC_FUC Argmax(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Argmin, intl::MegDNNOprWrapperFwd<megdnn::Argmin>) // {
public:
    MGE_WIN_DECLSPEC_FUC Argmin(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});
};

/*!
 * \brief Argsort operator.
 *
 * Performing m independent argsort operations on m arrays of length n.
 *
 * \param[in] in_tensor \f$(m, n)\f$ input tensor
 * \param[out] out_tensor the first output: \f$(m, n)\f$ sorted output tensor
 * \param[out] indices the second output: \f$(m, n)\f$ sorted indices
 */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ArgsortForward, intl::MegDNNOprWrapperFwd<megdnn::ArgsortForward>) // {
protected:
    NodeProp* do_make_node_prop() const override;
    void scn_do_execute() override;
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;

public:
    MGE_WIN_DECLSPEC_FUC ArgsortForward(
            VarNode* in_tensor, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static std::array<SymbolVar, 2> make(
            SymbolVar in_tensor, const Param& param = {},
            const OperatorNodeConfig& config = {});
};
using Argsort = ArgsortForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ArgsortBackward, intl::MegDNNOprWrapperBwd<megdnn::ArgsortBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC ArgsortBackward(
            VarNode* out_diff, VarNode* indices, VarNode* result_shape,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar out_diff, SymbolVar indices, SymbolVar result_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {});
    static SymbolVar make(
            SymbolVar out_diff, SymbolVar indices, const Param& param = {},
            const OperatorNodeConfig& config = {}) {
        return make(out_diff, indices, out_diff, param, config);
    }
};

//! cumulative product along given axis
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Cumprod,
        cg::SingleCNOperatorNodeBaseT<mixin::MegDNNOprHolderImpl<megdnn::Cumprod>>) // {
    void add_input_layout_constraint() override;

public:
    MGE_WIN_DECLSPEC_FUC Cumprod(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    // for serialization
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar opr, const Param& param, const OperatorNodeConfig& config = {});

protected:
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
};

//! cumulative sum along given axis
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Cumsum,
        cg::SingleCNOperatorNodeBaseT<mixin::MegDNNOprHolderImpl<megdnn::Cumsum>>) // {
    void add_input_layout_constraint() override;

public:
    MGE_WIN_DECLSPEC_FUC Cumsum(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    // for serialization
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar opr, const Param& param, const OperatorNodeConfig& config = {});

protected:
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
};

#if MGB_CUDA
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(NvOf, cg::SingleCNOperatorNodeBase) // {
public:
    using Param = megdnn::param::NvOf;
    MGE_WIN_DECLSPEC_FUC NvOf(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    // for serialization
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar opr, const Param& param, const OperatorNodeConfig& config = {});

    static SymbolVar make(SymbolVar opr, const OperatorNodeConfig& config = {}) {
        return make(opr, {}, config);
    }

    Param param() const { return m_param; }

protected:
    void init_output_dtype() override;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;

private:
    std::shared_ptr<NVFlowExtractor> nv_flow_extractor;
    std::vector<size_t> vshape;
    Param m_param;
    std::mutex m_lock;
    bool init_flag = false;
};
#endif

namespace intl {
using CondTakeBase = cg::SingleCNOperatorNode<
        cg::OperatorNodeBase, mixin::MegDNNOprHolderImpl<megdnn::CondTake>>;
using TopKBase = cg::SingleCNOperatorNode<
        cg::OperatorNodeBase, mixin::MegDNNOprHolderImpl<megdnn::TopK>>;
using CheckNonFiniteBase = cg::SingleCNOperatorNode<
        cg::OperatorNodeBase, mixin::MegDNNOprHolderImpl<megdnn::CheckNonFinite>>;
}  // namespace intl

/*!
 * \brief take values conditionally
 * outputs: values, indices
 */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(CondTake, intl::CondTakeBase) // {
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void add_input_layout_constraint() override;
    NodeProp* do_make_node_prop() const override;

public:
    MGE_WIN_DECLSPEC_FUC CondTake(
            VarNode* data, VarNode* mask, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static std::array<SymbolVar, 2> make(
            SymbolVar data, SymbolVar mask, const Param& param,
            const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(TopK, intl::TopKBase) // {
    void init_output_dtype() override;
    void add_input_layout_constraint() override;
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void record_execute_deps(ExecDependencyArray& deps) override;

public:
    MGE_WIN_DECLSPEC_FUC TopK(
            VarNode* data, VarNode* k, const Param& param,
            const OperatorNodeConfig& config);

    //! note: for KTH_ONLY mode, the second output would be nullptr
    MGE_WIN_DECLSPEC_FUC static std::array<SymbolVar, 2> make(
            SymbolVar data, SymbolVar k, const Param& param,
            const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS(CheckNonFinite, intl::CheckNonFiniteBase) // {
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    void add_input_layout_constraint() override;
    float m_scale = 1;

public:
    MGE_WIN_DECLSPEC_FUC CheckNonFinite(
            const VarNodeArrayView& inp, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            const VarNodeArrayView& inp, const Param& param = {},
            const OperatorNodeConfig& config = {});
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
