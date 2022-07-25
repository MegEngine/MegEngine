#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/utils/persistent_cache.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {
namespace mixin {

class ConvolutionBackwardDataMixin : public cg::OperatorNodeMixinBase {
protected:
    //! init output desc for conv backward data oprs; it handles both grad
    //! usage and deconv usage
    template <class MgbOpr, class MegDNNOpr>
    static void init_output_static_infer_desc_for_bwd_data(cg::OperatorNodeBase* self);
};

class RegionConvBackwardDataMixin : public cg::OperatorNodeMixinBase {
protected:
    template <typename MGBOPR, typename DNNOPR>
    static void init_output_static_infer_desc_for_bwd_data(cg::OperatorNodeBase* self);
};

class WeightPreprocessExecutor : public cg::OperatorNodeMixinBase {
    class PreprocessedFilterExecDep;

    using PreprocessedFilter = megdnn::detail::PreprocessedFilter;
    std::unique_ptr<PreprocessedFilter> m_preprocessed_filter;
    SmallVector<DeviceTensorND> m_filter_storage;

protected:
    //! this should only be called in scn_do_execute or similar functions (i.e.
    //! post dispatch-to-ExecEnv)
    void mixin_update_preprocessed_filter(OperatorNodeBase& opr);
    void record_preprocessed_weight(cg::GraphExecutable::ExecDependencyArray& deps);
    PreprocessedFilter* preprocessed_filter() const {
        return m_preprocessed_filter.get();
    }

    bool mixin_allow_weight_preprocess(const OperatorNodeBase& opr) const;
    virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout() = 0;
    virtual void scn_do_execute_preprocess() = 0;
    virtual ~WeightPreprocessExecutor() = default;
};

}  // namespace mixin

namespace intl {
//! glue class to apply mixin::WeightPreprocessExecutor
template <
        class Base = cg::OperatorNodeBase,
        class MixinImpl = mixin::WeightPreprocessExecutor>
class OprWithWeightPreprocess : public mixin::CheckBase<Base>::Base, public MixinImpl {
protected:
    using Base::Base;

    void update_preprocessed_filter() { this->mixin_update_preprocessed_filter(*this); }

    bool allow_weight_preprocess() const {
        return this->mixin_allow_weight_preprocess(*this);
    }
};

using ConvBiasBase = cg::SingleCNOperatorNode<
        cg::OutshapePureByInshapeOpr<>,
        mixin::MegDNNOprHolderImpl<megdnn::ConvBiasForward>>;
using ConvBiasForwardBase = OprWithWeightPreprocess<WorkspaceSizeInfer<ConvBiasBase>>;

using DeformableConvBackwardDataT = cg::SingleCNOperatorNode<
        cg::OutshapePureByInshapeOpr<>,
        mixin::MegDNNOprHolderImpl<megdnn::DeformableConvBackwardData>>;
using DeformableConvBackwardDataBase = WorkspaceSizeInfer<DeformableConvBackwardDataT>;

using BatchConvBiasBase = cg::SingleCNOperatorNode<
        cg::OutshapePureByInshapeOpr<>,
        mixin::MegDNNOprHolderImpl<megdnn::BatchConvBiasForward>>;
using BatchConvBiasForwardBase = WorkspaceSizeInfer<BatchConvBiasBase>;

using ConvolutionForwardBase = OprWithWeightPreprocess<WorkspaceSizeInfer<
        typename MegDNNOprWrapperFwdBase<megdnn::ConvolutionForward>::Base>>;
}  // namespace intl

namespace testing {

class ConvolutionTestingPeer;

}  // namespace testing

/* ==================== RegionRestrictedConvolutionForward  ==================== */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        RegionRestrictedConvolutionForward,
        intl::MegDNNOprWrapperFwd<megdnn::RegionRestrictedConvolutionForward>) // {
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void init_output_dtype() override;

public:
    MGE_WIN_DECLSPEC_FUC RegionRestrictedConvolutionForward(
            VarNode* src, VarNode* filter, VarNode* region_in, VarNode* region_out,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar region_in, SymbolVar region_out,
            const Param& param, const OperatorNodeConfig& config = {});
};
using RegionRestrictedConvolution = RegionRestrictedConvolutionForward;

/* ==================== RegionRestrictedConvolutionBackwardData  ==================== */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        RegionRestrictedConvolutionBackwardData,
        cg::SingleCNOperatorNodeBaseT<mixin::MegDNNOprHolderImpl<
                megdnn::RegionRestrictedConvolutionBackwardData>>,
        public mixin::RegionConvBackwardDataMixin) // {
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
    void init_output_dtype() override;

public:
    MGE_WIN_DECLSPEC_FUC RegionRestrictedConvolutionBackwardData(
            VarNode* filter, VarNode* diff, VarNode* region_in, VarNode* region_out,
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    // grad mode
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar diff, SymbolVar region_in, SymbolVar region_out,
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});

    // sereg for deconv mode
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar diff, SymbolVar region_in, SymbolVar region_out,
            const Param& param, const OperatorNodeConfig& config = {});

    // user interface for deconv
    MGE_WIN_DECLSPEC_FUC static SymbolVar make_deconv(
            SymbolVar data, SymbolVar filter, SymbolVar region_in, SymbolVar region_out,
            const Param& param = {}, const OperatorNodeConfig& config = {}) {
        return make(filter, data, region_in, region_out, param, config);
    }
};

/* ==================== RegionRestrictedConvolutionBackwardFilter  ==================== */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        RegionRestrictedConvolutionBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::RegionRestrictedConvolutionBackwardFilter>) // {
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void scn_do_execute() override;

public:
    MGE_WIN_DECLSPEC_FUC RegionRestrictedConvolutionBackwardFilter(
            VarNode* src, VarNode* diff, VarNode* region_in, VarNode* region_out,
            VarNode* filter, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar diff, SymbolVar region_in, SymbolVar region_out,
            SymbolVar filter, const Param& param,
            const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ConvolutionForward, intl::ConvolutionForwardBase,
        public mixin::AlgoChooserHelper) // {
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
    void init_output_format() override;
    void scn_do_execute() override;
    void add_input_layout_constraint() override;
    void init_output_static_infer_desc() override;
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override final;
    void record_execute_deps(cg::GraphExecutable::ExecDependencyArray& deps) override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout() override;
    void scn_do_execute_preprocess() override;

    friend testing::ConvolutionTestingPeer;

public:
    MGE_WIN_DECLSPEC_FUC ConvolutionForward(
            VarNode* src, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});
};
using Convolution = ConvolutionForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ConvBiasForward, intl::ConvBiasForwardBase, public mixin::AlgoChooserHelper) // {
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
    void scn_do_execute() override;
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;

    void init_output_static_infer_desc() override;
    void init_output_format() override;
    void add_input_layout_constraint() override;
    void record_execute_deps(cg::GraphExecutable::ExecDependencyArray& deps) override {
        this->record_megdnn_opr(deps);
        this->record_preprocessed_weight(deps);
    }
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout() override;
    void scn_do_execute_preprocess() override;

public:
    //! src * filter
    MGE_WIN_DECLSPEC_FUC ConvBiasForward(
            VarNode* src, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});

    //! src * filter + bias
    MGE_WIN_DECLSPEC_FUC ConvBiasForward(
            VarNode* src, VarNode* filter, VarNode* bias, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar bias, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});

    //! src * filter + bias + z
    MGE_WIN_DECLSPEC_FUC ConvBiasForward(
            VarNode* src, VarNode* filter, VarNode* bias, VarNode* z,
            const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar bias, SymbolVar z,
            const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    MGE_WIN_DECLSPEC_FUC static void check_winograd_param_valid(
            const megdnn::ConvBias::WinogradParam& param, const DType& dtype);
    MGE_WIN_DECLSPEC_FUC static megdnn::param::MatrixMul::Format get_matmul_format(
            const megdnn::ConvBias::WinogradParam& param);
};
using ConvBias = ConvBiasForward;

/*!
 * \brief Can be used in two ways: compute gradient of conv, or deconv
 */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ConvolutionBackwardData,
        cg::SingleCNOperatorNodeBaseT<
                mixin::MegDNNOprHolderImpl<megdnn::ConvolutionBackwardData>>,
        public mixin::AlgoChooserHelper, public mixin::ConvolutionBackwardDataMixin) // {
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    void init_output_format() override;

    void add_input_layout_constraint() override;

    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;

public:
    MGE_WIN_DECLSPEC_FUC ConvolutionBackwardData(
            VarNode* filter, VarNode* diff, VarNode* src_for_shp, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    //! grad mode; original data shape is required
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar diff, SymbolVar src_for_shp,
            const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    //! sereg for deconvolution mode
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar data, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    //! user interface for deconv
    MGE_WIN_DECLSPEC_FUC static SymbolVar make_deconv(
            SymbolVar data, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {}) {
        return make(filter, data, param, policy, config);
    }
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ConvolutionBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::ConvolutionBackwardFilter>,
        public mixin::AlgoChooserHelper) // {
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    MGE_WIN_DECLSPEC_FUC ConvolutionBackwardFilter(
            VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar diff, SymbolVar filter, const Param& param,
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        MaskConvolution, intl::MegDNNOprWrapperFwd<megdnn::MaskConvolution>) // {
    void init_output_dtype() override final;

public:
    MGE_WIN_DECLSPEC_FUC MaskConvolution(
            VarNode* src, VarNode* filter, VarNode* mask, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar mask, const Param& param,
            const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        MaskPropagate, intl::MegDNNOprWrapperFwd<megdnn::MaskPropagate>) // {
    void init_output_dtype() override final;

public:
    MGE_WIN_DECLSPEC_FUC MaskPropagate(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Convolution3DForward, intl::MegDNNOprWrapperFwd<megdnn::Convolution3DForward>,
        public mixin::AlgoChooserHelper) // {
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    MGE_WIN_DECLSPEC_FUC Convolution3DForward(
            VarNode* src, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});
};
using Convolution3D = Convolution3DForward;

/*!
 * \brief Can be used in two ways: compute gradient of conv, or deconv
 */
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Convolution3DBackwardData,
        cg::SingleCNOperatorNodeBaseT<
                mixin::MegDNNOprHolderImpl<megdnn::Convolution3DBackwardData>>,
        public mixin::AlgoChooserHelper, public mixin::ConvolutionBackwardDataMixin) // {
    void init_output_static_infer_desc() override;

    void add_input_layout_constraint() override;

    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;

public:
    MGE_WIN_DECLSPEC_FUC Convolution3DBackwardData(
            VarNode* filter, VarNode* diff, VarNode* src_for_shp, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    //! grad mode; original data shape is required
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar diff, SymbolVar src_for_shp,
            const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    //! sereg for deconvolution3D mode
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar data, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    //! user interface for deconv
    static SymbolVar make_deconv(
            SymbolVar data, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {}) {
        return make(filter, data, param, policy, config);
    }
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        Convolution3DBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::Convolution3DBackwardFilter>,
        public mixin::AlgoChooserHelper) // {
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    MGE_WIN_DECLSPEC_FUC Convolution3DBackwardFilter(
            VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar diff, SymbolVar filter, const Param& param,
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        LocalShareForward, intl::MegDNNOprWrapperFwd<megdnn::LocalShareForward>,
        public mixin::AlgoChooserHelper) // {
    void init_output_dtype() override;
    void init_output_format() override;

    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    MGE_WIN_DECLSPEC_FUC LocalShareForward(
            VarNode* src, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});
};
using LocalShare = LocalShareForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        LocalShareBackwardData,
        cg::SingleCNOperatorNodeBaseT<
                mixin::MegDNNOprHolderImpl<megdnn::LocalShareBackwardData>>,
        public mixin::AlgoChooserHelper, public mixin::ConvolutionBackwardDataMixin) // {
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;

    void add_input_layout_constraint() override;

    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;

public:
    MGE_WIN_DECLSPEC_FUC LocalShareBackwardData(
            VarNode* filter, VarNode* diff, VarNode* src_for_shp, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    //! grad mode; original data shape is required
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar filter, SymbolVar diff, SymbolVar src_for_shp,
            const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        LocalShareBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::LocalShareBackwardFilter>,
        public mixin::AlgoChooserHelper) // {
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    MGE_WIN_DECLSPEC_FUC LocalShareBackwardFilter(
            VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar diff, SymbolVar filter, const Param& param,
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        DeformableConvForward, intl::MegDNNOprWrapperFwd<megdnn::DeformableConvForward>,
        public mixin::AlgoChooserHelper) // {
public:
    MGE_WIN_DECLSPEC_FUC DeformableConvForward(
            VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
            const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar offset, SymbolVar mask,
            const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

private:
    void init_output_dtype() override;
    void init_output_format() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
};
using DeformableConv = DeformableConvForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        DeformableConvBackwardData, intl::DeformableConvBackwardDataBase,
        public mixin::AlgoChooserHelper, public mixin::ConvolutionBackwardDataMixin) // {
public:
    MGE_WIN_DECLSPEC_FUC DeformableConvBackwardData(
            VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
            VarNode* diff, const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make_all(
            SymbolVar src, SymbolVar filter, SymbolVar offset, SymbolVar mask,
            SymbolVar diff, const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar offset, SymbolVar mask,
            SymbolVar diff, const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    void scn_do_execute() override;

private:
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray&, const TensorShapeArray&) const override;
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    void init_output_format() override;

    NodeProp* do_make_node_prop() const override;

    void add_input_layout_constraint() override {
        mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
    }
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        DeformableConvBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::DeformableConvBackwardFilter>,
        public mixin::AlgoChooserHelper) // {
public:
    MGE_WIN_DECLSPEC_FUC DeformableConvBackwardFilter(
            VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
            VarNode* diff, const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar offset, SymbolVar mask,
            SymbolVar diff, const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    void scn_do_execute() override;

private:
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        BatchConvBiasForward, intl::BatchConvBiasForwardBase,
        public mixin::AlgoChooserHelper) // {
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
    void scn_do_execute() override;
    void get_output_var_shape(
            const TensorShapeArray& input_shapes,
            TensorShapeArray& output_shapes) const override;
    void init_output_static_infer_desc() override;
    void init_output_format() override;
    void add_input_layout_constraint() override;
    void record_execute_deps(cg::GraphExecutable::ExecDependencyArray& deps) override {
        this->record_megdnn_opr(deps);
    }

public:
    //! src * filter
    MGE_WIN_DECLSPEC_FUC BatchConvBiasForward(
            VarNode* src, VarNode* filter, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});

    //! src * filter + bias
    MGE_WIN_DECLSPEC_FUC BatchConvBiasForward(
            VarNode* src, VarNode* filter, VarNode* bias, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar bias, const Param& param = {},
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});

    //! src * filter + bias + z
    MGE_WIN_DECLSPEC_FUC BatchConvBiasForward(
            VarNode* src, VarNode* filter, VarNode* bias, VarNode* z,
            const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar filter, SymbolVar bias, SymbolVar z,
            const Param& param = {}, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});
};
using BatchConvBias = BatchConvBiasForward;

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
