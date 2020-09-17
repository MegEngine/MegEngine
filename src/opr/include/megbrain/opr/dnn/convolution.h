/**
 * \file src/opr/include/megbrain/opr/dnn/convolution.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/opr/param_defs.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {
namespace mixin {

/*!
 * \brief Convolution base class
 */
class Convolution {
    public:
        using ExecutionPolicy = megdnn::param::ExecutionPolicy;

        const ExecutionPolicy& execution_policy() const {
            if (!m_policy_accessed) {
                m_policy_accessed = true;
            }
            return m_policy;
        }

        /*!
         * \brief get current policy without marking it as having been accessed
         *
         * This is primarily used for getting current policy before calling
         * set_execution_policy().
         */
        const ExecutionPolicy& execution_policy_transient() const {
            return m_policy;
        }

        /*!
         * \brief modify execution policy
         *
         * Exception would be thrown if execution_policy() has been accessed,
         * since it would influence cache and many other decisions.
         */
        void set_execution_policy(const ExecutionPolicy& policy);

        AlgoChooserProfileCache& profile_cache() const;

        virtual std::pair<const void*, size_t> param_blob() const = 0;

    protected:
        ~Convolution();

        mutable bool m_policy_accessed = false;
        ExecutionPolicy m_policy;

        std::unique_ptr<AlgoChooserProfileCache> m_profile_cache;

        virtual void init_profile_cache() = 0;

        //! init output desc for conv backward data oprs; it handles both grad
        //! usage and deconv usage
        template <class MgbOpr, class MegDNNOpr>
        static void init_output_static_infer_desc_for_bwd_data(
                cg::OperatorNodeBase* self);
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
    void record_preprocessed_weight(
            cg::GraphExecutable::ExecDependencyArray& deps);
    PreprocessedFilter* preprocessed_filter() const {
        return m_preprocessed_filter.get();
    }

    bool mixin_allow_weight_preprocess(const OperatorNodeBase& opr) const;
    virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout() = 0;
    virtual void scn_do_execute_preprocess() = 0;
    virtual ~WeightPreprocessExecutor() = default;
};

} // namespace mixin

namespace intl {
    //! glue class to apply mixin::WeightPreprocessExecutor
    template<class Base = cg::OperatorNodeBase,
             class MixinImpl = mixin::WeightPreprocessExecutor>
    class OprWithWeightPreprocess: public mixin::CheckBase<Base>::Base,
                                   public MixinImpl {
    protected:
        using Base::Base;

        void update_preprocessed_filter() {
            this->mixin_update_preprocessed_filter(*this);
        }

        bool allow_weight_preprocess() const {
            return this->mixin_allow_weight_preprocess(*this);
        }
    };

    using ConvBiasBase = cg::SingleCNOperatorNode<
            cg::OutshapePureByInshapeOpr<>,
            mixin::MegDNNOprHolderImpl<megdnn::ConvBiasForward>>;
    using ConvBiasForwardBase =
            OprWithWeightPreprocess<WorkspaceSizeInfer<ConvBiasBase>>;

    using DeformableConvBackwardDataT = cg::SingleCNOperatorNode<
            cg::OutshapePureByInshapeOpr<>,
            mixin::MegDNNOprHolderImpl<megdnn::DeformableConvBackwardData>>;
    using DeformableConvBackwardDataBase = WorkspaceSizeInfer<DeformableConvBackwardDataT>;

    using BatchConvBiasBase = cg::SingleCNOperatorNode<
            cg::OutshapePureByInshapeOpr<>,
            mixin::MegDNNOprHolderImpl<megdnn::BatchConvBiasForward>>;
    using BatchConvBiasForwardBase = WorkspaceSizeInfer<BatchConvBiasBase>;

    using ConvolutionForwardBase = OprWithWeightPreprocess<
            WorkspaceSizeInfer<typename MegDNNOprWrapperFwdBase<
                    megdnn::ConvolutionForward>::Base>>;
}  // namespace intl

namespace testing {

class ConvolutionTestingPeer;

}  // namespace testing

MGB_DEFINE_OPR_CLASS(ConvolutionForward,
        intl::ConvolutionForwardBase, public mixin::Convolution) // {

    void init_profile_cache() override;
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const override final;
    void init_output_format() override;
    void scn_do_execute() override;
    void add_input_layout_constraint() override;
    void init_output_static_infer_desc() override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override final;
    void record_execute_deps(
            cg::GraphExecutable::ExecDependencyArray& deps) override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout() override;
    void scn_do_execute_preprocess() override;

    friend testing::ConvolutionTestingPeer;

    public:
        ConvolutionForward(VarNode *src, VarNode *filter,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar src, SymbolVar filter,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        std::pair<const void*, size_t> param_blob() const override;
};
using Convolution = ConvolutionForward;

MGB_DEFINE_OPR_CLASS(ConvBiasForward, intl::ConvBiasForwardBase,
        public mixin::Convolution) // {

    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const override final;
    void scn_do_execute() override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;

    void init_output_static_infer_desc() override;
    void init_output_format() override;
    void add_input_layout_constraint() override;
    void record_execute_deps(
            cg::GraphExecutable::ExecDependencyArray& deps) override {
        this->record_megdnn_opr(deps);
        this->record_preprocessed_weight(deps);
    }
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout() override;
    void scn_do_execute_preprocess() override;

public:
    //! src * filter
    ConvBiasForward(VarNode* src, VarNode* filter, const Param& param,
                    const ExecutionPolicy& policy,
                    const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter,
                          const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    //! src * filter + bias
    ConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                    const Param& param, const ExecutionPolicy& policy,
                    const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                          const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    //! src * filter + bias + z
    ConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias, VarNode* z,
                    const Param& param, const ExecutionPolicy& policy,
                    const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                          SymbolVar z, const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    void init_profile_cache() override;
    std::pair<const void*, size_t> param_blob() const override;

    static void check_winograd_param_valid(
            const megdnn::ConvBias::WinogradParam& param,
            const DType& dtype);
    static megdnn::param::MatrixMul::Format get_matmul_format(
            const megdnn::ConvBias::WinogradParam& param);
};
using ConvBias = ConvBiasForward;

/*!
 * \brief Can be used in two ways: compute gradient of conv, or deconv
 */
MGB_DEFINE_OPR_CLASS(ConvolutionBackwardData,
        cg::SingleCNOperatorNodeBaseT<
            mixin::MegDNNOprHolderImpl<megdnn::ConvolutionBackwardData>>,
        public mixin::Convolution) // {
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    void init_output_format() override;

    void add_input_layout_constraint() override;
    void init_profile_cache() override;

    void scn_do_execute() override;
    NodeProp *do_make_node_prop() const override;

    public:
        ConvolutionBackwardData(
                VarNode *filter, VarNode *diff, VarNode *src_for_shp,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);

        //! grad mode; original data shape is required
        static SymbolVar make(
                SymbolVar filter, SymbolVar diff, SymbolVar src_for_shp,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        //! sereg for deconvolution mode
        static SymbolVar make(
                SymbolVar filter, SymbolVar data,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);

        //! user interface for deconv
        static SymbolVar make_deconv(
                SymbolVar data, SymbolVar filter,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {}) {
            return make(filter, data, param, policy, config);
        }

        std::pair<const void*, size_t> param_blob() const override;
};

MGB_DEFINE_OPR_CLASS(ConvolutionBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::ConvolutionBackwardFilter>,
        public mixin::Convolution ) // {

    void init_profile_cache() override final;

    size_t get_workspace_size_bytes(
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const override final;

    public:
        ConvolutionBackwardFilter(VarNode *src, VarNode *diff, VarNode *filter,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);
        static SymbolVar make(SymbolVar src, SymbolVar diff, SymbolVar filter,
                const Param &param,
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        std::pair<const void*, size_t> param_blob() const override;
};

MGB_DEFINE_OPR_CLASS(MaskConvolution,
        intl::MegDNNOprWrapperFwd<megdnn::MaskConvolution>) // {

    void init_output_dtype() override final;

public:
    MaskConvolution(VarNode* src, VarNode* filter, VarNode* mask,
            const Param& param,
            const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar mask,
            const Param& param, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS(MaskPropagate,
                     intl::MegDNNOprWrapperFwd<megdnn::MaskPropagate>)  // {

    void init_output_dtype() override final;

public:
    MaskPropagate(VarNode* src, const Param& param,
                  const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, const Param& param,
                          const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS(Convolution3DForward,
        intl::MegDNNOprWrapperFwd<megdnn::Convolution3DForward>,
        public mixin::Convolution) // {

    void init_profile_cache() override;
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const override final;

    public:
        Convolution3DForward(VarNode *src, VarNode *filter,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar src, SymbolVar filter,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        std::pair<const void*, size_t> param_blob() const override;
};
using Convolution3D = Convolution3DForward;

/*!
 * \brief Can be used in two ways: compute gradient of conv, or deconv
 */
MGB_DEFINE_OPR_CLASS(Convolution3DBackwardData,
        cg::SingleCNOperatorNodeBaseT<
            mixin::MegDNNOprHolderImpl<megdnn::Convolution3DBackwardData>>,
        public mixin::Convolution) // {
    void init_output_static_infer_desc() override;

    void add_input_layout_constraint() override;
    void init_profile_cache() override;

    void scn_do_execute() override;
    NodeProp *do_make_node_prop() const override;

    public:
        Convolution3DBackwardData(
                VarNode *filter, VarNode *diff, VarNode *src_for_shp,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);

        //! grad mode; original data shape is required
        static SymbolVar make(
                SymbolVar filter, SymbolVar diff, SymbolVar src_for_shp,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        //! sereg for deconvolution3D mode
        static SymbolVar make(
                SymbolVar filter, SymbolVar data,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);

        //! user interface for deconv
        static SymbolVar make_deconv(
                SymbolVar data, SymbolVar filter,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {}) {
            return make(filter, data, param, policy, config);
        }

        std::pair<const void*, size_t> param_blob() const override;
};

MGB_DEFINE_OPR_CLASS(Convolution3DBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::Convolution3DBackwardFilter>,
        public mixin::Convolution) // {

    void init_profile_cache() override final;

    size_t get_workspace_size_bytes(
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const override final;

    public:
        Convolution3DBackwardFilter(VarNode *src, VarNode *diff, VarNode *filter,
                const Param &param,
                const ExecutionPolicy &policy,
                const OperatorNodeConfig &config);
        static SymbolVar make(SymbolVar src, SymbolVar diff, SymbolVar filter,
                const Param &param,
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        std::pair<const void*, size_t> param_blob() const override;
};

MGB_DEFINE_OPR_CLASS(LocalShareForward,
                     intl::MegDNNOprWrapperFwd<megdnn::LocalShareForward>,
                     public mixin::Convolution)  // {
    void init_profile_cache() override final;
    void init_output_dtype() override;
    void init_output_format() override;

    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    LocalShareForward(VarNode* src, VarNode* filter, const Param& param,
                      const ExecutionPolicy& policy,
                      const OperatorNodeConfig& config);
    static SymbolVar make(SymbolVar src, SymbolVar filter, const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});
    std::pair<const void*, size_t> param_blob() const override;
};
using LocalShare = LocalShareForward;

MGB_DEFINE_OPR_CLASS(
        LocalShareBackwardData,
        cg::SingleCNOperatorNodeBaseT<
                mixin::MegDNNOprHolderImpl<megdnn::LocalShareBackwardData>>,
        public mixin::Convolution) // {
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;

    void add_input_layout_constraint() override;
    void init_profile_cache() override;

    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;

public:
    LocalShareBackwardData(VarNode* filter, VarNode* diff, VarNode* src_for_shp,
                           const Param& param, const ExecutionPolicy& policy,
                           const OperatorNodeConfig& config);

    //! grad mode; original data shape is required
    static SymbolVar make(SymbolVar filter, SymbolVar diff,
                          SymbolVar src_for_shp, const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    std::pair<const void*, size_t> param_blob() const override;
};

MGB_DEFINE_OPR_CLASS(
        LocalShareBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::LocalShareBackwardFilter>,
        public mixin::Convolution) // {
    void init_profile_cache() override final;

    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;

public:
    LocalShareBackwardFilter(VarNode* src, VarNode* diff, VarNode* filter,
                             const Param& param, const ExecutionPolicy& policy,
                             const OperatorNodeConfig& config);
    static SymbolVar make(SymbolVar src, SymbolVar diff, SymbolVar filter,
                          const Param& param,
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    std::pair<const void*, size_t> param_blob() const override;
};

MGB_DEFINE_OPR_CLASS(DeformableConvForward,
        intl::MegDNNOprWrapperFwd<megdnn::DeformableConvForward>,
        public mixin::Convolution) // {
    public:
        DeformableConvForward(
                VarNode *src, VarNode *filter, VarNode *offset, VarNode *mask,
                const Param &param,
                const ExecutionPolicy& policy,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar offset,
                SymbolVar mask,
                const Param &param = {},
                const ExecutionPolicy &policy = {},
                const OperatorNodeConfig &config = {});

        std::pair<const void*, size_t> param_blob() const override;
    private:
        void init_profile_cache() override;
        void init_output_dtype() override;
        void init_output_format() override;
        size_t get_workspace_size_bytes(
                const TensorShapeArray &input_shapes,
                const TensorShapeArray &output_shapes) const override final;
};
using DeformableConv = DeformableConvForward;

MGB_DEFINE_OPR_CLASS(DeformableConvBackwardData,
                     intl::DeformableConvBackwardDataBase,
                     public mixin::Convolution) // {
public:
    DeformableConvBackwardData(
            VarNode * src, VarNode * filter, VarNode * offset, VarNode * mask,
            VarNode * diff, const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    static SymbolVarArray make_all(SymbolVar src, SymbolVar filter,
                                   SymbolVar offset, SymbolVar mask,
                                   SymbolVar diff, const Param& param = {},
                                   const ExecutionPolicy& policy = {},
                                   const OperatorNodeConfig& config = {});

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar offset,
                          SymbolVar mask, SymbolVar diff,
                          const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    void scn_do_execute() override;
    std::pair<const void*, size_t> param_blob() const override;

private:
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    size_t get_workspace_size_bytes(const TensorShapeArray&,
                                    const TensorShapeArray&) const override;
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    void init_output_format() override;

    NodeProp* do_make_node_prop() const override;

    void add_input_layout_constraint() override {
        mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
    }
    void init_profile_cache() override;
};

MGB_DEFINE_OPR_CLASS(
        DeformableConvBackwardFilter,
        intl::MegDNNOprWrapperBwd<megdnn::DeformableConvBackwardFilter>,
        public mixin::Convolution) // {
public:
    DeformableConvBackwardFilter(
            VarNode * src, VarNode * filter, VarNode * offset, VarNode * mask,
            VarNode * diff, const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar offset,
                          SymbolVar mask, SymbolVar diff,
                          const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    void scn_do_execute() override;
    std::pair<const void*, size_t> param_blob() const override;

private:
    void init_profile_cache() override;
    size_t get_workspace_size_bytes(const TensorShapeArray& input_shapes,
                                    const TensorShapeArray& output_shapes)
            const override final;
};

MGB_DEFINE_OPR_CLASS(BatchConvBiasForward, intl::BatchConvBiasForwardBase, 
        public mixin::Convolution) // {
    
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
    void scn_do_execute() override;
    void get_output_var_shape(const TensorShapeArray& input_shapes,
                              TensorShapeArray& output_shapes) const override;
    void init_output_static_infer_desc() override;
    void init_output_format() override;
    void add_input_layout_constraint() override;
    void record_execute_deps(
            cg::GraphExecutable::ExecDependencyArray& deps) override {
        this->record_megdnn_opr(deps);
    }

public:
    //! src * filter
    BatchConvBiasForward(VarNode* src, VarNode* filter, const Param& param,
                         const ExecutionPolicy& policy,
                         const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter,
                          const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    //! src * filter + bias
    BatchConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                         const Param& param, const ExecutionPolicy& policy,
                         const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                          const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    //! src * filter + bias + z
    BatchConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                         VarNode* z, const Param& param,
                         const ExecutionPolicy& policy,
                         const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                          SymbolVar z, const Param& param = {},
                          const ExecutionPolicy& policy = {},
                          const OperatorNodeConfig& config = {});

    void init_profile_cache() override;
    std::pair<const void*, size_t> param_blob() const override;
};
using BatchConvBias = BatchConvBiasForward;

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
