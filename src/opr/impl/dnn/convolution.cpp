/**
 * \file src/opr/impl/dnn/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/search_policy/algo_chooser.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/system.h"
#include "megbrain/utils/hash_ct.h"
#include "megbrain/utils/timer.h"

#include "megdnn/oprs/utils.h"

#include "../internal/invoke.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "../search_policy/workspace_need_limit_getter.inl"

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

using namespace mgb;
using namespace opr;
using namespace cg::static_infer;
using intl::WorkspaceLimitGetter;

/* ==================== misc impl  ==================== */

template <class MgbOpr, class MegDNNOpr>
void mixin::ConvolutionBackwardDataMixin::
        init_output_static_infer_desc_for_bwd_data(cg::OperatorNodeBase* self) {
    using namespace cg::static_infer;
    auto&& mgr = self->owner_graph()->static_infer_manager();

    DepVal inp_deps;
    inp_deps.reserve(4);
    for (int i = 0; i < 2; ++i) {
        inp_deps.push_back({self->input(i), DepType::SHAPE});
    }

    // output shape
    if (self->input().size() == 3) {
        mgr.register_shape_infer(self->output(0),
                                 ShapeInferDesc::make_identity(self->input(2)));
    } else {
        auto infer_shp = [self](TensorShape& dest, const InpVal& inp) {
            TensorLayout ol{self->output(0)->dtype()};
            static_cast<MgbOpr*>(self)->megdnn_opr()->deduce_layout(
                    {inp.val.at(0).shape(), self->input(0)->dtype()},
                    {inp.val.at(1).shape(), self->input(1)->dtype()}, ol);
            dest = ol;
            return true;
        };
        mgr.register_shape_infer(self->output(0),
                                 {SourceType::DEP, inp_deps, infer_shp});
    }

    // workspace size
    auto infer_wk = [self](TensorShape& dest, const InpVal& inp) {
        auto&& iv = inp.val;
        dest.ndim = 1;
        dest.shape[0] = AlgoChooser<MegDNNOpr>::setup_algo(
                {TensorLayout{iv[0].shape(), self->input(0)->dtype(),
                              self->input(0)->format()},
                 {iv[1].shape(), self->input(1)->dtype(),
                  self->input(1)->format()},
                 {iv.at(2).shape(), self->output(0)->dtype(),
                  self->output(0)->format()}},
                static_cast<MgbOpr*>(self)->megdnn_opr(),
                static_cast<MgbOpr*>(self));
        return true;
    };
    inp_deps.push_back({self->output(0), DepType::SHAPE});
    auto workspace_dep_var =
            intl::WorkspaceLimitGetter::register_to_graph(self->owner_graph());
    if (workspace_dep_var) {
        inp_deps.push_back({workspace_dep_var, DepType::VALUE});
    }
    mgr.register_shape_infer(self->output(1),
                             {SourceType::DEP, inp_deps, infer_wk});
}

#define IMPL_CONV(_cls) MGB_DYN_TYPE_OBJ_FINAL_IMPL(_cls)

class mixin::WeightPreprocessExecutor::PreprocessedFilterExecDep final
        : public cg::GraphExecutable::ExecDependency {
    std::unique_ptr<PreprocessedFilter> m_pf;
    SmallVector<DeviceTensorND> m_filter_storage;

public:
    explicit PreprocessedFilterExecDep(
            std::unique_ptr<PreprocessedFilter> preprocessed_filter,
            SmallVector<DeviceTensorND> filter_storage)
            : m_pf(std::move(preprocessed_filter)),
              m_filter_storage(std::move(filter_storage)) {}
};

void mixin::WeightPreprocessExecutor::mixin_update_preprocessed_filter(
        cg::OperatorNodeBase& opr) {
    if (!mixin_allow_weight_preprocess(opr)) {
        return;
    }
    auto new_layout = deduce_preprocessed_filter_layout();
    size_t new_size = new_layout.size();
    //! No preprocess layout means no need weight preprocess
    if (new_layout.empty()) {
        return;
    }
    //! all layouts arm empty means no need weight preprocess
    bool layout_valid = false;
    for (auto&& layout : new_layout) {
        if (!layout.is_empty()) {
            layout_valid = true;
        }
    }
    if (!layout_valid) {
        return;
    }

    if (m_preprocessed_filter) {
        for (size_t i = 0; i < new_size; i++) {
            mgb_assert(new_layout[i].eq_layout(
                               m_preprocessed_filter->tensors[i].layout),
                       "weight preprocess layout changed, please keep input "
                       "shape unchanged when weight preprocess is enabled");
        }
        return;
    }
    m_preprocessed_filter.reset(new PreprocessedFilter{});
    m_preprocessed_filter->tensors.resize(new_size);
    m_filter_storage.resize(new_size);
    m_preprocessed_filter->algorithm_id = nullptr;
    for (size_t i = 0; i < new_size; i++) {
        m_filter_storage[i] = {opr.output(0)->comp_node(), new_layout[i],
                               new_layout[i].dtype, new_layout[i].format};
        m_preprocessed_filter->tensors[i] = m_filter_storage[i].as_megdnn();
    }
    scn_do_execute_preprocess();
}

void mixin::WeightPreprocessExecutor::record_preprocessed_weight(
        cg::GraphExecutable::ExecDependencyArray& deps) {
    deps.emplace_back(new PreprocessedFilterExecDep{
            std::move(m_preprocessed_filter), std::move(m_filter_storage)});
}

bool mixin::WeightPreprocessExecutor::mixin_allow_weight_preprocess(
        const cg::OperatorNodeBase& opr) const {
    if (!opr.owner_graph()->options().graph_opt.weight_preprocess) {
        return false;
    }
    if (!opr.input(1)->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE))
        return false;
    if (cg::is_const_var_value(opr.input(1)))
        return true;
    auto* input_opr = opr.input(1)->owner_opr();
    if (input_opr->same_type<opr::MultipleDeviceTensorHolder>() ||
        input_opr->same_type<opr::MultipleDeviceTensorWithFormatHolder>())
        return true;
    auto* sdt = input_opr->try_cast_final<opr::SharedDeviceTensor>();
    if (sdt && sdt->const_value())
        return true;
    auto* sdtf = input_opr->try_cast_final<opr::SharedDeviceTensorWithFormat>();
    if (sdtf && sdtf->const_value())
        return true;
    return false;
}

/* ==================== ConvolutionForward  ==================== */

IMPL_CONV(ConvolutionForward);

ConvolutionForward::ConvolutionForward(VarNode* src, VarNode* filter,
                                       const Param& param,
                                       const ExecutionPolicy& policy,
                                       const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

SymbolVar ConvolutionForward::make(SymbolVar src, SymbolVar filter,
                                   const Param& param,
                                   const ExecutionPolicy& policy,
                                   const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvolutionForward>(
            src.node(), filter.node(), param, policy, config);
}

void ConvolutionForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    megdnn_opr()->deduce_dtype(input(0)->dtype(), input(1)->dtype(),
                               output_dtype);
    output(0)->dtype(output_dtype);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ConvolutionForward) {
    mgb_assert(opr.input(0)->dtype().category() == DTypeCategory::FLOAT,
               "only float data type supported for grad");
    mgb_assert(wrt_idx == 0 || wrt_idx == 1);
    mgb_assert(out_grad.size() == 2);
    if (wrt_idx == 0) {
        // data
        SymbolVar grad = ConvolutionBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0), opr.param(),
                opr.execution_policy());
        return grad.node();
    } else {
        // filter
        SymbolVar grad = ConvolutionBackwardFilter::make(
                opr.input(0), out_grad[0], opr.input(1), opr.param(),
                opr.execution_policy());
        return grad.node();
    }
}
#endif

size_t ConvolutionForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::ConvolutionForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this, allow_weight_preprocess());
}

void ConvolutionForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

void ConvolutionForward::scn_do_execute() {
    update_preprocessed_filter();
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       preprocessed_filter(),
                       intl::get_megdnn_workspace_from_var(output().back()));
}

void ConvolutionForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void ConvolutionForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::ConvolutionForward>::val);
}

void ConvolutionForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    TensorLayout input_layout{inp_shape[0], input(0)->dtype(),
                              input(0)->format()};
    TensorLayout filter_layout{inp_shape[1], input(1)->dtype(),
                               input(1)->format()};
    TensorLayout dst_layout{output(0)->dtype(), output(0)->format()};
    megdnn_opr()->deduce_layout(input_layout, filter_layout, dst_layout);
    out_shape[0] = dst_layout;
}

void ConvolutionForward::record_execute_deps(
        cg::GraphExecutable::ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
    record_preprocessed_weight(deps);
}

SmallVector<TensorLayout>
ConvolutionForward::deduce_preprocessed_filter_layout() {
    return megdnn_opr()->deduce_preprocessed_filter_layout(
            input(0)->layout(), input(1)->layout(), output(0)->layout());
}

void ConvolutionForward::scn_do_execute_preprocess() {
    megdnn_opr()->exec_preprocess(
            input(0)->layout(), input(1)->dev_tensor().as_megdnn(),
            output(0)->layout(), preprocessed_filter(),
            intl::get_megdnn_workspace_from_var(output().back()));
    //! Flag the input(1) no use later, which can be freed when no other
    //! var depend on its dev_value, host_value and shape.
    auto receiver_info =
            input(1)->owner_graph()->var_receiver_in_current_comp_seq(input(1));
    if (receiver_info.dev_value == 1 && receiver_info.host_value == 0 &&
        receiver_info.shape == 0) {
        input(1)->add_flag(VarNode::Flag::MEMORY_NO_NEED);
    }
}

/* ==================== ConvolutionBackwardData  ==================== */
IMPL_CONV(ConvolutionBackwardData);

ConvolutionBackwardData::ConvolutionBackwardData(
        VarNode* filter, VarNode* diff, VarNode* src_for_shp,
        const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "conv_bwd_data",
                {filter, diff}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({filter, diff});
    if (src_for_shp) {
        add_input({src_for_shp});
    }
}

SymbolVar ConvolutionBackwardData::make(SymbolVar filter, SymbolVar diff,
                                        SymbolVar src, const Param& param,
                                        const ExecutionPolicy& policy,
                                        const OperatorNodeConfig& config) {
    return filter.insert_single_output_opr<ConvolutionBackwardData>(
            filter.node(), diff.node(), src.node(), param, policy, config);
}

SymbolVar ConvolutionBackwardData::make(SymbolVar filter, SymbolVar data,
                                        const Param& param,
                                        const ExecutionPolicy& policy,
                                        const OperatorNodeConfig& config) {
    return make(filter, data, {}, param, policy, config);
}

void ConvolutionBackwardData::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void ConvolutionBackwardData::init_output_static_infer_desc() {
    init_output_static_infer_desc_for_bwd_data<ConvolutionBackwardData,
                                               megdnn::ConvolutionBackwardData>(
            this);
}

void ConvolutionBackwardData::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    megdnn_opr()->deduce_dtype(input(0)->dtype(), input(1)->dtype(),
                               output_dtype);
    output(0)->dtype(output_dtype);
}

void ConvolutionBackwardData::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(1)->format());
}

cg::OperatorNodeBase::NodeProp* ConvolutionBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    if (input().size() == 3) {
        using D = NodeProp::DepType;
        prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::SHAPE});
    }
    return prop;
}

void ConvolutionBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(1)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ConvolutionBackwardData) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return ConvolutionBackwardFilter::make(out_grad[0], opr.input(1),
                                               opr.input(0), opr.param(),
                                               opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return Convolution::make(out_grad[0], opr.input(0), opr.param(),
                                 opr.execution_policy())
                .node();
    }
    return nullptr;
}
#endif

/* ==================== ConvolutionBackwardFilter  ==================== */
IMPL_CONV(ConvolutionBackwardFilter);

ConvolutionBackwardFilter::ConvolutionBackwardFilter(
        VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "conv_bwd_filter",
                 {src, diff, filter}},
                2, false) {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, diff, filter});
}

SymbolVar ConvolutionBackwardFilter::make(SymbolVar src, SymbolVar diff,
                                          SymbolVar filter, const Param& param,
                                          const ExecutionPolicy& policy,
                                          const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvolutionBackwardFilter>(
            src.node(), diff.node(), filter.node(), param, policy, config);
}

size_t ConvolutionBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 3 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::ConvolutionBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ConvolutionBackwardFilter) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return ConvolutionBackwardData::make(out_grad[0], opr.input(1),
                                             opr.input(0), opr.param(),
                                             opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return Convolution::make(opr.input(0), out_grad[0], opr.param(),
                                 opr.execution_policy())
                .node();
    }
    return nullptr;
}
#endif
/* ==================== Convolution3DForward ==================== */

IMPL_CONV(Convolution3DForward);

Convolution3DForward::Convolution3DForward(VarNode* src, VarNode* filter,
                                           const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv3d", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

SymbolVar Convolution3DForward::make(SymbolVar src, SymbolVar filter,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<Convolution3DForward>(
            src.node(), filter.node(), param, policy, config);
}

void Convolution3DForward::init_output_dtype() {
    switch (param().data_type) {
        case Param::DataType::FLOAT:
            output(0)->dtype(input(0)->dtype());
            break;
#if !MEGDNN_DISABLE_FLOAT16
        case Param::DataType::FLOAT_IO16xC32:
            mgb_assert(input(0)->dtype() == dtype::Float16(),
                       "invalid input dtype %s", input(0)->name().c_str());
            output(0)->dtype(input(0)->dtype());
            break;
#endif
        default:
            mgb_throw(MegBrainError, "bad data_type enum");
    }
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Convolution3DForward) {
    mgb_assert(opr.param().data_type ==
                       Convolution3DForward::Param::DataType::FLOAT,
               "only float data type supported for grad");
    mgb_assert(wrt_idx == 0 || wrt_idx == 1);
    mgb_assert(out_grad.size() == 2);
    if (wrt_idx == 0) {
        // data
        SymbolVar grad = Convolution3DBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0), opr.param(),
                opr.execution_policy());
        return grad.node();
    } else {
        // filter
        SymbolVar grad = Convolution3DBackwardFilter::make(
                opr.input(0), out_grad[0], opr.input(1), opr.param(),
                opr.execution_policy());
        return grad.node();
    }
}
#endif

size_t Convolution3DForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::Convolution3DForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

/* ==================== Convolution3DBackwardData  ==================== */
IMPL_CONV(Convolution3DBackwardData);

Convolution3DBackwardData::Convolution3DBackwardData(
        VarNode* filter, VarNode* diff, VarNode* src_for_shp,
        const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "conv3d_bwd_data",
                {filter, diff}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({filter, diff});
    if (src_for_shp) {
        add_input({src_for_shp});
    }
}

SymbolVar Convolution3DBackwardData::make(SymbolVar filter, SymbolVar diff,
                                          SymbolVar src, const Param& param,
                                          const ExecutionPolicy& policy,
                                          const OperatorNodeConfig& config) {
    return filter.insert_single_output_opr<Convolution3DBackwardData>(
            filter.node(), diff.node(), src.node(), param, policy, config);
}

SymbolVar Convolution3DBackwardData::make(SymbolVar filter, SymbolVar data,
                                          const Param& param,
                                          const ExecutionPolicy& policy,
                                          const OperatorNodeConfig& config) {
    return make(filter, data, {}, param, policy, config);
}

void Convolution3DBackwardData::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void Convolution3DBackwardData::init_output_static_infer_desc() {
    init_output_static_infer_desc_for_bwd_data<
            Convolution3DBackwardData, megdnn::Convolution3DBackwardData>(this);
}

cg::OperatorNodeBase::NodeProp* Convolution3DBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    if (input().size() == 3) {
        using D = NodeProp::DepType;
        prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::SHAPE});
    }
    return prop;
}

void Convolution3DBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(1)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Convolution3DBackwardData) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return Convolution3DBackwardFilter::make(out_grad[0], opr.input(1),
                                                 opr.input(0), opr.param(),
                                                 opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return Convolution3D::make(out_grad[0], opr.input(0), opr.param(),
                                   opr.execution_policy())
                .node();
    }
    return nullptr;
}
#endif

/* ==================== Convolution3DBackwardFilter  ==================== */
IMPL_CONV(Convolution3DBackwardFilter);

Convolution3DBackwardFilter::Convolution3DBackwardFilter(
        VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "conv3d_bwd_filter",
                 {src, diff, filter}},
                2, false) {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, diff, filter});
}

SymbolVar Convolution3DBackwardFilter::make(SymbolVar src, SymbolVar diff,
                                            SymbolVar filter,
                                            const Param& param,
                                            const ExecutionPolicy& policy,
                                            const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<Convolution3DBackwardFilter>(
            src.node(), diff.node(), filter.node(), param, policy, config);
}

size_t Convolution3DBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 3 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::Convolution3DBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

/* ========================== MaskConvolution  ========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MaskConvolution);

MaskConvolution::MaskConvolution(VarNode* src, VarNode* filter, VarNode* mask,
                                 const Param& param,
                                 const OperatorNodeConfig& config)
        : Super(src->owner_graph(), config, "mask_conv_fwd",
                {src, filter, mask}) {
    init_megdnn_opr(*this, param);
    add_input({src, filter, mask});
}

SymbolVar MaskConvolution::make(SymbolVar src, SymbolVar filter, SymbolVar mask,
                                const Param& param,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<MaskConvolution>(
            src.node(), filter.node(), mask.node(), param, config);
}

void MaskConvolution::init_output_dtype() {
    auto dtype = input(2)->dtype();
    mgb_assert(dtype == dtype::Int32() || dtype == dtype::Int16() ||
                       dtype == dtype::Int8(),
               "dtype must be int8, int16 or int32, while get %s",
               dtype.name());
    output(0)->dtype(input(0)->dtype());
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MaskPropagate);

MaskPropagate::MaskPropagate(VarNode* src, const Param& param,
                             const OperatorNodeConfig& config)
        : Super(src->owner_graph(), config, "mask_propagate", {src}) {
    init_megdnn_opr(*this, param);
    add_input({src});
}

void MaskPropagate::init_output_dtype() {
    auto dtype = input(0)->dtype();
    mgb_assert(dtype == dtype::Int32() || dtype == dtype::Int16() ||
               dtype == dtype::Int8());
    output(0)->dtype(dtype);
}

SymbolVar MaskPropagate::make(SymbolVar src, const Param& param,
                              const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<MaskPropagate>(src.node(), param,
                                                       config);
}

/* ==================== ConvBiasForward  ==================== */
IMPL_CONV(ConvBiasForward);

ConvBiasForward::ConvBiasForward(VarNode* src, VarNode* filter,
                                 const Param& param,
                                 const ExecutionPolicy& policy,
                                 const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv_bias", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

ConvBiasForward::ConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                                 const Param& param,
                                 const ExecutionPolicy& policy,
                                 const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "conv_bias", {src, filter, bias}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias});
}

ConvBiasForward::ConvBiasForward(VarNode* src, VarNode* filter, VarNode* bias,
                                 VarNode* z, const Param& param,
                                 const ExecutionPolicy& policy,
                                 const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "conv_bias",
                {src, filter, bias, z}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias, z});
}

void ConvBiasForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

SymbolVar ConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                const Param& param,
                                const ExecutionPolicy& policy,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvBiasForward>(
            src.node(), filter.node(), param, policy, config);
}

SymbolVar ConvBiasForward::make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                                const Param& param,
                                const ExecutionPolicy& policy,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvBiasForward>(
            src.node(), filter.node(), bias.node(), param, policy, config);
}

SymbolVar ConvBiasForward::make(SymbolVar src, SymbolVar filter, SymbolVar bias,
                                SymbolVar z, const Param& param,
                                const ExecutionPolicy& policy,
                                const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ConvBiasForward>(
            src.node(), filter.node(), bias.node(), z.node(), param, policy,
            config);
}

void ConvBiasForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    DType i0, i1, i2, i3;
    mgb_assert(input().size() >= 2 && input().size() <= 4);
    i0 = input(0)->dtype();
    i1 = input(1)->dtype();
    if (input().size() >= 3)
        i2 = input(2)->dtype();
    if (input().size() == 4)
        i3 = input(3)->dtype();
    megdnn_opr()->deduce_dtype(i0, i1, i2, i3, output_dtype);
    output(0)->dtype(output_dtype);
}

size_t ConvBiasForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    auto mo = megdnn_opr();
    TensorLayout i0, i1, i2, i3;
    mgb_assert(input_shapes.size() >= 2 && input_shapes.size() <= 4);
    i0 = {input_shapes[0], input(0)->dtype(), input(0)->format()};
    i1 = {input_shapes[1], input(1)->dtype(), input(1)->format()};
    if (input_shapes.size() >= 3)
        i2 = {input_shapes[2], input(2)->dtype(), input(2)->format()};
    else {
        DType dtype;
        mo->deduce_dtype(input(0)->dtype(), input(1)->dtype(), DType{}, DType{},
                         dtype);
        i2 = {{}, dtype};
    }
    if (input_shapes.size() == 4)
        i3 = {input_shapes[3], input(3)->dtype(), input(3)->format()};
    else
        i3 = {{}, output(0)->dtype(), output(0)->format()};

    return AlgoChooser<megdnn::ConvBias>::setup_algo(
            {i0,
             i1,
             i2,
             i3,
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            mo, this, allow_weight_preprocess());
}

void ConvBiasForward::scn_do_execute() {
    update_preprocessed_filter();

    auto&& inp = input();
    auto mo = megdnn_opr();
    if (inp.size() == 2) {
        TensorLayout bias_layout;
        bias_layout.ndim = 0;
        if (output(0)->dtype().enumv() == DTypeEnum::QuantizedS8) {
            bias_layout.dtype = dtype::QuantizedS32(
                    output(0)->dtype().param<dtype::QuantizedS8>().scale);
        } else {
            bias_layout.dtype = output(0)->dtype();
        }
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND bias_tensor{nullptr, bias_layout};
        megdnn::TensorND z_tensor{nullptr, z_layout};
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(), bias_tensor, z_tensor,
                 output(0)->dev_tensor().as_megdnn(), preprocessed_filter(),
                 intl::get_megdnn_workspace_from_var(output().back()));

    } else if (inp.size() == 3) {
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND z_tensor{nullptr, z_layout};

        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(), z_tensor,
                 output(0)->dev_tensor().as_megdnn(), preprocessed_filter(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    } else {
        mgb_assert(inp.size() == 4);
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(),
                 inp[3]->dev_tensor().as_megdnn(),
                 output(0)->dev_tensor().as_megdnn(), preprocessed_filter(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    }
}

void ConvBiasForward::get_output_var_shape(const TensorShapeArray& inp_shape,
                                           TensorShapeArray& out_shape) const {
    auto mo = megdnn_opr();
    TensorLayout dst;
    mo->deduce_layout({inp_shape[0], input(0)->dtype(), input(0)->format()},
                      {inp_shape[1], input(1)->dtype(), input(0)->format()}, {},
                      {}, dst);
    out_shape[0] = dst;
}

void ConvBiasForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::ConvBiasForward>::val);
}

void ConvBiasForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

void ConvBiasForward::check_winograd_param_valid(
        const megdnn::ConvBias::WinogradParam& param, const DType& dtype) {
    if (dtype.enumv() == DTypeEnum::Float32) {
        mgb_assert(param.channel_block_size == 1 ||
                           param.channel_block_size == 4 ||
                           param.channel_block_size == 8,
                   "only support 1/4/8 for the channel_block_size of "
                   "winograd param, got %u",
                   param.channel_block_size);
    } else {
        mgb_assert((MEGDNN_FLOAT16_SELECT(dtype.enumv() == DTypeEnum::Float16,
                                          false) ||
                    dtype.enumv() == DTypeEnum::QuantizedS8 ||
                    dtype.enumv() == DTypeEnum::Quantized8Asymm) &&
                           (param.channel_block_size == 1 ||
                            param.channel_block_size == 4 ||
                            param.channel_block_size == 8),
                   "only support 1/4/8 for the channel_block_size of "
                   "winograd param, got %u",
                   param.channel_block_size);
    }
}

megdnn::param::MatrixMul::Format ConvBiasForward::get_matmul_format(
        const megdnn::ConvBias::WinogradParam& param) {
    switch (param.channel_block_size) {
        case 1:
            return megdnn::param::MatrixMul::Format::DEFAULT;
            break;
        case 4:
            return megdnn::param::MatrixMul::Format::MK4;
            break;
        case 8:
            return megdnn::param::MatrixMul::Format::MK8;
            break;
        default:
            mgb_throw(InternalError,
                      "Only Support 1/4/8 for "
                      "channel_block_size, got: %u",
                      param.channel_block_size);
    }
}

SmallVector<TensorLayout> ConvBiasForward::deduce_preprocessed_filter_layout() {
    TensorLayout i2, i3;
    if (input().size() > 2) {
        i2 = input(2)->layout();
    }
    if (input().size() > 3) {
        i3 = input(3)->layout();
    }
    return megdnn_opr()->deduce_preprocessed_filter_layout(
            input(0)->layout(), input(1)->layout(), i2, i3,
            output(0)->layout());
}

void ConvBiasForward::scn_do_execute_preprocess() {
    TensorLayout bias_layout(output(0)->dtype()), z_layout(output(0)->dtype());
    if (input().size() > 2) {
        bias_layout = input(2)->layout();
    }
    if (input().size() > 3) {
        z_layout = input(3)->layout();
    }
    if (input().size() > 2) {
        megdnn_opr()->exec_preprocess(
                input(0)->layout(), input(1)->dev_tensor().as_megdnn(),
                input(2)->dev_tensor().as_megdnn(), z_layout,
                output(0)->layout(), preprocessed_filter(),
                intl::get_megdnn_workspace_from_var(output().back()));
    } else {
        megdnn::TensorND bias_tensor{nullptr, bias_layout};
        megdnn_opr()->exec_preprocess(
                input(0)->layout(), input(1)->dev_tensor().as_megdnn(),
                bias_tensor, z_layout, output(0)->layout(),
                preprocessed_filter(),
                intl::get_megdnn_workspace_from_var(output().back()));
    }
    //! Flag the weight and bias no use later, which can be freed when no other
    //! var depend on its dev_value, host_value and shape.
    auto receiver_info_weight =
            input(1)->owner_graph()->var_receiver_in_current_comp_seq(input(1));
    if (receiver_info_weight.dev_value == 1 &&
        receiver_info_weight.host_value == 0 &&
        receiver_info_weight.shape == 0) {
        input(1)->add_flag(VarNode::Flag::MEMORY_NO_NEED);
    }
    //! if bias is preprocessd
    if (input().size() > 2) {
        auto preprocessed_layouts =
                megdnn_opr()->deduce_preprocessed_filter_layout(
                        input(0)->layout(), input(1)->layout(), bias_layout,
                        z_layout, output(0)->layout());
        if (preprocessed_layouts.size() > 1 &&
            !preprocessed_layouts[1].is_empty()) {
            auto receiver_info_bias =
                    input(2)->owner_graph()->var_receiver_in_current_comp_seq(
                            input(2));
            if (receiver_info_bias.dev_value == 1 &&
                receiver_info_bias.host_value == 0 &&
                receiver_info_bias.shape == 0) {
                input(2)->add_flag(VarNode::Flag::MEMORY_NO_NEED);
            }
        }
    }
}

/* ===================== LocalShareForward ==================== */

IMPL_CONV(LocalShareForward);

LocalShareForward::LocalShareForward(VarNode* src, VarNode* filter,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "local_share", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

SymbolVar LocalShareForward::make(SymbolVar src, SymbolVar filter,
                                  const Param& param,
                                  const ExecutionPolicy& policy,
                                  const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<LocalShareForward>(
            src.node(), filter.node(), param, policy, config);
}

void LocalShareForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
}

void LocalShareForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

size_t LocalShareForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::LocalShareForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LocalShareForward) {
    mgb_assert(opr.input(0)->dtype().category() == DTypeCategory::FLOAT,
               "only float data type supported for grad");
    mgb_assert(wrt_idx == 0 || wrt_idx == 1);
    mgb_assert(out_grad.size() == 2);
    if (wrt_idx == 0) {
        // data
        SymbolVar grad = LocalShareBackwardData::make(opr.input(1), out_grad[0],
                                                      opr.input(0), opr.param(),
                                                      opr.execution_policy());
        return grad.node();
    } else {
        // filter
        SymbolVar grad = LocalShareBackwardFilter::make(
                opr.input(0), out_grad[0], opr.input(1), opr.param(),
                opr.execution_policy());
        return grad.node();
    }
}
#endif

/* ===================== LocalShareBackwardData ==================== */

IMPL_CONV(LocalShareBackwardData);

LocalShareBackwardData::LocalShareBackwardData(VarNode* filter, VarNode* diff,
                                               VarNode* src_for_shp,
                                               const Param& param,
                                               const ExecutionPolicy& policy,
                                               const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "local_share_bwd_data",
                {filter, diff}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({filter, diff});
    if (src_for_shp) {
        add_input({src_for_shp});
    }
}

SymbolVar LocalShareBackwardData::make(SymbolVar filter, SymbolVar diff,
                                       SymbolVar src, const Param& param,
                                       const ExecutionPolicy& policy,
                                       const OperatorNodeConfig& config) {
    return filter.insert_single_output_opr<LocalShareBackwardData>(
            filter.node(), diff.node(), src.node(), param, policy, config);
}

void LocalShareBackwardData::init_output_static_infer_desc() {
    init_output_static_infer_desc_for_bwd_data<LocalShareBackwardData,
                                               megdnn::LocalShareBackwardData>(
            this);
}

void LocalShareBackwardData::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
}

void LocalShareBackwardData::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

cg::OperatorNodeBase::NodeProp* LocalShareBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    mgb_assert(input().size() == 3);
    using D = NodeProp::DepType;
    prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::SHAPE});
    return prop;
}

void LocalShareBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(1)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LocalShareBackwardData) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return LocalShareBackwardFilter::make(out_grad[0], opr.input(1),
                                              opr.input(0), opr.param(),
                                              opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return LocalShare::make(out_grad[0], opr.input(0), opr.param(),
                                opr.execution_policy())
                .node();
    }
    return nullptr;
}
#endif

/* ==================== LocalShareBackwardFilter  ==================== */

IMPL_CONV(LocalShareBackwardFilter);

LocalShareBackwardFilter::LocalShareBackwardFilter(
        VarNode* src, VarNode* diff, VarNode* filter, const Param& param,
        const ExecutionPolicy& policy, const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "local_share_bwd_filter",
                 {src, diff, filter}},
                2, false) {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, diff, filter});
}

SymbolVar LocalShareBackwardFilter::make(SymbolVar src, SymbolVar diff,
                                         SymbolVar filter, const Param& param,
                                         const ExecutionPolicy& policy,
                                         const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<LocalShareBackwardFilter>(
            src.node(), diff.node(), filter.node(), param, policy, config);
}

size_t LocalShareBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 3 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::LocalShareBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LocalShareBackwardFilter) {
    mgb_assert(!out_grad[1]);
    if (wrt_idx == 0) {
        return LocalShareBackwardData::make(out_grad[0], opr.input(1),
                                            opr.input(0), opr.param(),
                                            opr.execution_policy())
                .node();
    }
    if (wrt_idx == 1) {
        return LocalShare::make(opr.input(0), out_grad[0], opr.param(),
                                opr.execution_policy())
                .node();
    }
    return nullptr;
}
#endif

/* ===================== DeformableConvForward ==================== */

IMPL_CONV(DeformableConvForward);

DeformableConvForward::DeformableConvForward(VarNode* src, VarNode* filter,
                                             VarNode* offset, VarNode* mask,
                                             const Param& param,
                                             const ExecutionPolicy& policy,
                                             const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "deformable_conv",
                {src, filter, offset, mask}} {
    mgb_assert(src->dtype() == dtype::Float32() &&
                       filter->dtype() == dtype::Float32() &&
                       offset->dtype() == dtype::Float32() &&
                       mask->dtype() == dtype::Float32(),
               "input should be float32, got %s, %s, %s, %s",
               src->dtype().name(), filter->dtype().name(),
               offset->dtype().name(), mask->dtype().name());

    init_megdnn_opr(*this, param);
    m_policy = policy;

    add_input({src, filter, offset, mask});
}

SymbolVar DeformableConvForward::make(SymbolVar src, SymbolVar filter,
                                      SymbolVar offset, SymbolVar mask,
                                      const Param& param,
                                      const ExecutionPolicy& policy,
                                      const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<DeformableConvForward>(
            src.node(), filter.node(), offset.node(), mask.node(), param,
            policy, config);
}

void DeformableConvForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
}

void DeformableConvForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

size_t DeformableConvForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 4 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::DeformableConvForward>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[1], input(1)->dtype(), input(1)->format()},
             {input_shapes[2], input(2)->dtype(), input(2)->format()},
             {input_shapes[3], input(3)->dtype(), input(3)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(DeformableConvForward) {
    mgb_assert(opr.input(0)->dtype() == dtype::Float32(),
               "only float data type supported for grad");
    mgb_assert(wrt_idx < 4);
    mgb_assert(!out_grad[1]);
    mgb_assert(out_grad.size() == 2);

    // data, offset and mask
    auto grad_arr = DeformableConvBackwardData::make_all(
            opr.input(0), opr.input(1), opr.input(2), opr.input(3), out_grad[0],
            opr.param(), opr.execution_policy(), opr.config());
    // filter
    auto filter_grad = DeformableConvBackwardFilter::make(
            opr.input(0), opr.input(1), opr.input(2), opr.input(3), out_grad[0],
            opr.param(), opr.execution_policy(), opr.config());

    SymbolVarArray grads = {grad_arr[0], filter_grad, grad_arr[1], grad_arr[2]};
    return grads[wrt_idx].node();
}
#endif

/* ==================== DeformableConvBackwardData  ==================== */

IMPL_CONV(DeformableConvBackwardData);

DeformableConvBackwardData::DeformableConvBackwardData(
        VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
        VarNode* diff, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super{filter->owner_graph(),
                config,
                "deformable_conv_backward_data",
                {src, filter, offset, mask, diff}} {
    mgb_assert(src->dtype() == dtype::Float32() and
                       filter->dtype() == dtype::Float32() and
                       offset->dtype() == dtype::Float32() and
                       mask->dtype() == dtype::Float32() and
                       diff->dtype() == dtype::Float32(),
               "input should be float32, got %s, %s, %s, %s %s",
               src->dtype().name(), filter->dtype().name(),
               offset->dtype().name(), mask->dtype().name(),
               diff->dtype().name());

    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter, offset, mask, diff});
}

SymbolVarArray DeformableConvBackwardData::make_all(
        SymbolVar src, SymbolVar filter, SymbolVar offset, SymbolVar mask,
        SymbolVar diff, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config) {
    auto graph = src.node()->owner_graph();

    auto back_node =
            graph->insert_opr(std::make_unique<DeformableConvBackwardData>(
                    src.node(), filter.node(), offset.node(), mask.node(),
                    diff.node(), param, policy, config));

    return {back_node->output(0), back_node->output(1), back_node->output(2)};
}

SymbolVar DeformableConvBackwardData::make(SymbolVar src, SymbolVar filter,
                                           SymbolVar offset, SymbolVar mask,
                                           SymbolVar diff, const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config) {
    auto&& all =
            make_all(src, filter, offset, mask, diff, param, policy, config);
    return all[0];
}

void DeformableConvBackwardData::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),   // src
                       input(1)->dev_tensor().as_megdnn(),   // filter
                       input(2)->dev_tensor().as_megdnn(),   // offset
                       input(3)->dev_tensor().as_megdnn(),   // mask
                       input(4)->dev_tensor().as_megdnn(),   // diff
                       output(0)->dev_tensor().as_megdnn(),  // src_grad
                       output(1)->dev_tensor().as_megdnn(),  // offset_grad
                       output(2)->dev_tensor().as_megdnn(),  // mask_grad
                       intl::get_megdnn_workspace_from_var(output(3)));
}

void DeformableConvBackwardData::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    TensorShape im_shp = inp_shape[0];
    TensorShape offset_shp = inp_shape[2];
    TensorShape mask_shp = inp_shape[3];

    mgb_assert(im_shp.ndim == 4, "invalid src shape: %s",
               im_shp.to_string().c_str());
    mgb_assert(offset_shp.ndim == 4, "invalid offset shape: %s",
               offset_shp.to_string().c_str());
    mgb_assert(mask_shp.ndim == 4, "invalid mask shape: %s",
               mask_shp.to_string().c_str());
    mgb_assert(out_shape.size() == 3);

    out_shape[0] = im_shp;
    out_shape[1] = offset_shp;
    out_shape[2] = mask_shp;
}

size_t DeformableConvBackwardData::get_workspace_size_bytes(
        const TensorShapeArray& inp_shape,
        const TensorShapeArray& out_shape) const {
    size_t ws = AlgoChooser<megdnn::DeformableConvBackwardData>::setup_algo(
            {TensorLayout{inp_shape[0], input(0)->dtype(), input(0)->format()},
             {inp_shape[1], input(1)->dtype(), input(1)->format()},
             {inp_shape[2], input(2)->dtype(), input(2)->format()},
             {inp_shape[3], input(3)->dtype(), input(3)->format()},
             {inp_shape[4], input(4)->dtype(), input(4)->format()},
             {out_shape[0], output(0)->dtype(), output(0)->format()},
             {out_shape[1], output(1)->dtype(), output(1)->format()},
             {out_shape[2], output(2)->dtype(), output(2)->format()}},
            megdnn_opr(), this);
    return ws;
}

void DeformableConvBackwardData::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
    output(1)->dtype(output_dtype);
    output(2)->dtype(output_dtype);
}

void DeformableConvBackwardData::init_output_format() {
    mgb_assert(output().size() == 4);
    output(0)->format(input(0)->format());
    output(1)->format(input(2)->format());
    output(2)->format(input(3)->format());
}

cg::OperatorNodeBase::NodeProp* DeformableConvBackwardData::do_make_node_prop()
        const {
    auto prop = Super::Super::do_make_node_prop();
    using D = NodeProp::DepType;
    mgb_assert(input().size() == 5);
    prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::DEV_VALUE,
                                   D::DEV_VALUE, D::DEV_VALUE});
    return prop;
}

void DeformableConvBackwardData::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::DeformableConvBackwardData>::val);
}

/* ==================== DeformableConvBackwardFilter  ==================== */

IMPL_CONV(DeformableConvBackwardFilter);

DeformableConvBackwardFilter::DeformableConvBackwardFilter(
        VarNode* src, VarNode* filter, VarNode* offset, VarNode* mask,
        VarNode* diff, const Param& param, const ExecutionPolicy& policy,
        const OperatorNodeConfig& config)
        : Super({src->owner_graph(),
                 config,
                 "deformable_conv_backward_filter",
                 {src, filter, offset, mask, diff}},
                1, false) {
    mgb_assert(src->dtype() == dtype::Float32() and
                       filter->dtype() == dtype::Float32() and
                       offset->dtype() == dtype::Float32() and
                       mask->dtype() == dtype::Float32() and
                       diff->dtype() == dtype::Float32(),
               "input should be float32, got %s, %s, %s, %s %s",
               src->dtype().name(), filter->dtype().name(),
               offset->dtype().name(), mask->dtype().name(),
               diff->dtype().name());
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter, offset, mask, diff});
}

SymbolVar DeformableConvBackwardFilter::make(SymbolVar src, SymbolVar filter,
                                             SymbolVar offset, SymbolVar mask,
                                             SymbolVar diff, const Param& param,
                                             const ExecutionPolicy& policy,
                                             const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<DeformableConvBackwardFilter>(
            src.node(), filter.node(), offset.node(), mask.node(), diff.node(),
            param, policy, config);
}

void DeformableConvBackwardFilter::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),   // src
                       input(2)->dev_tensor().as_megdnn(),   // offset
                       input(3)->dev_tensor().as_megdnn(),   // mask
                       input(4)->dev_tensor().as_megdnn(),   // diff
                       output(0)->dev_tensor().as_megdnn(),  // filter_diff
                       intl::get_megdnn_workspace_from_var(output(1)));
}

size_t DeformableConvBackwardFilter::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    mgb_assert(input_shapes.size() == 5 && output_shapes.size() == 1);
    return AlgoChooser<megdnn::DeformableConvBackwardFilter>::setup_algo(
            {TensorLayout{input_shapes[0], input(0)->dtype(),
                          input(0)->format()},
             {input_shapes[2], input(2)->dtype(), input(2)->format()},
             {input_shapes[3], input(3)->dtype(), input(3)->format()},
             {input_shapes[4], input(4)->dtype(), input(4)->format()},
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            megdnn_opr(), this);
}

/* ==================== BatchConvBiasForward  ==================== */
IMPL_CONV(BatchConvBiasForward);

BatchConvBiasForward::BatchConvBiasForward(VarNode* src, VarNode* filter,
                                           const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "batch_conv_bias", {src, filter}} {
    init_megdnn_opr(*this, param);
    m_policy = policy;
    add_input({src, filter});
}

BatchConvBiasForward::BatchConvBiasForward(VarNode* src, VarNode* filter,
                                           VarNode* bias, const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "batch_conv_bias",
                {src, filter, bias}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias});
}

BatchConvBiasForward::BatchConvBiasForward(VarNode* src, VarNode* filter,
                                           VarNode* bias, VarNode* z,
                                           const Param& param,
                                           const ExecutionPolicy& policy,
                                           const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "batch_conv_bias",
                {src, filter, bias, z}} {
    m_policy = policy;
    init_megdnn_opr(*this, param);
    add_input({src, filter, bias, z});
}

void BatchConvBiasForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

SymbolVar BatchConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<BatchConvBiasForward>(
            src.node(), filter.node(), param, policy, config);
}

SymbolVar BatchConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                     SymbolVar bias, const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<BatchConvBiasForward>(
            src.node(), filter.node(), bias.node(), param, policy, config);
}

SymbolVar BatchConvBiasForward::make(SymbolVar src, SymbolVar filter,
                                     SymbolVar bias, SymbolVar z,
                                     const Param& param,
                                     const ExecutionPolicy& policy,
                                     const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<BatchConvBiasForward>(
            src.node(), filter.node(), bias.node(), z.node(), param, policy,
            config);
}

void BatchConvBiasForward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    DType i0, i1, i2, i3;
    mgb_assert(input().size() >= 2 && input().size() <= 4);
    i0 = input(0)->dtype();
    i1 = input(1)->dtype();
    if (input().size() >= 3)
        i2 = input(2)->dtype();
    if (input().size() == 4)
        i3 = input(3)->dtype();
    megdnn_opr()->deduce_dtype(i0, i1, i2, i3, output_dtype);
    output(0)->dtype(output_dtype);
}

size_t BatchConvBiasForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    auto mo = megdnn_opr();
    TensorLayout i0, i1, i2, i3;
    mgb_assert(input_shapes.size() >= 2 && input_shapes.size() <= 4);
    i0 = {input_shapes[0], input(0)->dtype(), input(0)->format()};
    i1 = {input_shapes[1], input(1)->dtype(), input(1)->format()};
    if (input_shapes.size() >= 3)
        i2 = {input_shapes[2], input(2)->dtype(), input(2)->format()};
    else {
        DType dtype;
        mo->deduce_dtype(input(0)->dtype(), input(1)->dtype(), DType{}, DType{},
                         dtype);
        i2 = {{}, dtype};
    }
    if (input_shapes.size() == 4)
        i3 = {input_shapes[3], input(3)->dtype(), input(3)->format()};
    else
        i3 = {{}, output(0)->dtype(), output(0)->format()};

    return AlgoChooser<megdnn::BatchConvBias>::setup_algo(
            {i0,
             i1,
             i2,
             i3,
             {output_shapes[0], output(0)->dtype(), output(0)->format()}},
            mo, this);
}

void BatchConvBiasForward::scn_do_execute() {
    auto&& inp = input();
    auto mo = megdnn_opr();
    if (inp.size() == 2) {
        TensorLayout bias_layout;
        bias_layout.ndim = 0;
        if (output(0)->dtype().enumv() == DTypeEnum::QuantizedS8) {
            bias_layout.dtype = dtype::QuantizedS32(
                    output(0)->dtype().param<dtype::QuantizedS8>().scale);
        } else {
            bias_layout.dtype = output(0)->dtype();
        }
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND bias_tensor{nullptr, bias_layout};
        megdnn::TensorND z_tensor{nullptr, z_layout};
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(), bias_tensor, z_tensor,
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));

    } else if (inp.size() == 3) {
        TensorLayout z_layout;
        z_layout.ndim = 0;
        z_layout.dtype = output(0)->dtype();
        megdnn::TensorND z_tensor{nullptr, z_layout};

        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(), z_tensor,
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    } else {
        mgb_assert(inp.size() == 4);
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(),
                 inp[3]->dev_tensor().as_megdnn(),
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    }
}

void BatchConvBiasForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    auto mo = megdnn_opr();
    TensorLayout dst;
    mo->deduce_layout({inp_shape[0], input(0)->dtype(), input(0)->format()},
                      {inp_shape[1], input(1)->dtype(), input(0)->format()}, {},
                      {}, dst);
    out_shape[0] = dst;
}

void BatchConvBiasForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::BatchConvBiasForward>::val);
}

void BatchConvBiasForward::init_output_format() {
    mgb_assert(output().size() == 2);
    output(0)->format(input(0)->format());
}

#undef IMPL_CONV
#undef MGB_FOREACH_FASTRUN_OPR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
