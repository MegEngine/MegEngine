/**
 * \file src/gopt/impl/profiler_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./opr_format_modifier.h"
#include "./utils.h"
#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/profiler.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/base.h"
#include "megbrain/serialization/sereg.h"
#include "megdnn/tensor_format.h"

using namespace mgb;
using namespace cg;
using namespace opr;
using namespace gopt;
using ReformatKey = ReformatManager::ReformatKey;

namespace {
class GraphPartitionProfiler final : public PluginBase {
    using CompNodeEventPtr = std::unique_ptr<CompNode::Event>;

public:
    using OprFilter = thin_function<bool(OperatorNodeBase*)>;
    struct OprKernEvent {
        CompNodeEventPtr start, end;
    };
    GraphPartitionProfiler(ComputingGraph* graph, OprFilter opr_filter);
    ~GraphPartitionProfiler() noexcept;
    float duration_in_usec() const;

private:
    void record_event(CompNodeEventPtr& dest, CompNode cn) {
        if (dest == nullptr)
            dest = cn.create_event(CompNode::Event::NEED_TIMER);
        dest->record();
    }
    ThinHashMap<OperatorNodeBase*, OprKernEvent> m_kern_event;
    OprFilter m_opr_filter;
};

GraphPartitionProfiler::GraphPartitionProfiler(
        ComputingGraph* graph, OprFilter opr_filter)
        : PluginBase(graph), m_opr_filter(opr_filter) {
    using namespace event;
    auto on_before_kern = [this](BeforeKernel const& event) {
        if (!m_opr_filter(event.opr))
            return;
        auto evptr = &m_kern_event[event.opr].start;
        record_event(*evptr, event.comp_node);
    };
    auto on_after_kern = [this](AfterKernel const& event) {
        if (!m_opr_filter(event.opr))
            return;
        auto evptr = &m_kern_event[event.opr].end;
        record_event(*evptr, event.comp_node);
    };
    auto&& ev = graph->event();
    add_event_handler(ev.register_receiver<BeforeKernel>(on_before_kern));
    add_event_handler(ev.register_receiver<AfterKernel>(on_after_kern));
}

GraphPartitionProfiler::~GraphPartitionProfiler() noexcept {
    auto wait = [](const CompNodeEventPtr& ev) {
        if (ev)
            ev->host_wait();
    };
    for (auto&& i : m_kern_event) {
        wait(i.second.start);
        wait(i.second.end);
    }
}

float GraphPartitionProfiler::duration_in_usec() const {
    float device_duration = 0.f;
    for (auto&& kern_ev : m_kern_event) {
        auto&& event = kern_ev.second;
        event.end->host_wait();
        device_duration += 1e6 * event.start->elapsed_time_until(*event.end);
    }
    return device_duration;
}

/*!
 * \brief An operator that indicates its input var node is contiguous
 */
// clang-format off
MGB_DEFINE_OPR_CLASS(MarkInputContiguous, SingleCNOperatorNodeBase) //{
    void scn_do_execute() override {};
    void init_output_static_infer_desc() override;
    void add_input_layout_constraint() override {
        input(0)->add_layout_constraint_contiguous();
    }
public:
    MarkInputContiguous(VarNode* input, const OperatorNodeConfig& config);
    static SymbolVar make(SymbolVar input, const OperatorNodeConfig& config = {});
};
// clang-format on

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MarkInputContiguous);

MarkInputContiguous::MarkInputContiguous(
        VarNode* input, const OperatorNodeConfig& config)
        : Super(input->owner_graph(), config, "mark_contiguous", {input}) {
    add_input({input});
    add_output(None);
}

SymbolVar MarkInputContiguous::make(SymbolVar input, const OperatorNodeConfig& config) {
    return input.insert_single_output_opr<MarkInputContiguous>(input.node(), config);
}

void MarkInputContiguous::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));
}
}  // namespace

/* ================== ProfilerImpl =================*/
ProfilerImpl::ProfilerImpl(int runs, float opr_threshold, float var_node_threshold)
        : m_opr_threshold{opr_threshold},
          m_var_node_threshold{var_node_threshold},
          m_runs{runs} {
    m_opr_filter = [this](const OperatorNodeBase* opr, OperatorNodeBase* new_opr) {
        /// \note: for the considerations of performance, we skip nchw(naive)
        /// kernels for conv bias on CUDA platform. to remove this later
        if (auto conv = try_cast_as_op<opr::ConvBiasForward>(new_opr)) {
            if (conv->output(0)->comp_node().device_type() ==
                        CompNode::DeviceType::CUDA &&
                conv->input(0)->dtype().category() == DTypeCategory::QUANTIZED &&
                conv->param().format == OprFormat::NCHW) {
                return false;
            }
        }
        float comp1 =
                m_opr_footprint.get_computation(const_cast<OperatorNodeBase*>(opr));
        float comp2 = m_opr_footprint.get_computation(new_opr);
        if (comp2 > m_opr_threshold * comp1)
            return false;
        return true;
    };
    m_var_node_filter = [this](const VarNode* var, TensorShape from, TensorShape to,
                               ReformatKey key) {
        /// \note: due to the alignment requirement of low-bit tensor, we skip
        /// some layout transform for low-bit tensors. The skipped layout
        /// transforms do not have corresponding dnn kernel and cannot be
        /// implemented by tensor manip operators (like reshape, dimshuffle,
        /// subtensor, etc.).
        if (var->dtype().enumv() == DTypeEnum::QuantizedS4 ||
            var->dtype().enumv() == DTypeEnum::Quantized4Asymm) {
            if (key.input_format == TensorFormats::NCHW &&
                key.output_format != TensorFormats::NHWC &&
                key.output_format != TensorFormats::NCHWc64) {
                return false;
            }
            if (key.output_format == TensorFormats::NCHW &&
                key.input_format != TensorFormats::NHWC &&
                key.input_format != TensorFormats::NCHWc64) {
                return false;
            }
        }
        TensorLayout orig_ly = {var->shape(), var->dtype()},
                     from_ly = {from, var->dtype()}, to_ly = {to, var->dtype()};
        float orig_memory = orig_ly.span().dist_byte() * 2.f;
        float reformat_memory = from_ly.span().dist_byte() + to_ly.span().dist_byte();
        if (reformat_memory > orig_memory * m_var_node_threshold)
            return false;
        return true;
    };
}

ProfilerImpl::OperatorNodeRecord ProfilerImpl::profile_operator(
        const OperatorNodeBase* opr, TensorFormats base_format,
        const SmallVector<TensorFormats>& available_tensor_formats,
        ReformatAttribute extra_attribute) const {
    OperatorNodeRecord record;
    record.opr = opr;
    auto& costs = record.costs;
    for (auto&& f : available_tensor_formats) {
        auto config_id = tensor_formats_to_config_id(f);
        costs[config_id] = profile_operator(opr, base_format, f, extra_attribute);
    }
    return record;
}

float ProfilerImpl::profile_operator(
        const OperatorNodeBase* opr, TensorFormats base_format,
        TensorFormats tensor_format, ReformatAttribute extra_attribute) const {
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().var_sanity_check_first_run = false;
    OperatorNodeBase* new_opr;
    /// \note: Concat operators are specially treated. The reasons are as
    /// follows:
    /// 1. Padding the input varnodes of the
    /// Concat opr is not allowed. If we pad the input varnodes of Concat opr, the
    /// padding information of output varnode should be propagated in the layout
    /// selection algorithm. But this feature has not been implemented now.
    /// 2. The axis of concat operator should be modified, because the layouts of the
    /// input varnodes has been modified. So we handle the new axis in the OprMaker
    /// function.
    bool allow_aligned = intl::allow_aligned_layout(opr);
    VarNodeArray new_inps(opr->input().size());
    for (size_t i = 0; i < opr->input().size(); ++i) {
        auto&& var = opr->input(i);
        auto&& cn = var->comp_node();
        auto&& dtype = var->dtype();
        auto dval = std::make_shared<DeviceTensorND>(cn, dtype);
        auto new_shape = ReformatManager::try_make_tensor_shape(
                var, base_format, tensor_format, extra_attribute, allow_aligned);
        if (new_shape.ndim == 0)
            return PROFILE_TIME_OUT;
        dval->resize(new_shape);
        auto new_var = opr::VolatileSharedDeviceTensor::make(*graph, dval);
        new_inps[i] = new_var.node();
    }
    if (intl::has_opr_format_modifier(opr)) {
        intl::OprFormatInfo opr_format_info;
        opr_format_info.tensor_formats = {base_format, tensor_format};
        auto new_var = intl::modify_opr_format(opr_format_info, new_inps, opr);
        if (new_var)
            new_opr = new_var->owner_opr();
        else
            return PROFILE_TIME_OUT;
    } else {
        new_opr = serialization::copy_opr_shallow(
                *opr, new_inps, opr->config(), {graph.get()});
    }
    if (!m_opr_filter(opr, new_opr))
        return PROFILE_TIME_OUT;
    auto y = new_opr->output(0);
    auto mark = MarkInputContiguous::make(SymbolVar(y));
    auto func = graph->compile({{mark, {}}});
    auto filter = [new_opr](OperatorNodeBase* opr) { return opr == new_opr; };
    auto profiler =
            std::make_unique<GraphPartitionProfiler>(graph.get(), std::move(filter));
    for (int i = 0; i < m_runs; ++i)
        func->execute();
    return profiler->duration_in_usec();
}

ProfilerImpl::OperatorNodeRecord ProfilerImpl::profile_operator(
        const OperatorNodeBase* opr, const OprTensorFormatsConfiguration& base_config,
        const SmallVector<OprTensorFormatsConfiguration>& available_configs,
        ReformatAttribute extra_attribute) const {
    OperatorNodeRecord record;
    record.opr = opr;
    auto& costs = record.costs;
    for (auto&& i : available_configs) {
        costs[i.config_id] = profile_operator(opr, base_config, i, extra_attribute);
    }
    return record;
}

float ProfilerImpl::profile_operator(
        const OperatorNodeBase* opr, const OprTensorFormatsConfiguration& base_config,
        const OprTensorFormatsConfiguration& config,
        ReformatAttribute extra_attribute) const {
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().graph_opt.weight_preprocess =
            opr->owner_graph()->options().graph_opt.weight_preprocess;
    graph->options().var_sanity_check_first_run = false;
    VarNodeArray new_inps(opr->input().size());
    size_t i = 0;
    size_t nr_input_tensor =
            std::min(config.input_tensor_formats.size(), opr->input().size());
    for (; i < nr_input_tensor; ++i) {
        auto&& var = opr->input(i);
        TensorShape aligned_shape;
        if (config.input_tensor_types[i] == TensorType::WEIGHT) {
            mgb_assert(base_config.input_tensor_types[i] == TensorType::WEIGHT);
            aligned_shape = ReformatManager::make_aligned_weight_shape(
                    var, base_config.input_tensor_formats[i],
                    config.input_tensor_formats[i], config.output_tensor_formats[0],
                    extra_attribute);
        } else {
            mgb_assert(
                    base_config.input_tensor_types[i] == config.input_tensor_types[i]);
            mgb_assert(base_config.input_tensor_types[i] == TensorType::FEATURE);
            aligned_shape = ReformatManager::make_aligned_tensor_shape(
                    var, base_config.input_tensor_formats[i],
                    config.input_tensor_formats[i], extra_attribute);
        }
        std::shared_ptr<DeviceTensorND> dval = create_device_tensor_helper(
                config, i, var, aligned_shape, extra_attribute);

        if (config.input_tensor_types[i] == TensorType::WEIGHT) {
            new_inps[i] =
                    opr::SharedDeviceTensorWithFormat::make_const(*graph, dval).node();
        } else {
            new_inps[i] = opr::VolatileSharedDeviceTensor::make(*graph, dval).node();
        }
    }
    for (; i < opr->input().size(); ++i) {
        auto&& var = opr->input(i);
        auto&& cn = var->comp_node();
        auto&& dtype = var->dtype();
        auto hval = std::make_shared<HostTensorND>(cn, dtype);
        hval->resize(var->shape());
        auto cb = [&](DeviceTensorND& d) { hval->copy_from(d).sync(); };
        {
            auto cg = var->owner_graph();
            cg->compile({{var, cb}})->execute();
        }
        auto imm = opr::ImmutableTensor::make(*graph, *hval);
        new_inps[i] = imm.node();
    }
    intl::OprFormatInfo opr_format_info;
    opr_format_info.opr_format = config.opr_format;
    VarNode* y = mgb::gopt::intl::modify_opr_format(opr_format_info, new_inps, opr);
    static const ThinHashSet<Typeinfo*> multi_algo_oprs = {
            opr::Convolution::typeinfo(),
            opr::ConvBiasForward::typeinfo(),
            opr::ConvolutionBackwardData::typeinfo(),
            opr::PoolingForward::typeinfo(),
    };
    if (multi_algo_oprs.count(opr->dyn_typeinfo()) &&
        (!mgb::gopt::intl::has_available_algo(new_inps, y->owner_opr()) ||
         !mgb::gopt::intl::has_no_naive_heuristic_algo(new_inps, y->owner_opr())))
        return PROFILE_TIME_OUT;
    if (!m_opr_filter(opr, y->owner_opr()))
        return PROFILE_TIME_OUT;
    auto mark = MarkInputContiguous::make(SymbolVar(y));
    auto func = graph->compile({{mark, {}}});
    auto new_opr = y->owner_opr();
    auto filter = [&new_opr](OperatorNodeBase* opr) { return opr == new_opr; };
    auto profiler =
            std::make_unique<GraphPartitionProfiler>(graph.get(), std::move(filter));
    for (int i = 0; i < m_runs; ++i)
        func->execute();
    return profiler->duration_in_usec();
}

ProfilerImpl::VarNodeRecord ProfilerImpl::profile_var_node(
        const VarNode* var, TensorFormats base_format,
        const SmallVector<TensorFormats>& available_tensor_formats,
        ReformatAttribute attribute) const {
    VarNodeRecord record;
    record.var = var;
    auto& costs = record.costs;
    for (auto&& i : available_tensor_formats) {
        for (auto&& o : available_tensor_formats) {
            if (i == o)
                continue;
            ReformatKey key{
                    i, o, attribute, var->dtype().enumv(), var->dtype().enumv()};
            costs[{i, o}] = profile_var_node(var, base_format, key);
        }
    }
    return record;
}

float ProfilerImpl::profile_var_node(
        const VarNode* var, TensorFormats base_format, const ReformatKey& key) const {
    auto&& cn = var->comp_node();
    auto&& dtype = var->dtype();
    auto aligned_tensor_shape = ReformatManager::make_aligned_tensor_shape(
            var, base_format, key.input_format, key.attribute);

    std::shared_ptr<DeviceTensorND> dval;
    if (key.input_format == TensorFormats::NHCWc4 &&
        key.attribute & ReformatAttribute::IMAGE2D) {
        size_t align_axis = 2;
        auto named_tensor = tensor_formats_to_named_tensor_shape(key.input_format);
        for (size_t n = 0; n < named_tensor.ndim; n++) {
            if (named_tensor[n].name() == megdnn::Dimension::Name::C) {
                align_axis = n;
                break;
            }
        }
        dval = std::make_shared<DeviceTensorND>(
                cn, aligned_tensor_shape, dtype,
                megdnn::Image2DPack4TensorFormat::make(
                        align_axis, opr::intl::get_megdnn_handle(cn)));
    } else
        dval = std::make_shared<DeviceTensorND>(cn, aligned_tensor_shape, dtype);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().var_sanity_check_first_run = false;
    auto aligned_var = opr::VolatileSharedDeviceTensor::make(*graph, dval);
    auto builder = ReformatManager::instance().auto_aligned_reformat_featrue(
            var, base_format, key);
    auto y = builder({aligned_var.node()});

    if (!m_var_node_filter(var, aligned_tensor_shape, y->shape(), key))
        return PROFILE_TIME_OUT;
    ThinHashSet<OperatorNodeBase*> set;
    DepOprIter iter([&set](OperatorNodeBase* opr) { set.insert(opr); });
    iter.add(y->owner_opr());
    iter.set_visited(aligned_var.node()->owner_opr());
    auto mark = MarkInputContiguous::make(SymbolVar(y));
    auto func = graph->compile({{mark, {}}});
    auto filter = [&set](OperatorNodeBase* opr) { return set.count(opr) > 0; };
    auto profiler =
            std::make_unique<GraphPartitionProfiler>(graph.get(), std::move(filter));
    for (int i = 0; i < m_runs; ++i)
        func->execute();
    return profiler->duration_in_usec();
}

ProfilerImpl::ProfilingResult ProfilerImpl::profile(const Problem& problem) const {
    ConstVarPropogate cvprop{ConstVarType::IMMUTABLE_AND_PARAM};
    {
        auto cb = [&cvprop](OperatorNodeBase* opr) { cvprop.add_opr(opr); };
        DepOprIter iter{cb};
        for (auto&& o : problem.graph_partition().output()) {
            iter.add(o->owner_opr());
        }
    }

    static const ThinHashMap<Typeinfo*, size_t> format_aware_input_tensors = {
#define cb(_Opr, _arity) {_Opr::typeinfo(), _arity}
            cb(Convolution, 2),
            cb(ConvBiasForward, 4),
            cb(ConvolutionBackwardData, 2),
            cb(PoolingForward, 1),
            cb(WarpPerspective, 1),
            cb(Resize, 1),
#undef cb
    };
    static const ThinHashSet<Typeinfo*> skip_opr_types = {
            TypeCvt::typeinfo(), Elemwise::typeinfo(), ElemwiseMultiType::typeinfo()};
    ThinHashSet<VarNode*> vars;
    ThinHashSet<OperatorNodeBase*> oprs;
    ThinHashSet<OperatorNodeBase*> skip_oprs;
    for (auto&& opr : problem.graph_partition().all_oprs()) {
        if (cvprop.is_const(opr))
            continue;
        bool skip = true;
        for (auto&& i : opr->input()) {
            skip &= problem.graph_partition().input().count(i) > 0 ||
                    skip_oprs.count(i->owner_opr()) > 0;
        }
        auto find = format_aware_input_tensors.find(opr->dyn_typeinfo());
        skip &= find == format_aware_input_tensors.end();
        if (skip)
            skip_oprs.insert(opr);
        oprs.insert(opr);
        if (find == format_aware_input_tensors.end()) {
            for (auto&& i : opr->input()) {
                if (!cvprop.is_const(i)) {
                    vars.insert(i);
                }
            }
        } else {
            size_t nr_input_tensor = std::min(find->second, opr->input().size());
            for (size_t i = 0; i < nr_input_tensor; ++i) {
                if (!cvprop.is_const(opr->input(i))) {
                    vars.insert(opr->input(i));
                }
            }
        }
        for (auto&& ov : opr->usable_output()) {
            vars.insert(ov);
        }
    }

    auto base_format = problem.base_format();
    auto&& available_tensor_formats = problem.available_tensor_formats();
    auto&& reformat_attribute = problem.attribute().reformat_attribute;

    ProfilingResult profiling_result;
    auto& opr_record = profiling_result.opr_record;
    auto& var_record = profiling_result.var_record;
    for (auto&& var : vars) {
        var_record[var] = profile_var_node(
                var, base_format, available_tensor_formats, reformat_attribute);
    }
    for (auto&& opr : oprs) {
        auto&& opr_configs = problem.opr_configs();
        auto find = opr_configs.find(opr->dyn_typeinfo());
        if (find == opr_configs.end()) {
            if (skip_oprs.count(opr) > 0) {
                SmallVector<TensorFormats> tensor_formats = {base_format};
                opr_record[opr] = profile_operator(
                        opr, base_format, tensor_formats, reformat_attribute);
            } else {
                opr_record[opr] = profile_operator(
                        opr, base_format, available_tensor_formats, reformat_attribute);
            }
        } else {
            auto&& dispatchers = find->second;
            SmallVector<OprTensorFormatsConfiguration> configs;
            for (const auto& item : dispatchers) {
                auto config = (*item.second)(opr);
                if (config.valid()) {
                    configs.emplace_back(config.val());
                }
            }
            auto base_config = problem.base_config(opr);
            opr_record[opr] =
                    profile_operator(opr, base_config, configs, reformat_attribute);
        }
    }
    for (auto&& rpair : opr_record) {
        mgb_log_debug("%s", rpair.second.to_string().c_str());
    }
    for (auto&& rpair : var_record) {
        mgb_log_debug("%s", rpair.second.to_string().c_str());
    }
    return profiling_result;
}

ProfilerImpl::OprFormatConfigID ProfilerImpl::tensor_formats_to_config_id(
        TensorFormats tensor_format) const {
    switch (tensor_format) {
        case TensorFormats::NCHW:
            return OprFormatConfigID::NCHW;
        case TensorFormats::NCHWc4:
            return OprFormatConfigID::NCHW4;
        case TensorFormats::NCHWc8:
            return OprFormatConfigID::NCHW8;
        case TensorFormats::NCHWc32:
            return OprFormatConfigID::NCHW32;
        case TensorFormats::NCHWc64:
            return OprFormatConfigID::NCHW64;
        case TensorFormats::NHWC:
            return OprFormatConfigID::NHWC;
        case TensorFormats::CHWNc4:
            return OprFormatConfigID::CHWN4;
        case TensorFormats::NHCWc4:
            return OprFormatConfigID::NHWCD4;
        default:
            mgb_throw(
                    MegBrainError, "tensor format(%u) is not supported",
                    static_cast<uint32_t>(tensor_format));
    }
}

std::shared_ptr<DeviceTensorND> ProfilerImpl::create_device_tensor_helper(
        const OprTensorFormatsConfiguration& config, const size_t inp_idx,
        const VarNode* var, const TensorShape aligned_shape,
        ReformatAttribute extra_attribute) const {
    auto&& cn = var->comp_node();
    auto&& dtype = var->dtype();
    std::shared_ptr<DeviceTensorND> dval;
    if (config.config_id == OprFormatConfigID::NHWCD4 &&
        extra_attribute & ReformatAttribute::IMAGE2D) {
        size_t align_axis = 2;
        auto named_tensor = tensor_formats_to_named_tensor_shape(
                config.input_tensor_formats[inp_idx]);
        for (size_t n = 0; n < named_tensor.ndim; n++) {
            if (named_tensor[n].name() == megdnn::Dimension::Name::C) {
                align_axis = n;
                break;
            }
        }
        // channel wise weight
        bool is_channel_wise =
                config.input_tensor_formats[inp_idx] == TensorFormats::C1RSc4;
        if (is_channel_wise)
            align_axis = 1;
        dval = std::make_shared<DeviceTensorND>(
                cn, aligned_shape, dtype,
                megdnn::Image2DPack4TensorFormat::make(
                        align_axis, opr::intl::get_megdnn_handle(cn)));
    } else {
        dval = std::make_shared<DeviceTensorND>(cn, aligned_shape, dtype);
    }
    return dval;
}

/* ================== ProfilerBase =================*/
std::string ProfilerBase::OperatorNodeRecord::to_string() const {
    auto str = ssprintf(
            "\nopr type: %s\nopr name: %s\ninputs:\n", opr->dyn_typeinfo()->name,
            opr->cname());
    for (auto&& i : opr->input()) {
        str += ssprintf(
                "\tvar: %s\n\tshape: %s\n", i->cname(), i->shape().to_string().c_str());
    }
    str += ssprintf(
            "outputs:\n\tvar: %s\n\tshape: %s\ncosts:\n", opr->output(0)->cname(),
            opr->output(0)->shape().to_string().c_str());
    for (auto&& cpair : costs) {
        str += ssprintf(
                "\tconfig: %s; cost:%f", config_id_to_string(cpair.first),
                cpair.second);
    }
    return str;
}

std::string ProfilerBase::VarNodeRecord::to_string() const {
    auto str = ssprintf("\nvar: %s\ncosts:", var->cname());
    for (auto&& cpair : costs) {
        auto&& formats = cpair.first;
        str += ssprintf(
                "\n\tformat: (i:%s;o:%s); cost:%f",
                tensor_formats_to_named_tensor_shape(formats.first).to_string().c_str(),
                tensor_formats_to_named_tensor_shape(formats.second)
                        .to_string()
                        .c_str(),
                cpair.second);
    }
    return str;
}

std::unique_ptr<ProfilerBase> ProfilerBase::make_profiler() {
    return std::make_unique<ProfilerImpl>();
}

std::unique_ptr<ProfilerBase> ProfilerBase::make_cached_profiler(const char* path) {
    return std::make_unique<CachedProfiler>(path);
}

/* ================== CachedProfiler =================*/
CachedProfiler::CachedProfiler(
        const char* path, int runs, float opr_threshold, float var_node_threshold)
        : ProfilerImpl(runs, opr_threshold, var_node_threshold), m_path{path} {
    if (m_path != nullptr) {  // file cache
        ProfilerCache::inst().set_impl(std::make_unique<InFilePersistentCache>(m_path));
    }
}

CachedProfiler::ProfilingResult CachedProfiler::profile(const Problem& problem) const {
    auto ret = ProfilerImpl::profile(problem);
    if (m_path != nullptr)
        ProfilerCache::inst().dump_cache(m_path);
    return ret;
}

float CachedProfiler::profile_operator(
        const OperatorNodeBase* opr, TensorFormats base_format,
        TensorFormats tensor_format, ReformatAttribute extra_attribute) const {
    ProfilerCache::Key key{
            opr, tensor_formats_to_config_id(tensor_format), extra_attribute};
    auto ret = ProfilerCache::inst().get(key);
    if (ret.valid())
        return ret.val();
    auto rst = ProfilerImpl::profile_operator(
            opr, base_format, tensor_format, extra_attribute);
    ProfilerCache::inst().put(key, rst);
    return rst;
}

float CachedProfiler::profile_operator(
        const OperatorNodeBase* opr, const OprTensorFormatsConfiguration& base_config,
        const OprTensorFormatsConfiguration& config,
        ReformatAttribute extra_attribute) const {
    ProfilerCache::Key key{opr, config.config_id, extra_attribute};
    auto ret = ProfilerCache::inst().get(key);
    if (ret.valid())
        return ret.val();
    auto rst =
            ProfilerImpl::profile_operator(opr, base_config, config, extra_attribute);
    ProfilerCache::inst().put(key, rst);
    return rst;
}

float CachedProfiler::profile_var_node(
        const VarNode* var, TensorFormats base_format, const ReformatKey& key) const {
    ProfilerCache::Key pf_key{var, key};
    auto ret = ProfilerCache::inst().get(pf_key);
    if (ret.valid())
        return ret.val();
    auto rst = ProfilerImpl::profile_var_node(var, base_format, key);
    ProfilerCache::inst().put(pf_key, rst);
    return rst;
}

// vim: syntax=cpp.doxygen
