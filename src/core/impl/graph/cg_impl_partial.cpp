/**
 * \file src/core/impl/graph/cg_impl_partial.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl_partial.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include <algorithm>

using namespace mgb;
using namespace cg;

#if MGB_ENABLE_PARTIAL_EXECUTION

/* ======================== ExecOrderChecker ======================== */
class ComputingGraphImpl::MultiPartCompiler::ExecOrderChecker final
        : public std::enable_shared_from_this<ExecOrderChecker>,
          public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    const size_t m_nr_part;
    size_t m_next = 0;
    SmallVector<ComputingGraph*> m_graphs;
    SmallVector<Maybe<DeviceTensorND>*> m_dev_tensors;

    void on_exec(size_t part) {
        if (!part) {
            // cancelling current execution is always allowed
            m_next = 1;

            // wait for all the funcs when starting a new execution to check
            // async error
            for (auto i : m_graphs) {
                auto func = i->current_comp_seq();
                mgb_assert(func);
                func->wait();
            }

            // clean alive tensors
            for (auto i : m_dev_tensors) {
                if (i->valid()) {
                    i->invalidate();
                }
            }
            return;
        }
        mgb_throw_if(part != m_next, GraphError,
                     "multi-part func: expected to execute part %zu, actual "
                     "part is %zu. Total number of parts: %zu",
                     m_next, part, m_nr_part);
        m_next = m_next == m_nr_part - 1 ? 0 : m_next + 1;
    }

public:
    explicit ExecOrderChecker(size_t nr_part) : m_nr_part{nr_part} {}

    void register_to_graph(ComputingGraph& graph, size_t part);

    void record_dev_tensor(Maybe<DeviceTensorND>* ptr) {
        m_dev_tensors.push_back(ptr);
    }

    static ExecOrderChecker* get_from_graph(ComputingGraph& graph) {
        auto ret = graph.options().user_data.get_user_data<ExecOrderChecker>();
        mgb_assert(ret.second == 1);
        return ret.first[0];
    }
};
MGB_TYPEINFO_OBJ_IMPL(ComputingGraphImpl::MultiPartCompiler::ExecOrderChecker);

void ComputingGraphImpl::MultiPartCompiler::ExecOrderChecker::register_to_graph(
        ComputingGraph& graph, size_t part) {
    auto cb = [this, part](const event::CompSeqExecBeforeStart&) {
        on_exec(part);
    };
    graph.event().register_receiver_permanent<event::CompSeqExecBeforeStart>(
            cb);
    m_graphs.push_back(&graph);
    graph.options().user_data.add_user_data(shared_from_this());
}

/* ======================== ShapeProvider ======================== */
MGB_DEFINE_OPR_CLASS(ComputingGraphImpl::MultiPartCompiler::ShapeProvider,
                           cg::SingleCNOperatorNodeBase) // {
    TensorShape m_shape;

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto infer_shp = [this](TensorShape& dest, const InpVal&) -> bool {
            dest = m_shape;
            return m_shape.ndim;
        };
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), {SourceType::MUTABLE, {}, infer_shp});
    }

    void scn_do_execute() override {}

public:
    ShapeProvider(ComputingGraph& graph, const TensorShape& shape, DType dtype,
                  const OperatorNodeConfig& config)

            : Super(&graph, config, "shape_provider", {}), m_shape{shape} {
        mgb_assert(config.has_comp_node_set(),
                   "comp node must be set in config for ShapeProvider");
        add_output(None)
                ->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
                .dtype(dtype);
        comp_node(config.get_single_comp_node());
        add_equivalence_component<ScalarHash<void*>>(this);
    }

    static ShapeProvider* make(ComputingGraph& graph, const TensorShape& shape,
                               DType dtype, const OperatorNodeConfig& config) {
        auto opr = graph.insert_opr(
                std::make_unique<ShapeProvider>(graph, shape, dtype, config));
        return &opr->cast_final_safe<ShapeProvider>();
    }

    //! update shape
    void shape(const TensorShape& shape) { m_shape = shape; }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(
        ComputingGraphImpl::MultiPartCompiler::ShapeProvider);

/* ======================== DeviceDataProvider ======================== */
MGB_DEFINE_OPR_CLASS(
        ComputingGraphImpl::MultiPartCompiler::DeviceDataProvider,
        cg::SingleCNOperatorNodeBase) // {
    bool m_recorded_in_checker = false;
    Maybe<DeviceTensorND> m_value;

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto infer_shp = [this](TensorShape& dest, const InpVal&) -> bool {
            if (!m_value.valid()) {
                return false;
            }
            dest = m_value->shape();
            return true;
        };
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), {SourceType::MUTABLE, {}, infer_shp});
    }

    void scn_do_execute() override {
        output(0)->reset_dev_tensor_from_tensor(m_value.val());
        m_value.invalidate();
    }

public:
    DeviceDataProvider(ComputingGraph& graph, DType dtype,
                       const OperatorNodeConfig& config)

            : Super(&graph, config, "device_value", {}) {
        mgb_assert(config.has_comp_node_set(),
                   "comp node must be set in config for DeviceDataProvider");
        add_output(None)
                ->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
                .dtype(dtype);
        comp_node(config.get_single_comp_node());
        add_equivalence_component<ScalarHash<void*>>(this);
    }

    static DeviceDataProvider* make(ComputingGraph& graph, DType dtype,
                                    const OperatorNodeConfig& config) {
        auto opr = graph.insert_opr(
                std::make_unique<DeviceDataProvider>(graph, dtype, config));
        return &opr->cast_final_safe<DeviceDataProvider>();
    }

    //! update value
    void value(DeviceTensorND value) {
        m_value.emplace(std::move(value));
        if (!m_recorded_in_checker) {
            ExecOrderChecker::get_from_graph(*owner_graph())
                    ->record_dev_tensor(&m_value);
        }
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(
        ComputingGraphImpl::MultiPartCompiler::DeviceDataProvider);

/* ======================== EmptyExecuteOpr ======================== */
MGB_DEFINE_OPR_CLASS(
        ComputingGraphImpl::MultiPartCompiler::EmptyExecuteOpr,
        cg::SingleCNOperatorNodeBase) // {
    void init_output_static_infer_desc() override {
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), static_infer::ShapeInferDesc::make_const({}));
    }

    void scn_do_execute() override {}

public:
    EmptyExecuteOpr(ComputingGraph& graph, const OperatorNodeConfig& config)
            : Super(&graph, config, "empty_exec", {}) {
        mgb_assert(config.has_comp_node_set());
        add_output(None)
                ->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                .dtype(dtype::Byte{});
        comp_node(config.get_single_comp_node());
    }

    static SymbolVar make(ComputingGraph& graph,
                          const OperatorNodeConfig& config) {
        return graph
                .insert_opr(std::make_unique<EmptyExecuteOpr>(graph, config))
                ->output(0);
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(
        ComputingGraphImpl::MultiPartCompiler::EmptyExecuteOpr);

/* ======================== VarSinkOpr ======================== */
MGB_DEFINE_OPR_CLASS(ComputingGraphImpl::MultiPartCompiler::VarSinkOpr,
                           OperatorNodeBase) // {
public:
    using DepTypeList = SmallVector<DepType>;

    /*!
     * \brief a callback that would be invoked when var value changes
     *
     * This first argument is the source var whose value changes.
     *
     * The two DeviceTensorND arguments would be non-null if the dependency
     * types include DEV_VALUE and HOST_VALUE, respectively
     */
    using ValueListener = thin_function<void(VarNode*, const DeviceTensorND*,
                                             const DeviceTensorND*)>;

    VarSinkOpr(const VarNodeArray& inp, const DepTypeList& deps)
            : Super(inp[0]->owner_graph(), {}, "var_sink", {}),
              m_inp_dep_type{deps} {
        mgb_assert(inp.size() == deps.size());
        for (size_t i = 0; i < inp.size(); ++i) {
            auto var = inp[i];
            // insert the entry in value listeners
            m_value_listeners[var];
            add_input({var});
            if (deps[i] & DepType::DEV_VALUE) {
                var->add_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC);
            }
        }
        add_output(None)
                ->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                .add_flag(VarNode::Flag::VOLATILE_CONTENT)
                .dtype(dtype::Byte{});
    }

    void add_value_listener(VarNode* var, ValueListener func) {
        m_value_listeners.at(var).emplace_back(std::move(func));
    }

    static VarSinkOpr* make(const VarNodeArray& inp, const DepTypeList& deps) {
        mgb_assert(!inp.empty());
        return &inp[0]->owner_graph()
                        ->insert_opr(std::make_unique<VarSinkOpr>(inp, deps))
                        ->cast_final_safe<VarSinkOpr>();
    }

    //! make a var in another graph that updates according to given \p old_var
    VarNode* make_var_and_add_listener(ComputingGraph& graph, VarNode* old_var,
                                       DepType dep_type);

private:
    DepTypeList m_inp_dep_type;
    ThinHashMap<VarNode*, SmallVector<ValueListener>> m_value_listeners;

    void on_output_comp_node_stream_changed() override {}

    void init_output_comp_node() override {
        output(0)->comp_node(input(0)->comp_node());
    }

    void init_output_static_infer_desc() override {
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), static_infer::ShapeInferDesc::make_const({}));
    }

    NodeProp* do_make_node_prop() const override {
        auto ret = Super::do_make_node_prop();
        ret->reset_dep_type(input(), m_inp_dep_type);
        ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY)
                .add_flag(NodeProp::Flag::NO_INPUT_WAITING);
        return ret;
    }

    void do_execute(ExecEnv& env) override {
        for (auto&& dep_entry : node_prop().dep_map()) {
            auto var = dep_entry.first;
            auto type = dep_entry.second;
            auto runner = [this, var, type]() {
                const DeviceTensorND* dv_ptr = nullptr;
                const DeviceTensorND* hv_ptr = nullptr;
                if (type & DepType::DEV_VALUE) {
                    dv_ptr = &var->dev_tensor();
                }
                if (type & DepType::HOST_VALUE) {
                    hv_ptr = &owner_graph()->static_infer_manager().infer_value(
                            var);
                }
                for (auto&& i : m_value_listeners.at(var)) {
                    i(var, dv_ptr, hv_ptr);
                }
            };
            env.dispatch_on_comp_node(var->comp_node(), runner);
        }
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ComputingGraphImpl::MultiPartCompiler::VarSinkOpr);

VarNode*
ComputingGraphImpl::MultiPartCompiler::VarSinkOpr::make_var_and_add_listener(
        ComputingGraph& graph, VarNode* old_var, DepType dep_type) {
    VarNode* new_var;
    ValueListener listener;
    auto new_name = ssprintf("fwd_var%zu(%s)", old_var->id(), old_var->cname());
    if (dep_type & DepType::HOST_VALUE) {
        auto host_val = std::make_shared<HostTensorND>(
                old_var->comp_node(), TensorShape{}, old_var->dtype());
        new_var = opr::Host2DeviceCopy::make(graph, host_val, new_name).node();
        listener = [host_val](VarNode*, const DeviceTensorND*,
                              const DeviceTensorND* new_hv) {
            mgb_assert(new_hv);
            if (new_hv->layout().is_contiguous()) {
                HostTensorStorage storage;
                storage.reset(new_hv->comp_node(), new_hv->storage().size(),
                              new_hv->storage().raw_storage());
                host_val->reset(storage, new_hv->layout());
            } else {
                host_val->copy_from(*new_hv);
            }
        };
    } else if (dep_type & DepType::DEV_VALUE) {
        auto data_opr = DeviceDataProvider::make(
                graph, old_var->dtype(),
                OperatorNodeConfig{old_var->comp_node()}.name(new_name));
        new_var = data_opr->output(0);
        listener = [data_opr](VarNode*, const DeviceTensorND* new_dv,
                              const DeviceTensorND*) {
            mgb_assert(new_dv);
            data_opr->value(*new_dv);
        };
    } else if (dep_type & DepType::SHAPE) {
        auto shp_opr = ShapeProvider::make(
                graph, old_var->shape(), old_var->dtype(),
                OperatorNodeConfig{old_var->comp_node()}.name(new_name));
        new_var = shp_opr->output(0);
        listener = [shp_opr](VarNode* var, const DeviceTensorND*,
                             const DeviceTensorND*) {
            shp_opr->shape(var->shape());
        };
    } else {
        mgb_assert(dep_type == DepType::DEV_COMP_ORDER,
                   "unhandled dep type: %d", static_cast<int>(dep_type));
        auto cn = old_var->comp_node();
        new_var = EmptyExecuteOpr::make(graph, cn).node();
        listener = [cn](VarNode* var, const DeviceTensorND*,
                        const DeviceTensorND*) {
            mgb_assert(var->comp_node() == cn);
        };
    }

    add_value_listener(old_var, std::move(listener));
    return new_var;
}

/* ======================== MultiPartCompiler ======================== */

struct ComputingGraphImpl::MultiPartCompiler::PartIOInfo {
    //! vars (in the original graph) in previous parts that are needed by this
    //! part
    ThinHashMap<VarNode*, DepType> inp;
    //! vars (in the original graph) produced in this part that are needed by
    //! future parts
    SmallVector<std::pair<VarNode*, DepType>> out;
    VarSinkOpr* sink_opr = nullptr;

    VarSinkOpr* safe_sink_opr() const {
        mgb_assert(sink_opr);
        return sink_opr;
    }
};

SmallVector<std::unique_ptr<AsyncExecutable>>
ComputingGraphImpl::MultiPartCompiler::compile(
        const SmallVector<OutputSpec>& out_specs) {
    mgb_assert(!out_specs.empty(), "can not deal with empty out specs");
    m_out_specs = out_specs;
    update_out_specs();

    const size_t nr_part = m_out_specs.size();
    SmallVector<std::unique_ptr<AsyncExecutable>> ret;
    ret.reserve(nr_part);
    for (size_t i = 0; i < nr_part; ++i) {
        ret.emplace_back(m_sub_graphs[i]->compile(m_out_specs[i]));
    }
    return ret;
}

void ComputingGraphImpl::MultiPartCompiler::update_out_specs() {
    // 1. Determine the overall computing sequence
    // 2. Determine inter-dependencies between parts
    // 3. Copy oprs of each part into a new graph, and use VarSinkOpr to pass
    //    values between graphs

    auto&& opr_seq = *concat_and_prepare();
    init_opr_trait_and_var_reader_type(opr_seq);

    auto owner_graph = m_out_specs.at(0).at(0).first.node()->owner_graph();

    // var replacement map for current part; cleared for at each part iter
    ThinHashMap<VarNode*, VarNode*> cur_part_var_repl;

    VarNodeArray tmp_new_inputs;
    size_t cur_part;
    serialization::OprShallowCopyContext copy_ctx;
    // copy each opr and record in cur_part_var_repl
    auto on_opr = [&](OperatorNodeBase* opr) {
        VarNodeArray& new_inputs = tmp_new_inputs;
        new_inputs.clear();
        for (auto inp : opr->input()) {
            new_inputs.emplace_back(cur_part_var_repl.at(inp));
        }
        OperatorNodeBase* new_opr = serialization::copy_opr_shallow(
                *opr, new_inputs, opr->config(), copy_ctx);
        auto&& new_dep_map = const_cast<OperatorNodeBase::NodeProp::DepMap&>(
                new_opr->node_prop().dep_map());
        // copy dep entries added by TopoSorter (and maybe others)
        for (auto&& dep_entry : opr->node_prop().dep_map()) {
            if (dep_entry.second == DepType::DEV_COMP_ORDER &&
                !has_different_comp_node(opr, dep_entry.first->comp_node())) {
                continue;
            }
            auto new_var = cur_part_var_repl.at(dep_entry.first);
            auto iter = new_dep_map.find(new_var);
            if (iter == new_dep_map.end()) {
                iter = new_dep_map.insert({new_var, DepType{}}).first;
            }
            if (iter->second != dep_entry.second) {
                mgb_assert((iter->second & dep_entry.second) == iter->second);
                iter->second = dep_entry.second;
            }
        }

        new_opr->node_prop().attribute().priority =
                m_opr_trait.at(opr).priority;

        auto&& out0 = opr->output();
        auto&& out1 = new_opr->output();
        mgb_assert(out0.size() == out1.size());
        for (size_t i = 0; i < out0.size(); ++i) {
            cur_part_var_repl[out0[i]] = out1[i];
        }
    };
    DepOprIter opr_iter{on_opr};

    // map from outer graph var to the var used as VarSinkOpr input (i.e. when
    // it is first produced in a part and recorded in the VarSinkOpr)
    ThinHashMap<VarNode*, VarNode*> var_to_sink_inp_map;
    auto part_io_info = make_part_io_info();
    auto exec_order_checker =
            std::make_shared<ExecOrderChecker>(m_out_specs.size());

    for (cur_part = 0; cur_part < m_out_specs.size(); ++cur_part) {
        cur_part_var_repl.clear();

        // create new graph and set options
        m_sub_graphs.emplace_back(ComputingGraph::make());
        auto cur_graph = m_sub_graphs.back().get();
        cur_graph->share_device_memory_with(*owner_graph);
        assign_graph_opt(cur_graph->options(), owner_graph->options());
        copy_ctx.owner_graph(cur_graph);
        exec_order_checker->register_to_graph(*cur_graph, cur_part);

        // create listeners for sink vars in previous parts needed in this graph
        for (auto&& dep_entry : part_io_info.at(cur_part).inp) {
            if (should_dup_between_part(dep_entry.first)) {
                auto old_opr = dep_entry.first->owner_opr(),
                     new_opr = serialization::copy_opr_shallow(
                             *old_opr, {}, old_opr->config(), copy_ctx);
                cur_part_var_repl[old_opr->output(0)] = new_opr->output(0);
                continue;
            }
            auto orig_graph_var = dep_entry.first,
                 sub_graph_var = var_to_sink_inp_map.at(orig_graph_var);
            auto src_part = m_opr_trait.at(orig_graph_var->owner_opr()).part;
            auto new_var = part_io_info.at(src_part)
                                   .safe_sink_opr()
                                   ->make_var_and_add_listener(
                                           *cur_graph, sub_graph_var,
                                           dep_entry.second);
            cur_part_var_repl[orig_graph_var] = new_var;
        }

        // setup priorities for initial oprs
        for (auto&& i : cur_part_var_repl) {
            i.second->owner_opr()->node_prop().attribute().priority =
                    m_opr_trait.at(i.first->owner_opr()).priority;
        }

        // replace all the oprs
        for (auto&& i : m_out_specs[cur_part]) {
            opr_iter.add(i.first);
            i.first = cur_part_var_repl.at(i.first.node());
        }

        // create VarSinkOpr for vars needed by future parts
        auto&& sink_inputs = tmp_new_inputs;
        sink_inputs.clear();
        VarSinkOpr::DepTypeList sink_inputs_deps;
        for (auto&& i : part_io_info[cur_part].out) {
            if (should_dup_between_part(i.first)) {
                continue;
            }
            auto src_var = i.first,
                 sub_graph_var = cur_part_var_repl.at(src_var);
            auto ins = var_to_sink_inp_map.insert({src_var, sub_graph_var});
            mgb_assert(ins.second);
            sink_inputs.push_back(sub_graph_var);
            sink_inputs_deps.push_back(i.second);
        }
        if (!sink_inputs.empty()) {
            auto sink = VarSinkOpr::make(sink_inputs, sink_inputs_deps);
            part_io_info[cur_part].sink_opr = sink;
            m_out_specs[cur_part].push_back({sink->output(0), {}});
        }
    }
}

SmallVector<ComputingGraphImpl::MultiPartCompiler::PartIOInfo>
ComputingGraphImpl::MultiPartCompiler::make_part_io_info() const {
    const size_t nr_part = m_out_specs.size();
    SmallVector<PartIOInfo> ret;
    ret.reserve(nr_part);
    for (size_t i = 0; i < nr_part; ++i) {
        ret.emplace_back();
    }
    for (auto&& i : m_var_receiver_type) {
        DepType merged_type = DepType{};
        auto var = i.first;
        for (auto&& j : i.second) {
            merged_type |= j.second;
            ret.at(j.first).inp[var] |= j.second;
        }
        ret.at(m_opr_trait.at(var->owner_opr()).part)
                .out.emplace_back(var, merged_type);
    }
    return ret;
}

void ComputingGraphImpl::MultiPartCompiler::init_opr_trait_and_var_reader_type(
        const OprNodeArray& opr_seq) {
    mgb_assert(opr_seq.size() < std::numeric_limits<size_t>::max());
    m_opr_trait.reserve(opr_seq.size());
    // assign priority
    for (size_t i = 0; i < opr_seq.size(); ++i) {
        // use negative number so they can be sorted before new oprs (actually
        // there should not be any new oprs) and have non-zero priority
        m_opr_trait[opr_seq[i]].priority =
                static_cast<ptrdiff_t>(i) -
                static_cast<ptrdiff_t>(opr_seq.size());
    }

    // assign part number and var reader type
    size_t cur_part;
    auto on_dep_entry = [&](VarNode* src_var, DepType dep_type) {
        auto src_part = m_opr_trait[src_var->owner_opr()].part;
        if (static_cast<size_t>(src_part) != cur_part) {
            mgb_assert(src_part != -1 &&
                       static_cast<size_t>(src_part) < cur_part);
            m_var_receiver_type[src_var][cur_part] |= dep_type;
        }
    };
    auto on_opr = [&](OperatorNodeBase* opr) {
        m_opr_trait[opr].part = cur_part;
        for (auto&& entry : opr->node_prop().dep_map()) {
            if (entry.second == DepType::DEV_COMP_ORDER &&
                !has_different_comp_node(opr, entry.first->comp_node())) {
                // ignore DEV_COMP_ORDER deps on the same comp node
            } else {
                on_dep_entry(entry.first, entry.second);
            }
        }
    };
    DepOprIter opr_iter{on_opr};
    for (cur_part = 0; cur_part < m_out_specs.size(); ++cur_part) {
        for (auto&& i : m_out_specs[cur_part]) {
            auto opr = i.first.node()->owner_opr();
            if (opr_iter.visited(opr)) {
                // var in outspec already computed
                on_dep_entry(i.first.node(), DepType::DEV_VALUE);
            } else {
                opr_iter.add(opr);
            }
        }
    }
}

const OprNodeArray*
ComputingGraphImpl::MultiPartCompiler::concat_and_prepare() {
    // no callback in out_spec_concat, so CallbackCaller would not be inserted
    OutputSpec out_spec_concat;
    std::vector<bool> out_spec_concat_from_original;

    // init out_spec_concat
    {
        SymbolVarArray part_vars;
        ExtraDependencyMerger dep_merger;
        for (size_t part = 0; part < m_out_specs.size(); ++part) {
            part_vars.clear();
            for (auto&& i : m_out_specs[part]) {
                part_vars.push_back(i.first);
            }
            auto&& dest_vars = dep_merger.add(part_vars);
            for (size_t i = 0; i < dest_vars.size(); ++i) {
                out_spec_concat.push_back({dest_vars[i].node(), {}});
                out_spec_concat_from_original.push_back(i < part_vars.size());
            }
            dest_vars.clear();
        }
    }

    auto remap_priority = [&](const VarNodeArray& dest_vars,
                              const TopoSorter::PriorityItem* items,
                              size_t nr_item) {
        const size_t nr_part = m_out_specs.size();
        mgb_assert(
                nr_item <= static_cast<size_t>(std::numeric_limits<int>::max()),
                "too many oprs");

        ThinHashMap<const OperatorNodeBase*, size_t> endpoint2part;

        // remap optimized vars to specs and init endpoint2part
        mgb_assert(dest_vars.size() == out_spec_concat.size());
        size_t dest_var_idx = 0;
        auto skip_non_orig_vars = [&](size_t part) {
            while (!out_spec_concat_from_original[dest_var_idx]) {
                ++dest_var_idx;
                if (dest_var_idx == dest_vars.size()) {
                    mgb_assert(part == m_out_specs.size() - 1);
                    break;
                } else {
                    // add extra vars to out spec so we do not need to handle
                    // extra_vardeps in graph copy
                    m_out_specs[part].push_back(
                            {dest_vars[dest_var_idx - 1], {}});
                }
            }
        };
        for (size_t part = 0; part < nr_part; ++part) {
            int begin = dest_var_idx;
            for (auto&& i : m_out_specs[part]) {
                mgb_assert(out_spec_concat_from_original[dest_var_idx]);
                i.first = dest_vars[dest_var_idx++];
            }

            if (dest_var_idx < dest_vars.size()) {
                skip_non_orig_vars(part);
            }

            for (size_t i = begin; i < dest_var_idx; ++i) {
                // use insert so duplicated vars would use the earliest part
                // number
                endpoint2part.insert({dest_vars[i]->owner_opr(), part});
            }
        }
        mgb_assert(dest_var_idx == dest_vars.size());

        SmallVector<size_t> part_last_step(nr_part);
        for (size_t i = 0; i < nr_item; ++i) {
            auto iter = endpoint2part.find(items[i].opr);
            if (iter != endpoint2part.end()) {
                update_max(part_last_step[iter->second], items[i].dfs_step_num);
            }
        }
        for (size_t i = 1; i < part_last_step.size(); ++i) {
            // oprs in the part may have been used in previous parts, so we
            // normalize part_last_step[] for binary search
            part_last_step[i] =
                    std::max(part_last_step[i], part_last_step[i - 1]);
        }
        mgb_assert(part_last_step.back() == nr_item - 1);

        // sort oprs according to (part, original priority)
        using PriKey = std::pair<int, int>;
        SmallVector<std::pair<PriKey, const TopoSorter::PriorityItem*>> oprs(
                nr_item);
        for (size_t i = 0; i < nr_item; ++i) {
            size_t part = std::lower_bound(part_last_step.begin(),
                                           part_last_step.end(),
                                           items[i].dfs_step_num) -
                          part_last_step.begin();
            mgb_assert(part < m_out_specs.size());
            oprs[i] = {{static_cast<int>(part),
                        items[i].opr->node_prop().attribute().priority},
                       items + i};
        }
        std::sort(oprs.begin(), oprs.end());

        int next_pri_num = -1;
        PriKey prev_pri{-1, -1};
        for (auto&& i : oprs) {
            if (i.first != prev_pri) {
                prev_pri = i.first;
                ++next_pri_num;
            }
            *i.second->priority = next_pri_num;
        }
    };

    m_owner->topo_sorter().set_priority_remapper(remap_priority);
    return m_owner->compile_prepare(out_spec_concat).opr_seq;
}

bool ComputingGraphImpl::MultiPartCompiler::should_dup_between_part(
        VarNode* var) {
    // duplicate SharedDeviceTensor so AddUpdate oprs can work as expected
    auto type = var->owner_opr()->dyn_typeinfo();
    return type == opr::SharedDeviceTensor::typeinfo() ||
           type == opr::ImmutableTensor::typeinfo();
}

void ComputingGraphImpl::MultiPartCompiler::assign_graph_opt(
        Options& dst, const Options& src) {
#define S(x) dst.x = src.x
    S(log_level);
    S(async_exec_level);
    S(force_dynamic_alloc);
    S(var_sanity_check_first_run);
    S(allocate_static_mem_after_graph_compile);
    S(enable_var_mem_defragment);
#undef S
    mgb_assert(!src.fake_next_exec && !src.comp_node_seq_record_level);
    dst.graph_opt_level = 0;
    dst.seq_opt.enable_seq_comp_node_opt = false;
}

bool ComputingGraphImpl::MultiPartCompiler::has_different_comp_node(
        OperatorNodeBase* opr, CompNode cn) {
    for (auto i : opr->output()) {
        if (i->comp_node() != cn) {
            return true;
        }
    }
    return false;
}

SmallVector<Typeinfo*>
ComputingGraphImpl::MultiPartCompiler::test_get_internal_opr_types() {
    return {ShapeProvider::typeinfo(), DeviceDataProvider::typeinfo(),
            EmptyExecuteOpr::typeinfo(), VarSinkOpr::typeinfo()};
}

#endif  // MGB_ENABLE_PARTIAL_EXECUTION

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
