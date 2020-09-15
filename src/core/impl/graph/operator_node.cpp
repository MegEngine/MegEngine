/**
 * \file src/core/impl/graph/operator_node.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"

#include "megbrain/comp_node_env.h"

#include "megbrain/graph/event.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/graph/helper.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/execution_mask.h"

#include "megbrain/utils/hash.h"
#include "megbrain/utils/metahelper.h"

#include "megbrain/plugin/var_sanity_check.h"

using namespace mgb;
using namespace cg;

namespace {
class PostExecActions {
    using StaticInferHandlerPtr =
            static_infer::StaticInferManagerImpl::TagHandler*;
    struct Item {
        VarNode* var;
        //! non-null if VarSanityCheck::check_var_after_exec() is needed
        const ComputingGraph::VarReceiverInfo* recv_info = nullptr;
        StaticInferHandlerPtr shape_sync_hdl = nullptr,
                              value_sync_hdl = nullptr;
        //! non-null if on_var_node_device_comp_finish() is needed
        VarNodeMemManager* need_mem_mgr = nullptr;

        bool empty() const {
            return !need_mem_mgr && !shape_sync_hdl && !value_sync_hdl;
        }
    };
    CompNode m_comp_node;
    // VarNodes in m_items should be listed in the same order as in the
    // output of the owner_opr, because opr would generate input_wating_spec()
    // according to this order
    // see `SeqCompNodeOptimizerImpl::init_ready_event()` for more details
    SmallVector<Item> m_items;
    MGB_IF_COND_EXEC(ExecutionMask* m_mask = nullptr);

    void add(VarNode* var);

    void perform();

public:
    PostExecActions(VarNode* var) {
        m_comp_node = var->comp_node();
        add(var);
    }

    PostExecActions(PostExecActions&&) = default;
    PostExecActions& operator=(PostExecActions&&) = default;

    static void process_opr(OperatorNodeBase& opr,
                            OperatorNodeBase::ExecEnv& env);
};
}  // anonymous namespace

inline VarNode::VarNode(Maybe<std::string> name, OperatorNodeBase *owner):
    GraphNodeBase(owner->owner_graph()),
    m_name{std::move(name)},
    m_owner(owner)
{
}

void GraphExecutable::record_execute_deps(ExecDependencyArray&) {}

bool GraphExecutable::ExecDependency::has_runtime_check() const {
    return false;
}

void GraphExecutable::ExecDependency::do_runtime_check() {}

/* ===================== OperatorNodeBase =====================  */

OperatorNodeBase::OperatorNodeBase(ComputingGraph *owner,
                const OperatorNodeConfig &config,
                const std::string &default_name,
                const VarNodeArrayView& input_var_naming):
    GraphNodeBase{owner}, m_config{config}
{
    m_name = config.make_name(default_name, input_var_naming, id());
}

OperatorNodeBase::~OperatorNodeBase() noexcept {
    for (auto i: m_output) {
        owner_graph()->free_varnode(i);
    }
}

void OperatorNodeBase::execute(ExecEnv &env) {
    if (owner_graph()->options().imperative_proxy_graph) {
        do_execute(env);
        return;
    }

    owner_graph()->event().signal_inplace<event::OprExecStart>(this, &env);

    // dispatch waiting commands
    for (auto&& wspec : input_waiting_spec()) {
        auto runner = [ this, ps = &wspec ]() {
            for (VarNode* i : ps->dev_ready) {
                auto&& event = VarNodeMemManager::var_node_cn_sync_manager(i)
                                       ->busy_wait_set_ready_and_get_event();
                ps->comp_node.device_wait_event(event);
            }
            owner_graph()->event().signal_inplace<event::AfterWait>(
                    ps->comp_node, this);
        };
        // always maintain var sync order, so we dispatch without execution mask
        env.dispatch_on_comp_node_with_mask(wspec.comp_node, runner, nullptr);
    }

    // allocate output with dynamic storage
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .alloc_var_node_mem_dynamic(env, this);

    // find shape-dep inputs:
    // shape/value deps whose static infer source is missing are added to
    // DEV_COMP_ORDER dep by topo sorter, so it is guaranteed that static infer
    // would success here. For host-value deps, opr would query
    // static_infer_manager so the value would be up-to-date; however for shape
    // deps, oprs would access the shape directly, so we need to insert some
    // code here to ensure it is up-to-date.
    if (!ComputingGraphImpl::downcast(owner_graph())
                 ->eager_eval_manager()
                 .enabled()) {
        VarNodeArray vars_to_set;
        auto cg = ComputingGraphImpl::downcast(owner_graph());
        auto step_cur = cg->opr_step_num_in_cur_comp_seq(this).val();
        mgb_assert(step_cur < std::numeric_limits<size_t>::max());
        using DT = NodeProp::DepType;
        CompNode uniq_cn;   // all outputs should be on the same comp node
        for (auto &&i: node_prop().dep_map()) {
            if ((i.second & DT::SHAPE) && !(i.second & DT::DEV_VALUE)) {
                auto var = i.first;
                if (!uniq_cn.valid()) {
                    uniq_cn = output(0)->comp_node();
                    for (auto i: output()) {
                        mgb_assert(uniq_cn == i->comp_node(),
                                "opr that has shape dep should be on a "
                                "single comp node; opr=%s{%s}",
                                cname(), dyn_typeinfo()->name);
                    }
                }

                auto vs = cg->opr_step_num_in_cur_comp_seq(var->owner_opr());
                if (!vs.valid() || step_cur < vs.val() ||
                        var->comp_node() != uniq_cn) {
                    vars_to_set.push_back(var);
                }
            }
        }

        if (!vars_to_set.empty()) {
            auto cb = [arr=std::move(vars_to_set)]() {
                auto &&mgr = arr[0]->owner_graph()->static_infer_manager();
                for (auto i: arr)
                    i->shape(mgr.infer_shape(i));
            };
            env.dispatch_on_comp_node(uniq_cn, cb);
        }
    }

    owner_graph()->event().signal_inplace<event::OprExecKernelStart>(
            this, &env);
    do_execute(env);
    owner_graph()->event().signal_inplace<event::OprExecKernelEnd>(
            this, &env);
    PostExecActions::process_opr(*this, env);
    owner_graph()->event().signal_inplace<event::OprExecFinished>(this, &env);
}

const VarNodeArray OperatorNodeBase::usable_output() const {
    VarNodeArray outputs;
    for (auto oup: m_output) {
        if(!oup->contain_flag(cg::VarNode::Flag::VOLATILE_CONTENT)) {
            outputs.push_back(oup);
        }
    }
    return outputs;
}

size_t OperatorNodeBase::hash() const {
    XXHash hstate;
    hstate.update(m_input.data(), sizeof(m_input[0]) * m_input.size());
    size_t extra_size = 2 + m_config.comp_node().size() +
                m_extra_equiv_comp.size(),
           next = 0, extra[extra_size];

    // type info
    extra[next ++] = mgb::hash(dyn_typeinfo());

    // config
    extra[next ++] = m_config.hash();
    for (auto i: m_config.comp_node())
        extra[next ++] = mgb::hash(i);

    // extra
    for (const HashableContainer &i: m_extra_equiv_comp)
        extra[next ++] = mgb::hash(i);

    mgb_assert(next == extra_size);
    hstate.update(extra, sizeof(extra[0]) * extra_size);
    return hstate.digest();
}

bool OperatorNodeBase::is_same_st(const Hashable &rhs_) const {
    auto &&rhs = static_cast<const OperatorNodeBase&>(rhs_);
    if (m_input.size() != rhs.input().size() ||
            m_extra_equiv_comp.size() != rhs.m_extra_equiv_comp.size())
        return false;
    if (!m_config.is_same(rhs.m_config))
        return false;
    for (size_t i = 0; i < m_input.size(); i ++)
        if (m_input[i] != rhs.input()[i])
            return false;
    for (size_t i = 0; i < m_extra_equiv_comp.size(); i ++)
        if (!m_extra_equiv_comp[i].is_same(rhs.m_extra_equiv_comp[i]))
            return false;
    return true;
}

void OperatorNodeBase::add_input(
        std::initializer_list<VarNode*> list,
        AddInputSortType sort_type) {
    mgb_assert(!m_inserted_in_graph && !m_node_prop.valid(),
            "add input on an opr that has been inserted into graph");
    auto start_size = m_input.size();
    for (auto ptr: list) {
        mgb_assert(ptr && ptr->owner_graph() == owner_graph(),
                "input(%s) does not belong to same graph", ptr->cname());
        mgb_assert(!ptr->contain_flag(VarNode::Flag::VOLATILE_CONTENT),
                "use input of volatile content: %s",
                cg::dump_var_info({ptr}).c_str());
        m_input.push_back(ptr);
    }
    if (sort_type != AddInputSortType::NONE) {
        auto begin = m_input.begin(), end = m_input.end();
        if (sort_type == AddInputSortType::CUR_ADDED)
            begin += start_size;
        auto cmp = [](VarNode *a, VarNode *b) {
            return a->id() < b->id();
        };
        small_sort(begin, end, cmp);
    }
}

VarNode* OperatorNodeBase::add_output(const Maybe<std::string> &name) {

    mgb_assert(!m_inserted_in_graph && !m_node_prop.valid(),
            "add output on opr after it has been inserted into graph");

    auto ptr = owner_graph()->alloc_varnode(
                name.valid() ? this->name() + ":" + name.val() : name, this);
    m_output.push_back(ptr);
    return ptr;
}

const OperatorNodeBase::NodeProp& OperatorNodeBase::node_prop() const {
    if (!m_node_prop.valid()) {
        MGB_TRY {
            auto ret = do_make_node_prop();
            mgb_assert(ret == &m_node_prop.val());
        }
        MGB_CATCH(..., {
            m_node_prop.invalidate();
            throw;
        });
        update_priority();
#if !MGB_BUILD_SLIM_SERVING
        // check that node prop is valid
        auto&& dep_map = m_node_prop->dep_map();
        for (auto&& i : dep_map) {
            mgb_assert(i.first->owner_graph() == owner_graph());
            mgb_assert(find(m_input, i.first) != m_input.end(),
                       "dep map entry not in input var: %s",
                       cg::dump_var_info({i.first}).c_str());
            using DT = NodeProp::DepType;
            mgb_assert(!(i.second & DT::HOST_VALUE_DYNOUT) ||
                               (i.second & DT::HOST_VALUE),
                       "HOST_VALUE_DYNOUT must be used with HOST_VALUE");
        }
        for (auto i : input()) {
            mgb_assert(dep_map.count(i), "input var not in dep map: %s",
                       cg::dump_var_info({i}).c_str());
        }
#endif
    }
    return m_node_prop.val();
}

void OperatorNodeBase::init_output_dtype() {
    bool need_dtype = false;
    for (auto i: output()) {
        if (!i->dtype().valid()) {
            need_dtype = true;
            break;
        }
    }
    if (!need_dtype)
        return;

    mgb_assert(!input().empty());
    DType dtype;
    for (size_t i = 0; i < input().size(); ++ i) {
        if (!i)
            dtype = input(i)->dtype();
        else {
            mgb_assert(dtype == input(i)->dtype(),
                    "get different dtypes for input: %s vs %s",
                    dtype.name(), input(i)->dtype().name());
        }
    }
    mgb_assert(dtype.valid() && dtype != dtype::Byte());
    for (auto i: output()) {
        if (!i->dtype().valid())
            i->dtype(dtype);
    }
}

void OperatorNodeBase::init_output_format() {
    TensorFormat format, default_;
    for (auto i : input()) {
        auto cur = i->format();
        if (cur != default_) {
            if (format == default_) {
                format = cur;
            } else {
                mgb_assert(format == cur,
                           "multiple non-default formats in inputs: %s vs %s",
                           format.to_string().c_str(), cur.to_string().c_str());
            }
        }
    }
    for (auto i : output()) {
        if (i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            i->format(default_);
        } else {
            i->format(format);
        }
    }
}

void OperatorNodeBase::init_output_mem_plan(bool dynamic) {
    for (auto i: m_output) {
        if (is_static_var_storage(i) == !dynamic &&
                !i->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC))
            i->init_mem_plan();
    }
}

OperatorNodeBase::NodeProp* OperatorNodeBase::do_make_node_prop() const {
    auto ret = &m_node_prop.emplace();
    for (auto &&i: input())
        ret->add_dep_type(i, NodeProp::DepType::DEV_VALUE);
    return ret;
}

bool OperatorNodeBase::update_priority() const {
    if (output().size() == 1 && m_output[0]->contain_flag(
                VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
        // set PERSISTENT_DEVICE_VALUE vars to highest priority
        node_prop().attribute().priority = std::numeric_limits<decltype(
                NodeProp::Attribute::priority)>::min();
        return true;
    }
    return false;
}

OperatorNodeBase::OprEventCallback OperatorNodeBase::get_opr_event_callback() {
    return {};
}

void OperatorNodeBase::do_add_equivalence_component(
        HashableContainer &&hashable) {
    mgb_assert(!m_inserted_in_graph);
    m_extra_equiv_comp.emplace_back(std::move(hashable));
}

#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> OperatorNodeBase::to_json() const {
    auto cvt_var_array = [](const VarNodeArray &arr) {
        auto rst = json::Array::make();
        for (auto i: arr)
            rst->add(json::String::make(i->id_str()));
        return rst;
    };

    auto objptr = json::Object::make();
    auto &&obj = *objptr;
    obj["node_type"] = json::String::make("operator");
    obj["id"] = json::String::make(id_str());
    obj["name"] = json::String::make(name());
    obj["type"] = json::String::make(dyn_typeinfo()->name);
    obj["input"] = cvt_var_array(input());
    obj["output"] = cvt_var_array(output());
    obj["extra"] = to_json_extra_json;

    if (m_input_waiting_spec.valid()) {
        auto wpair_ptr = json::Object::make();
        obj["waiting_spec"] = wpair_ptr;

        auto &&wpair = *wpair_ptr;
        for (auto &&i: m_input_waiting_spec.val()) {
            wpair[i.comp_node.to_string()] = json::Object::make({
                    {"dev_ready", cvt_var_array(i.dev_ready)}});
        }
    } else {
        obj["waiting_spec"] = json::Null::make();
    }
    return objptr;
}
#endif


/* ===================== OperatorNodeConfig =====================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(OperatorNodeConfig);

OperatorNodeConfig::~OperatorNodeConfig() = default;

std::string OperatorNodeConfig::make_name(std::string default_name,
        const VarNodeArrayView &input_var, size_t opr_id) const {
    if (m_name.valid())
        return m_name.val();
    auto &&rst = default_name;
#if !MGB_BUILD_SLIM_SERVING
    if (input_var.size()) {
        rst.append("(");
        bool first = true;
        for (auto i: input_var) {
            if (first)
                first = false;
            else
                rst.append(",");

            if (rst.length() >= 64) {
                rst.append("...");
                break;
            }

            if (!i) {
                rst.append("<null>");
                continue;
            }

            std::string sn = i->name();

            // remove the outermost bracket
            size_t begin = sn.find('('), end = std::string::npos;
            if (begin != end) {
                end = sn.rfind(')');
                sn.erase(begin, end - begin + 1);
            }
            rst.append(sn);
        }
        rst.append(")");
    }
#endif // MGB_BUILD_SLIM_SERVING
    rst.append(ssprintf("[%zu]", opr_id));
    return rst;
}

OperatorNodeConfig& OperatorNodeConfig::comp_node(const CompNode &node) {
    m_comp_node.resize(1);
    m_comp_node[0] = node;
    return *this;
}

OperatorNodeConfig& OperatorNodeConfig::comp_node_arr(
        const CompNodeArray &arr) {
    m_comp_node = arr;
    return *this;
}

size_t OperatorNodeConfig::hash() const {
    return hash_pair_combine(
            hash_pair_combine(m_instance_id_hashed, mgb::hash(m_comp_node)),
            mgb::hash(m_output_dtype.handle()));
}

bool OperatorNodeConfig::is_same_st(const Hashable &rhs_) const {
    auto &&rhs = static_cast<const OperatorNodeConfig&>(rhs_);
    return m_comp_node == rhs.m_comp_node &&
           m_instance_id_hashed == rhs.m_instance_id_hashed &&
           m_output_dtype == rhs.m_output_dtype;
}

CompNode OperatorNodeConfig::get_single_comp_node() const {
    mgb_assert(m_comp_node.size() <= 1,
            "at most one comp node could be provided, got %zu instead",
            m_comp_node.size());
    if (m_comp_node.empty())
        return {};
    return m_comp_node[0];
}

OperatorNodeConfig& OperatorNodeConfig::output_dtype(DType dtype) {
    m_output_dtype = dtype;
    return *this;
}

/* ===================== NodeProp =====================  */
void OperatorNodeBase::NodeProp::reset_dep_type(const VarNodeArray &vars,
        const SmallVector<DepType> &dep_types) {
    mgb_assert(vars.size() == dep_types.size());
    m_dep_map.clear();
    for (size_t i = 0; i < vars.size(); ++ i)
        add_dep_type(vars[i], dep_types[i]);
}

/* ===================== mixins::SingleCNOperatorNode =====================  */

void mixin::SingleCNOperatorNode::mixin_init_output_comp_node(
        OperatorNodeBase& opr) {
    auto cn = mixin_infer_output_comp_node(
            opr,
            opr.node_prop().contain(NodeProp::Flag::CROSS_COMP_NODE_MEMORY));
    for (auto i : opr.output())
        i->comp_node(cn);
    opr.on_output_comp_node_stream_changed();
}

CompNode mixin::SingleCNOperatorNode::mixin_infer_output_comp_node(
        const OperatorNodeBase& opr, bool cross_mem) {
    CompNode cn = opr.config().get_single_comp_node();
    bool infer_from_input = !cn.valid();
    for (auto&& i : opr.input()) {
        CompNode cur = i->comp_node();
        if (infer_from_input && !cn.valid())
            cn = cur;
        if (!cross_mem) {
            mgb_assert(cn.mem_node() == cur.mem_node(),
                       "opr %s{%s} requires all input to be on the same memory "
                       "node of its output; expect=%s cur_var=%s cur_cn=%s",
                       opr.cname(), opr.dyn_typeinfo()->name,
                       cn.to_string().c_str(), i->cname(),
                       cur.to_string().c_str());
        }
        if (infer_from_input) {
            mgb_assert(cn == cur,
                       "comp_node of opr %s{%s} should be inferred from input, "
                       "but different input comp_nodes found: %s vs %s",
                       opr.cname(), opr.dyn_typeinfo()->name,
                       cn.to_string().c_str(), cur.to_string().c_str());
        }
    }
    mgb_throw_if(!cn.valid(), GraphError,
                 "could not infer comp node for opr %s{%s}", opr.cname(),
                 opr.dyn_typeinfo()->name);
    return cn;
}

void mixin::SingleCNOperatorNode::mixin_comp_node(
        OperatorNodeBase &opr, CompNode node) {
    mgb_assert(!m_comp_node.valid() && node.valid());
    m_comp_node = node;
    for (auto i: opr.output())
        i->comp_node(node);
}

OperatorNodeBase::NodeProp*
mixin::SingleCNOperatorNode::mixin_do_make_node_prop(
        const OperatorNodeBase &opr) const {
    auto ret = opr.OperatorNodeBase::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::SINGLE_COMP_NODE);
    return ret;
}

void mixin::SingleCNOperatorNode::mixin_do_execute(
        OperatorNodeBase &opr, ExecEnv &env) {
    auto runner = [this, &opr]() {
        opr.owner_graph()->event().signal_inplace<event::BeforeKernel>(
                &opr, m_comp_node);
        m_comp_node.activate();
        scn_do_execute();
        opr.owner_graph()->event().signal_inplace<event::AfterKernel>(
                &opr, m_comp_node);
    };
    for (auto i: opr.output())
        mgb_assert(i->comp_node() == m_comp_node);
    env.dispatch_on_comp_node(m_comp_node, runner);
}

void mixin::SingleCNOperatorNode::mixin_on_output_comp_node_stream_changed(
        OperatorNodeBase &opr) {
    m_comp_node = opr.output(0)->comp_node();
    for (auto i: opr.output()) {
        if (i->comp_node() != m_comp_node) {
            mgb_assert(i->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
            i->comp_node(m_comp_node);
        }
    }
}

/* =================== mixins::OutshapePureByInshapeOpr ===================  */

mixin::OutshapePureByInshapeOpr::~OutshapePureByInshapeOpr() = default;

void mixin::OutshapePureByInshapeOpr::mixin_set_nr_managed_outputs(
        OperatorNodeBase &opr, size_t nr) {
    mgb_assert(!m_nr_managed_outputs && nr && nr <= opr.output().size());
    m_nr_managed_outputs = nr;
}

void mixin::OutshapePureByInshapeOpr::mixin_init_output_static_infer_desc(
        OperatorNodeBase &opr) {
    if (!m_nr_managed_outputs) {
        m_nr_managed_outputs = opr.output().size();
    } else {
        mgb_assert(m_nr_managed_outputs <= opr.output().size());
    }
    using namespace std::placeholders;
    using namespace cg::static_infer;

    m_out_shp.resize(m_nr_managed_outputs);
    auto &&mgr = opr.owner_graph()->static_infer_manager();

    DepVal dep;
    for (auto i: opr.input())
        dep.push_back({i, DepType::SHAPE});

    for (size_t i = 0; i < m_nr_managed_outputs; ++ i) {
        mgr.register_shape_infer(opr.output(i), {
                dep.empty() ? SourceType::CONSTANT : SourceType::DEP, dep,
                std::bind(&OutshapePureByInshapeOpr::infer_desc,
                        this, i, _1, _2)});
    }
}

bool mixin::OutshapePureByInshapeOpr::infer_desc(size_t out_idx,
        TensorShape &dest, const StaticInferInpVal &inp) {
    if (inp.run_id != m_inp_run_id) {
        TensorShapeArray inp_shp(inp.val.size());
        for (size_t i = 0; i < inp_shp.size(); ++ i)
            inp_shp[i] = inp.val[i].shape();
        get_output_var_shape(inp_shp, m_out_shp);
        mgb_assert(m_out_shp.size() == m_nr_managed_outputs);
        m_inp_run_id = inp.run_id;
    }
    dest = m_out_shp.at(out_idx);
    return true;
}

/* =================== mixins::IOSameShapeOperatorNode ===================  */

void mixin::IOSameShapeOperatorNode::get_output_var_shape(
        const TensorShapeArray &inp_shape, TensorShapeArray &out_shape) const {

    auto &&shp = inp_shape[0];
    for (auto &&i: inp_shape) {
        mgb_assert(shp.eq_shape(i),
                "get different input shapes: prev=%s cur=%s",
                shp.to_string().c_str(), i.to_string().c_str());
    }
    for (auto &&i: out_shape)
        i = shp;
}

/* ===================== PostExecActions =====================  */

void PostExecActions::add(VarNode* var) {
    mgb_assert(m_comp_node == var->comp_node());
    auto graph = ComputingGraphImpl::downcast(var->owner_graph());

    auto&& infer_mgr = graph->static_infer_manager_impl();
    auto&& extra_info = graph->current_comp_seq_extra_info();
    Item item;
    if (graph->var_node_mem_manager().on_var_node_device_comp_finish_needed(
                var)) {
        item.need_mem_mgr = &graph->var_node_mem_manager();
    }
    if (extra_info.missing_for_shape.count(var))
        item.shape_sync_hdl = infer_mgr.get_tag_handler_for_shape(var);
    if (extra_info.missing_for_value.count(var))
        item.value_sync_hdl = infer_mgr.get_tag_handler_for_value(var);

    if (!item.empty() || !cg::is_static_var_storage(var)) {
        item.var = var;

        // always check with recv_info since it incurs no additional cost
        // note: if item.empty() && !is_static_storage, then we only check
        // recv_info
        item.recv_info = &graph->var_receiver_in_current_comp_seq(var);

        m_items.push_back(item);
    }
}

void PostExecActions::perform() {
    bool enable = true;
    MGB_IF_COND_EXEC(enable = m_mask ? m_mask->enabled() : true);

    for (auto&& i : m_items) {
        if (enable) {
            VarSanityCheck::check_var_after_exec(i.var, *i.recv_info);

            if (i.shape_sync_hdl)
                i.shape_sync_hdl->sync_from_var();
            if (i.value_sync_hdl)
                i.value_sync_hdl->sync_from_var();
        }

        if (i.need_mem_mgr) {
            i.need_mem_mgr->on_var_node_device_comp_finish(i.var, enable);
        }
    }
}

void PostExecActions::process_opr(OperatorNodeBase& opr,
                                  OperatorNodeBase::ExecEnv& env) {
    // PostExecActions should be empty most of the time; so we store the objects
    // directly in a SmallVector and copy them to a shared_ptr when non-empty
    SmallVector<PostExecActions> actions;
    for (auto i : opr.output()) {
        bool found = false;
        for (auto&& j : actions) {
            if (j.m_comp_node == i->comp_node()) {
                j.add(i);
                found = true;
                break;
            }
        }
        if (!found) {
            actions.emplace_back(i);
        }
    }

    for (auto&& i : actions) {
        if (!i.m_items.empty()) {
            auto cn = i.m_comp_node;
            MGB_IF_COND_EXEC(i.m_mask = ExecutionMask::get_from_opr(&opr));
            auto cb = [action = std::make_shared<PostExecActions>(
                               std::move(i))]() {
                action->perform();
            };
            env.dispatch_on_comp_node_with_mask(cn, cb, nullptr);
        }
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
