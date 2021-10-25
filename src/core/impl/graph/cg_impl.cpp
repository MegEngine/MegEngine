/**
 * \file src/core/impl/graph/cg_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"
#include "./cg_impl_partial.h"
#include "./cg_impl_seq.h"

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/misc.h"
#include "megbrain/graph/cg.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/helper.h"
#include "megbrain/opr/utility.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/opr_replace.h"
#endif

#if MGB_JIT
#include "megbrain/jit/fusion_pass.h"
#endif

using namespace mgb;
using namespace cg;

namespace {
void check_opr_not_cross_mem(OperatorNodeBase* opr) {
    if (opr->node_prop().contain(
                OperatorNodeBase::NodeProp::Flag::CROSS_COMP_NODE_MEMORY))
        return;
    MemNode mem_node_id;
    bool first = true;
    auto check = [&](VarNode* var) {
        auto cur = var->comp_node().mem_node();
        mgb_assert(cur);
        if (first) {
            first = false;
            mem_node_id = cur;
        } else
            mgb_assert(
                    mem_node_id == cur,
                    "for non cross-memory oprs, "
                    "all vars should reside on the same memory node");
    };
    for (auto i : opr->input()) {
        check(i);
    }
    for (auto i : opr->output()) {
        check(i);
    }
}

void update_output_shapes(
        static_infer::StaticInferManagerImpl& infer_mgr, OperatorNodeBase* opr,
        bool add_freeze_flag) {
    for (auto i : opr->output()) {
        if (add_freeze_flag) {
            i->add_flag(VarNode::Flag::FLAG_FREEZED);
        }

        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            using namespace static_infer;
            if (infer_mgr.get_infer_type(i).shape &
                (InferType::CONST | InferType::RT_STATIC)) {
                auto shp = infer_mgr.infer_shape_fallible(i);
                if (shp) {
                    i->shape(*shp);
                } else {
                    i->shape({});
                }
            } else {
                i->shape({});
            }
        }
    }
}

}  // anonymous namespace

/* ========================== global helpers ========================== */
void cg::update_output_var_shapes(OperatorNodeBase* opr) {
    update_output_shapes(
            static_cast<static_infer::StaticInferManagerImpl&>(
                    opr->owner_graph()->static_infer_manager()),
            opr, false);
}

/* ========================= DeviceMemoryAllocator ========================= */
void DeviceMemoryAllocator::alloc_static(
        ComputingGraph*, DeviceTensorStorage& dest, size_t size) {
    dest.ensure_size(size);
}

void DeviceMemoryAllocator::alloc_dynamic(
        VarNode*, DeviceTensorStorage& dest, size_t size) {
    dest.ensure_size(size);
}

void DeviceMemoryAllocator::defrag_prealloc_contig(
        ComputingGraph* graph, CompNode comp_node,
        size_t size){MGB_TRY{comp_node.free_device(comp_node.alloc_device(size));
}
MGB_CATCH(MemAllocError&, {})
}

size_t DeviceMemoryAllocator::static_alloc_version(ComputingGraph*) const {
    return 0;
}

/* ========================== ComputingGraph ========================== */
ComputingGraph::ComputingGraph() {
    static std::atomic_size_t tot_id{0};
    m_id = (tot_id++);
}

void ComputingGraph::assert_destroy(std::shared_ptr<ComputingGraph>& ptr) {
    mgb_assert(
            ptr.use_count() == 1, "unexpected use_count: %zu", size_t(ptr.use_count()));
    ptr.reset();
}

#if !MGB_THREAD_SAFE
size_t ComputingGraph::prealloc_static_storage(size_t size) {
    // note that in single-threaded mode, all cpus map to the same comp node
    static int version = 0;
    auto cn = CompNode::load("cpu0");
    mgb_assert(cn == CompNode::load("cpu1"));
    auto inst = StaticDeviceMemoryManager::make_default_impl();
    auto ret = inst->get_size(cn);
    inst->alloc(nullptr, cn, size, version).ptr();
    version = inst->version(nullptr);
    return ret;
}
#endif

/* ========================== JITConfig ========================== */

bool ComputingGraph::Options::GraphOpt::JITConfig::enabled() const {
    if (fuse_dimshuffle != UNSET)
        return true;
    if (fuse_reduce != UNSET)
        return true;
    return false;
}

void ComputingGraph::Options::GraphOpt::JITConfig::update(const JITConfig& modifier) {
    if (modifier.fuse_dimshuffle != UNSET) {
        this->fuse_dimshuffle = modifier.fuse_dimshuffle;
    }
    if (modifier.fuse_reduce != UNSET) {
        this->fuse_reduce = modifier.fuse_reduce;
    }
}

/* ========================== CallbackCaller ========================== */
MGB_DEFINE_OPR_CLASS(
        ComputingGraphImpl::CallbackCaller, SingleCNOperatorNodeBase) // {
    std::vector<std::vector<ComputingGraph::Callback>> m_cb;

    //! CallbackCaller supports change memory address in output tensor record mode(only
    //! on CPU). The whole callback will be dispatched(like dispatching tensor copy
    //! instead of dispatching memcpy).
    //! Side effect: sync() is not supported in callback anymore. Users should call
    //! func->wait() instead out of callback to sync data from Device to Host.
    //! Note : only record level 1 supports change memory address in output tensor.
    //! HostTensor captured in callback should not on cpu default.
    void scn_do_execute() override {
        for (size_t i = 0; i < input().size(); ++i) {
            auto&& in = input(i)->dev_tensor();
            for (auto&& callback : m_cb[i]) {
                if (this->owner_graph()->options().comp_node_seq_record_level == 1 &&
                    in.comp_node().device_type() == CompNode::DeviceType::CPU &&
                    in.comp_node() != CompNode::default_cpu()) {
                    auto record_cb = [&in, &callback]() {
                        auto comp_node = in.comp_node();
                        bool do_task_inplace = true;
                        comp_node.disable_dispatch(&do_task_inplace);
                        callback(const_cast<DeviceTensorND&>(in));
                        comp_node.enable_dispatch();
                    };
                    in.comp_node().add_callback(record_cb);
                } else {
                    // const cast for backward API compatibility
                    callback(const_cast<DeviceTensorND&>(in));
                }
            }
        }
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), ShapeInferDesc::make_const({}));
    }

    void add_input_layout_constraint() override {
        if (owner_graph()->options().comp_node_seq_record_level) {
            // the user callback usually copies from device to host, which
            // involves tmp alloc if input is not contiguous
            for (auto&& inp : input()) {
                inp->add_layout_constraint_contiguous();
            }
        }
    }

    void init_output_dtype() override {
        if (output(0)->dtype().valid()) {
            return;
        }

        mgb_assert(!input().empty());
        DType dtype = input(0)->dtype();
        mgb_assert(dtype.valid() && dtype != dtype::Byte());
        output(0)->dtype(dtype);
    }

    NodeProp* do_make_node_prop() const override {
        auto ret = Super::do_make_node_prop();
        for (auto&& inp : input()) {
            ret->add_dep_type_existing_var(inp, NodeProp::DepType::VALUE_ALLOW_EMPTY);
        }
        return ret;
    }

    bool update_priority() const override {
        node_prop().attribute().priority = std::numeric_limits<int>::min();
        return true;
    }

public:
    CallbackCaller(const VarNodeArrayView& inp)
            : Super{inp[0]->owner_graph(), {}, "callback", inp} {
        mgb_assert(!inp.empty());
        m_cb.resize(inp.size());
        for (auto&& i : inp) {
            add_input({i});
        }
        using F = VarNode::Flag;
        add_output(None)->add_flag(F::ALLOW_EMPTY_SHAPE).add_flag(F::VOLATILE_CONTENT);
    }

    static SymbolVar make(const VarNodeArrayView& inp) {
        mgb_assert(!inp.empty());
        return SymbolVar{inp[0]}
                .node()
                ->owner_graph()
                ->insert_opr(std::make_unique<CallbackCaller>(inp))
                ->output(0);
    }

    void add_callback(const ComputingGraph::Callback& cb, size_t i = 0) {
        mgb_assert(cb && i < m_cb.size());
        m_cb[i].push_back(cb);
    }

    void clear_callback() {
        for (size_t i = 0; i < m_cb.size(); ++i) {
            m_cb[i].clear();
        }
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ComputingGraphImpl::CallbackCaller);

/* ========================== ComputingGraphImpl ========================== */

ComputingGraphImpl::Components::Components(ComputingGraphImpl* owner)
        : topo_sorter{owner},
          var_node_mem_manager{owner},
          seq_comp_node_opt{owner},
          static_infer_manager{owner},
          static_infer_comp_seq_manager{owner},
          grad_manager{owner},
#if MGB_ENABLE_SUBLINEAR
          seq_modifier_for_sublinear_memory{
                  owner, &(owner->options().sublinear_mem_config)},
#endif
#if MGB_ENABLE_DTR
          seq_modifier_for_dtr{owner, &(owner->options().dtr_config)},
#endif
#if MGB_ENABLE_MEMORY_SWAP
          memory_swap_support{owner},
#endif
          eager_eval_manager{owner}

{
}

ComputingGraphImpl::ComputingGraphImpl() {
    auto ptr = new (&m_components_storage) Components{this};
    mgb_assert(ptr == &components());
}

ComputingGraphImpl::~ComputingGraphImpl() {
    if (!is_finalized()) {
        cleanup();
    }
}

std::shared_ptr<void> ComputingGraphImpl::on_comp_node_finalize() {
    // hold a reference because the object itself may be deleted by user data or
    // oprs
    std::shared_ptr<void> ref = shared_from_this();
    cleanup();
    return ref;
}

void ComputingGraphImpl::cleanup() {
    if (m_recorded_seq_level2_dtor_chk) {
        m_recorded_seq_level2_dtor_chk->enable();
    }
    // clear device memory storage and return them to comp node
    clear_device_memory();

    // so opr dtors would incur no overhead when deleting vars
    m_var_node_pool.disable_freelist();

    // TODO: call this after each graph exec when we have faster impl
    CompNode::try_coalesce_all_free_memory();

    options().user_data.clear_all_user_data();
    components().~Components();
    m_var_receiver.clear();
    m_opr_refkeeper.clear();
}

void* ComputingGraphImpl::alloc_varnode_storage() {
    return m_var_node_pool.alloc_raw();
};

void ComputingGraphImpl::free_varnode_storage(void* ptr) {
    m_var_node_pool.free_raw(ptr);
};

OperatorNodeBase* ComputingGraphImpl::insert_opr(
        std::unique_ptr<OperatorNodeBase> opr_uniqp) {
    auto opr = opr_uniqp.get();

    if (options().imperative_proxy_graph) {
        if (!opr->inserted_in_graph()) {
            m_opr_refkeeper.emplace_back(std::move(opr_uniqp));
            opr->set_inserted_in_graph();
            opr->init_output_comp_node();
            opr->init_output_dtype();
            opr->init_output_format();
            // register static infer
            {
                auto&& mgr = static_infer_manager_impl();
                auto old = mgr.set_register_allowed_opr(opr);
                opr->init_output_static_infer_desc();
                mgr.set_register_allowed_opr(old);
            }
        }
        return opr;
    }
    if (opr->inserted_in_graph()) {
        // FIXME: it's just a trick used for re-evaluation in eager evaluation
        // mode. Since comp_graph has already taken an ownership of the opr,
        // we can release it directly.
        mgb_throw_if(
#if MGB_BUILD_SLIM_SERVING
                true,
#else
                !options().eager_evaluation,
#endif
                GraphError,
                "an inserted opr %s re-insert into graph"
                "with eager evaluation mode OFF.",
                opr->cname());
        opr_uniqp.release();
        // No need to do the insert_post under eager mode
        eager_eval_manager().on_opr_insert(opr);
        return opr;
    }

    auto&& infer_mgr = static_infer_manager_impl();
    auto cleanup = [&]() {
        infer_mgr.set_register_allowed_opr(nullptr);
        for (auto i : opr->output()) {
            infer_mgr.clear_tag_handler(i);
            var_node_mem_manager().remove_var_node_mem_trait(i);
        }
    };

    if (auto ret = graph_optimizer().insert_pre(opr)) {
        bool should_update_shape = true;
#if !MGB_BUILD_SLIM_SERVING
        // in normal mode, we update the shape in deduplication in case shape
        // changes; in eager evaluation mode, shape is set by EagerEvalManager
        // and should not be modified
        should_update_shape = !options().eager_evaluation;
#endif
        if (should_update_shape) {
            update_output_shapes(infer_mgr, ret, false);
        }
        cleanup();
        event().signal_inplace<cg::event::OprInserted>(true, ret, nullptr);
        ret = graph_optimizer().insert_post(ret);
        eager_eval_manager().on_opr_insert(ret);
        return ret;
    }

    // record opr early, since exceptions may refer to the opr
    m_opr_refkeeper.emplace_back(std::move(opr_uniqp));

    MGB_TRY {
        mgb_assert(!opr->inserted_in_graph());
        mgb_assert(!opr->output().empty(), "operator must have at least one output");
        opr->set_inserted_in_graph();

        // basic init
        opr->init_output_comp_node();
        opr->init_output_dtype();
        opr->init_output_format();

        // check output initialized
        for (auto i : opr->output()) {
            mgb_assert(i->comp_node().valid() && i->dtype().valid());
        }

        // register static infer
        {
            auto old = infer_mgr.set_register_allowed_opr(opr);
            opr->init_output_static_infer_desc();
            infer_mgr.set_register_allowed_opr(old);
        }

        // more init
        opr->init_rt_force_dynamic_mem_alloc_imply_chain();

        // freeze output flag and static infer shape eagerly
        update_output_shapes(infer_mgr, opr, true);

        check_opr_not_cross_mem(opr);
    }
    MGB_CATCH(MegBrainError & exc, {
        cleanup();
        if (!exc.extra_info())
            OperatorNodeExcExtraInfo::record(opr, exc);
        event().signal_inplace<cg::event::OprInserted>(false, opr, &exc);
        throw;
    })

    // add to receiver list if above succeeds
    for (auto&& i : opr->input()) {
        auto iter = m_var_receiver.find(i);
        mgb_assert(iter != m_var_receiver.end());
        auto&& arr = iter->second;
        if (arr.empty() || arr.back() != opr) {
            // check if added, because opr may have identical inputs
            arr.push_back(opr);
        }
    }

    // alloc var receiver for the outputs
    for (auto&& i : opr->output()) {
        bool em = m_var_receiver[i].empty();
        mgb_assert(em);
    }

    event().signal_inplace<cg::event::OprInserted>(false, opr, nullptr);
    opr = graph_optimizer().insert_post(opr);
    eager_eval_manager().on_opr_insert(opr);
    return opr;
}

std::shared_ptr<ComputingGraph> ComputingGraph::make() {
    return std::make_shared<ComputingGraphImpl>();
}

std::unique_ptr<AsyncExecutable> ComputingGraphImpl::compile(
        const OutputSpec& out_spec) {
    return compile_commit(compile_prepare(out_spec));
}

SmallVector<std::unique_ptr<AsyncExecutable>> ComputingGraphImpl::compile_multi_part(
        const SmallVector<OutputSpec>& out_specs) {
#if MGB_ENABLE_PARTIAL_EXECUTION
    return MultiPartCompiler{this}.compile(out_specs);
#else
    mgb_throw(MegBrainError, "partial execution disabled at compile time");
#endif
}

ComputingGraphImpl::CompileState ComputingGraphImpl::compile_prepare(
        const OutputSpec& out_spec) {
    auto&& cmpnt = components();
    mgb_throw_if(
            m_recorded_seq_level2_dtor_chk, GraphError,
            "graphs with comp_node_seq_record_level==2 can only be "
            "compiled once");

    mgb_throw_if(
            out_spec.empty(), GraphError,
            "empty output spec given to ComputingGraph::compile");
    // topo sorter may have modified opr properties; restore them before this
    // new compiling
    topo_sorter().restore_opr_prop();
    cmpnt.seq_comp_node_opt.restore_comp_nodes();

    SpecialOprStat sopr_stat;
    auto dest_vars = get_dest_vars_from_out_spec(out_spec, sopr_stat);

#if MGB_ENABLE_SUBLINEAR
    if (options().enable_sublinear_memory_opt) {
        mgb_assert(!options().enable_dtr_memory_opt);
        if (!sopr_stat.has_virtual_grad) {
            mgb_log_debug(
                    "no virtual grad var; sublinear memory may produce "
                    "unsatisfying result");
        }
        seq_modifier_for_sublinear_memory().set_priority_before_opt(dest_vars);
    }
#else
    mgb_assert(!options().enable_sublinear_memory_opt);
#endif  //  MGB_ENABLE_SUBLINEAR

#if MGB_ENABLE_DTR
    if (options().enable_dtr_memory_opt) {
        mgb_assert(!options().enable_sublinear_memory_opt);
        seq_modifier_for_dtr().set_priority_before_opt(dest_vars);
    }
#else
    mgb_assert(!options().enable_dtr_memory_opt);
#endif  //   MGB_ENABLE_DTR

#if !MGB_BUILD_SLIM_SERVING
    mgb_assert(
            !options().eager_evaluation, "attempt to compile eager_evaluation graph");

    {
        bool need_opt = std::abs(options().graph_opt_level) >= 2;
        gopt::GraphOptimizer optimizer;
        optimizer.verbosity(options().log_level);
        optimizer.enable_check_result(options().graph_opt_level < 0);
        if (sopr_stat.has_virtual_grad) {
            if (need_opt) {
#if MGB_ENABLE_OPR_MM
                optimizer.add_pass<gopt::PackAllReduceScanPass>();
#endif
                optimizer.add_preset_passes(false, nullptr, &options());
            }
            optimizer.add_pass<gopt::ExpandVirtualGradPass>();
        }
        if (need_opt) {
            optimizer.add_preset_passes(true, nullptr, &options());
#if MGB_ENABLE_OPR_MM
            if (sopr_stat.has_virtual_grad) {
                optimizer.add_pass<gopt::PackAllReduceReplacePass>();
            }
#endif
        }
        optimizer.apply_inplace(dest_vars);
    }
#endif

#if MGB_ENABLE_TENSOR_RT
    if (options().graph_opt.tensorrt) {
        options().graph_opt.tensorrt = false;
        tensorrt::transform_dest_vars_inplace(dest_vars, options().graph_opt);
    }
#endif

#if MGB_JIT
    if (std::abs(options().graph_opt_level) == 0 &&
        (options().graph_opt.jit || options().graph_opt.jit_config.enabled())) {
        // Deprecated usage added previously. It allows NVRTC JIT optimization
        // when graph_opt_level is 0. This usage is not recommanded any more.
        mgb_log_warn(
                "It is not recommanded to enable JIT optimization when "
                "graph_opt_level is 0.");
        setenv("MGB_JIT_BACKEND", "NVRTC", 1);
        gopt::GraphOptimizer optimizer;
        optimizer.add_pass<gopt::JITFusionPass>(
                sopr_stat.has_virtual_grad, options().graph_opt.jit,
                options().graph_opt.jit_config);
        optimizer.apply_inplace(dest_vars);
    }
#endif
    gopt::GraphOptimizer optimizer;
    /**
     * \note We should reset options when we add passes indicated by optimize
     * options, As there exists `ParamFuse pass` will compile subgraph which may
     * cause ring invoking, \see
     * https://git-core.megvii-inc.com/brain-sdk/MegBrain/merge_requests/1717
     * for detail
     */
    optimizer.add_passes_for_optimize_options(options().graph_opt, true);
    optimizer.apply_inplace(dest_vars);

    if (sopr_stat.has_shape_hint) {
        // FIXME(zhangxuanrun): strictly speaking, it could and has to remove
        // ShapeHints even they were occured in subgraph
        mgb_assert(!m_parent_graph, "can not use ShapeHint in subgraph");
        // always need remove shape hint
        gopt::GraphOptimizer opt;
        opt.add_pass<gopt::RemoveShapeHintPass>();
        opt.apply_inplace(dest_vars);
    }

    const OprNodeArray* opr_seq = nullptr;
    CompSeqExtraInfo extra_info;
    cmpnt.seq_comp_node_opt.optimize_comp_nodes(dest_vars);

    bool init_flag = false;
    auto init_opr_seq = [&]() {
        mgb_assert(!init_flag);
        init_flag = true;
        ThinHashMap<VarNode*, size_t> var2idx;
        std::unordered_map<
                CallbackCallerKey, CallbackCallerVal, CallbackCallerKey::Hash>
                opr2vars;
        using F = VarNode::Flag;
        if (dest_vars[0]->owner_graph()->options().force_output_dynamic_alloc) {
            for (auto&& i : dest_vars) {
                if (!i->contain_flag(
                            F::NO_SYS_MEM_ALLOC | F::NO_SYS_STATIC_MEM_ALLOC)) {
                    mgb_assert(
                            !i->contain_flag(F::DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC),
                            "Can not force graph output dynamic alloc with "
                            "DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC flag, var: %s",
                            i->cname());
                    i->add_flag(F::NO_SYS_STATIC_MEM_ALLOC);
                }
                i->add_flag(F::NO_MEM_RECLAIM);
            }
        }
        for (size_t i = 0; i < out_spec.size(); ++i) {
            auto&& cb = out_spec[i].second;
            if (cb) {
                auto var = dest_vars[i];
                CallbackCallerKey key{var->owner_opr(), var->comp_node()};
                auto&& vals = opr2vars[key];
                auto&& var2idx_iter = var2idx.find(var);
                if (var2idx_iter == var2idx.end()) {
                    vals.vars.push_back(var);
                    vals.indexs.push_back({i});
                    var2idx[var] = vals.vars.size() - 1;
                } else {
                    vals.indexs[var2idx_iter->second].push_back(i);
                }
            }
        }
        for (auto& item : opr2vars) {
            auto&& val = item.second;
            auto dvar = CallbackCaller::make(val.vars);
            CallbackCaller* cb_caller =
                    &dvar.node()->owner_opr()->cast_final_safe<CallbackCaller>();
            ++extra_info.var2recvinfo[dvar.node()].nr_direct_comp_req;
            cb_caller->clear_callback();
            for (size_t i = 0; i < val.vars.size(); ++i) {
                for (auto&& idx : val.indexs[i]) {
                    cb_caller->add_callback(out_spec[idx].second, i);
                    dest_vars[idx] = cb_caller->output(0);
                }
            }
        }
        opr_seq = topo_sorter().get_comp_seq(extra_info, dest_vars);
    };

#if MGB_ENABLE_MEMORY_SWAP
    bool enable_swap_memory_after_sublinear =
            options().enable_sublinear_memory_opt && options().enable_memory_swap;

    bool enable_swap_memory_without_sublinear =
            !(options().enable_sublinear_memory_opt) && options().enable_memory_swap;

    if (enable_swap_memory_without_sublinear) {
        components().memory_swap_support.modify_dest_var_inplace(dest_vars);
    }
#else
    mgb_assert(!options().enable_memory_swap);
#endif

#if MGB_ENABLE_DTR
    if (options().enable_dtr_memory_opt) {
        MGB_TRY {
            seq_modifier_for_dtr().modify_endpoint_vars(dest_vars);
            init_opr_seq();
        }
        MGB_FINALLY(seq_modifier_for_dtr().restore_graph_option());
    }
#endif
#if MGB_ENABLE_SUBLINEAR
    if (options().enable_sublinear_memory_opt) {
        MGB_TRY {
            seq_modifier_for_sublinear_memory().modify_endpoint_vars(dest_vars);
#if MGB_ENABLE_MEMORY_SWAP
            if (enable_swap_memory_after_sublinear) {
                cmpnt.memory_swap_support.modify_dest_var_inplace(dest_vars);
            }
#endif

            init_opr_seq();
        }
        MGB_FINALLY(

                /*
                 * restore graph option immediately because it may be
                 * read/modified by user
                 */
                seq_modifier_for_sublinear_memory().restore_graph_option());
        seq_modifier_for_sublinear_memory().sanity_check(*opr_seq);
    }
#endif  //  MGB_ENABLE_SUBLINEAR
    if (!init_flag) {
        init_opr_seq();
    }

    return {std::move(extra_info), opr_seq, std::move(dest_vars)};
}

std::unique_ptr<AsyncExecutable> ComputingGraphImpl::compile_commit(
        CompileState state) {
    auto comp_seq = std::make_unique<ComputingSequence>(shared_from_this());
    comp_seq->extra_info = std::move(state.extra_info);
    comp_seq->set_output_vars(state.dest_vars);
    auto opr_seq = state.opr_seq;
    auto&& cmpnt = components();

    comp_seq->setup_opr_seq(opr_seq);
    for (auto&& i : *opr_seq) {
        for (auto&& j : i->node_prop().dep_map()) {
            if (OperatorNodeBase::NodeProp::is_device_value_dep(j.second)) {
                comp_seq->extra_info.var2recvinfo.at(j.first).last_dev_value_reader = i;
            }
        }
    }
    comp_seq->attach_to_graph();

    MGB_TRY {
        var_node_mem_manager().reset_opr_seq(comp_seq->extra_info, opr_seq);
        static_infer_comp_seq_manager().reset_dest(comp_seq->extra_info);
        cmpnt.seq_comp_node_opt.init_ready_event(comp_seq->extra_info, *opr_seq);

        if (options().allocate_static_mem_after_graph_compile)
            var_node_mem_manager().alloc_var_node_mem_static();
    }
    MGB_FINALLY({ var_node_mem_manager().on_graph_compile_finished(); });

    event().signal_inplace<event::CompSeqOrderDetermined>(this, comp_seq.get());

    if (options().comp_node_seq_record_level > 1) {
        mgb_assert(
                options().comp_node_seq_record_level <= 2,
                "invalid comp_node_seq_record_level: %u",
                options().comp_node_seq_record_level);
        mgb_assert(
                !options().fake_next_exec && !options().var_sanity_check_first_run,
                "both fake_next_exec and var_sanity_check_first_run "
                "must be false when comp_node_seq_record_level is 2");
        return comp_seq->as_recorded_seq();
    }
    return comp_seq;
}

VarNodeArray ComputingGraphImpl::get_dest_vars_from_out_spec(
        const OutputSpec& spec, SpecialOprStat& sopr_stat) {
    SymbolVarArray sym_vars;
    for (auto&& i : spec) {
        sym_vars.push_back(i.first);
    }
    return to_var_node_array(get_dest_vars_with_extra_deps(sym_vars, &sopr_stat));
}

const ComputingGraph::VarReceiverInfo& ComputingGraphImpl::
        var_receiver_in_current_comp_seq(const VarNode* var) const {
    static VarReceiverInfo empty;
    if (auto ret = components().eager_eval_manager.var_receiver_info(var)) {
        return *ret;
    }
    if (!m_current_comp_seq)
        return empty;
    auto cseq = static_cast<ComputingSequence*>(m_current_comp_seq);
    auto iter = cseq->extra_info.var2recvinfo.find(var);
    if (iter == cseq->extra_info.var2recvinfo.end())
        return empty;
    return iter->second;
}

VarNode* ComputingGraphImpl::find_var_by_id(size_t id) const {
    for (auto&& i : m_opr_refkeeper) {
        for (auto j : i->output()) {
            if (j->id() == id)
                return j;
        }
    }
    for (auto&& i : m_subgraphs) {
        auto sub = i->find_var_by_id(id);
        if (sub)
            return sub;
    }
    return nullptr;
}

#if MGB_ENABLE_SUBLINEAR
SeqModifierForSublinearMemory& ComputingGraphImpl::seq_modifier_for_sublinear_memory() {
    return components().seq_modifier_for_sublinear_memory;
}
#endif

#if MGB_ENABLE_DTR
SeqModifierForDTR& ComputingGraphImpl::seq_modifier_for_dtr() {
    return components().seq_modifier_for_dtr;
}
#endif

void ComputingGraphImpl::share_device_memory_with(ComputingGraph& other) {
    mgb_assert(
            !m_current_comp_seq,
            "share_device_memory_with must be called before compiling graph");
    auto&& oimpl = *ComputingGraphImpl::downcast(&other);
    var_node_mem_manager().static_device_memory_manager(
            oimpl.var_node_mem_manager().static_device_memory_manager());
}

void ComputingGraphImpl::set_device_memory_allocator(
        std::shared_ptr<DeviceMemoryAllocator> allocator) {
    var_node_mem_manager().static_device_memory_manager()->set_allocator(
            std::move(allocator));
}

size_t ComputingGraphImpl::get_device_memory_size(CompNode cn) {
    return var_node_mem_manager().static_device_memory_manager()->get_size(cn);
}

size_t ComputingGraphImpl::clear_device_memory() {
#if !MGB_BUILD_SLIM_SERVING
    if (options().eager_evaluation) {
        for (auto& opr : m_opr_refkeeper) {
            if (!opr->same_type<mgb::opr::SharedDeviceTensor>() &&
                !opr->same_type<mgb::opr::ImmutableTensor>()) {
                for (auto& var : opr->output()) {
                    if (var->mem_plan().valid())
                        var->mem_plan().release_chunk();
                }
            }
        }
    }
#endif
    return var_node_mem_manager().clear_static_device_memory();
}

void ComputingGraphImpl::set_as_subgraph(ComputingGraph& par_graph) {
    m_parent_graph = ComputingGraphImpl::downcast(&par_graph);
    m_parent_graph->m_subgraphs.emplace_back(this);
    m_node_id_counter = m_parent_graph->m_node_id_counter;
    options().var_sanity_check_first_run =
            par_graph.options().var_sanity_check_first_run;
    par_graph.event().signal_inplace<event::SubgraphAssociated>(&par_graph, this);
}

void ComputingGraphImpl::record_async_error(std::unique_ptr<MegBrainError> async_exc) {
    mgb_assert(m_current_comp_seq);
    static_cast<ComputingSequence*>(m_current_comp_seq)
            ->set_async_error(std::move(async_exc));
}

const CompSeqExtraInfo& ComputingGraphImpl::current_comp_seq_extra_info() {
    if (auto ret = eager_eval_manager().comp_seq_extra_info()) {
        return *ret;
    }
    mgb_assert(m_current_comp_seq);
    return static_cast<ComputingSequence*>(m_current_comp_seq)->extra_info;
}

GraphExecutable::ExecEnv* ComputingGraphImpl::current_exec_env() {
    if (auto ret = eager_eval_manager().exec_env()) {
        return ret;
    }
    if (m_current_comp_seq) {
        return &static_cast<ComputingSequence*>(m_current_comp_seq)->exec_env();
    }
    return nullptr;
}

Maybe<size_t> ComputingGraphImpl::opr_step_num_in_cur_comp_seq(OperatorNodeBase* opr) {
    mgb_assert(m_current_comp_seq && opr->owner_graph() == this);
    return static_cast<ComputingSequence*>(m_current_comp_seq)->opr2stepnum(opr);
}

std::string ComputingGraphImpl::VarReceiverInfo::to_string() const {
    return mgb_ssprintf_log(
            "VarReceiverInfo("
            "nr_direct_comp_req=%zu dev_value=%zu, host_value=%zu, shape=%zu, "
            "allow_empty_value=%zu)",
            nr_direct_comp_req, dev_value, host_value, shape, allow_empty_value);
}

std::string ComputingGraphImpl::get_mem_allocation_info() const {
#if MGB_ENABLE_JSON
    auto make_var_json = [](VarNode* single_var) {
        auto&& cur_mem_plan = single_var->mem_plan();
        if (cur_mem_plan.valid())
            return json::Object::make(
                    {{"name", json::String::make(single_var->name())},
                     {"memory", json::Number::make(cur_mem_plan.chunk().size())},
                     {"dev_ptr", json::NumberInt::make(reinterpret_cast<size_t>(
                                         single_var->dev_tensor().raw_ptr()))}});
        else
            return json::Object::make(
                    {{"name", json::String::make(single_var->name())},
                     {"memory", json::Null::make()},
                     {"dev_ptr", json::Null::make()}});
    };

    auto objlist = json::Array::make();

    for (auto& opri : m_opr_refkeeper) {
        auto cur_opr = opri.get();

        auto objptr = json::Object::make();
        auto&& objbody = *objptr;

        objbody["name"] = json::String::make(cur_opr->name());

        auto jvars = json::Array::make();
        for (auto& outputi : cur_opr->output()) {
            jvars->add(make_var_json(outputi));
        }
        objbody["output"] = jvars;

        auto obj = json::Object::make({{std::to_string(cur_opr->id()), objptr}});

        objlist->add(obj);
    }

    return objlist->to_string();
#endif  // MGB_ENABLE_JSON
    mgb_log_warn(
            "target is not configured with JSON BUILD on,"
            "get_mem_allocation_info returns null string");
    return std::string();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
