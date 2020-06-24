/**
 * \file src/opr/impl/internal/megdnn_opr_wrapper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */


#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/graph/event.h"
#include "megbrain/comp_node_env.h"
#include "./megdnn_opr_wrapper.inl"

#include "megdnn/oprs.h"

using namespace mgb;
using namespace opr;
using namespace intl;
using namespace mixin;

/* ================== global functions ================== */

namespace {
    template<class Opr>
    class MegDNNGlobalOprContainer final: public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;

        std::shared_ptr<megdnn::Handle> m_megdnn_handle;
        std::unique_ptr<Opr> m_opr;

        public:
            MegDNNGlobalOprContainer(CompNode cn):
                m_megdnn_handle{get_megdnn_handle_shared(cn)},
                m_opr{m_megdnn_handle->create_operator<Opr>()}
            {
                mgb_assert(m_opr->is_thread_safe());
            }

            Opr* get() const {
                return m_opr.get();
            }
    };

    template<class Opr>
    MGB_TYPEINFO_OBJ_IMPL(MegDNNGlobalOprContainer<Opr>);

    class TempStorageContainer final: public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;

        public:
            std::mutex mtx;
            CompNode::UnorderedMap<DeviceTensorStorage> cn2storage;
    };
    MGB_TYPEINFO_OBJ_IMPL(TempStorageContainer);
} // anonymous namespace

std::shared_ptr<megdnn::Handle> intl::get_megdnn_handle_shared(
        CompNode comp_node) {
    auto& handle = MegDNNHandle::get(CompNodeEnv::from_comp_node(comp_node));
    return {handle.shared_from_this(), handle.handle()};
}

megdnn::Handle *intl::get_megdnn_handle(CompNode comp_node) {
    return MegDNNHandle::get(CompNodeEnv::from_comp_node(comp_node)).handle();
}

template<typename Opr>
Opr* intl::get_megdnn_global_opr(CompNode comp_node) {
    using T = MegDNNGlobalOprContainer<Opr>;
    auto maker = [comp_node]() {
        return std::make_shared<T>(comp_node);
    };
    return CompNodeEnv::from_comp_node(comp_node).get_user_data<T>(maker).get();
}

namespace mgb {
namespace opr {
namespace intl {
#define INST(o) \
    template o* get_megdnn_global_opr<o>(CompNode)
    INST(megdnn::AddUpdate);
    INST(megdnn::Relayout);
    INST(megdnn::Checksum);
#undef INST
} // namespace intl
} // namespace opr
} // namespace mgb

DeviceTensorStorage& intl::get_temp_storage(ComputingGraph& graph,
                                            CompNode comp_node) {
    auto container =
            graph.options()
                    .user_data.get_user_data_or_create<TempStorageContainer>();

    MGB_LOCK_GUARD(container->mtx);
    auto&& ret = container->cn2storage[comp_node];
    if (!ret.comp_node_valid()) {
        ret.comp_node(comp_node);
    }
    return ret;
}

DeviceTensorND intl::get_temp_tensor(ComputingGraph* graph, CompNode comp_node,
                                     const TensorLayout& layout) {
    if (graph) {
        DeviceTensorND ret;
        auto&& storage = get_temp_storage(*graph, comp_node);
        storage.ensure_size(layout.span().dist_byte());
        // use sub to disallow growing
        ret.reset(storage.sub(0), layout);
        return ret;
    }
    return {comp_node, layout};
}

void megdnn_utils::add_input_layout_constraint_contig(OperatorNodeBase &opr) {
    for (auto i: opr.input())
        i->add_layout_constraint_contiguous();
}

void megdnn_utils::add_output_vars(
        OperatorNodeBase &opr, size_t nr_output, bool add_workspace) {
    mgb_assert(opr.output().empty() && nr_output);
    if (nr_output == 1)
        opr.add_output(None);
    else {
        for (size_t i = 0; i < nr_output; ++ i)
            opr.add_output(ssprintf("o%zu", i));
    }
    if (add_workspace) {
        cg::add_workspace_output(&opr);
    }
}

megdnn::Workspace intl::get_megdnn_workspace_from_var(VarNode *var) {
    var->dtype().assert_is(dtype::Byte());
    if (!var->shape().ndim || !var->shape().shape[0])
        return {};
    auto &&val = var->dev_tensor();
    mgb_assert(val.layout().ndim == 1);
    return {val.raw_ptr(), val.shape()[0]};
}

/* ================== WorkspaceLimitGetter  ================== */
#if MGB_BUILD_SLIM_SERVING && !MGB_CUDA
size_t WorkspaceLimitGetter::get_workspace_limit(
        ComputingGraph *, CompNode, size_t old_limit) {
    return old_limit;
}


VarNode* WorkspaceLimitGetter::register_to_graph(ComputingGraph *) {
    return nullptr;
}

bool WorkspaceLimitGetter::is_prealloc_run(ComputingGraph *graph) {
    return false;
}

#else

class WorkspaceLimitGetter::Impl final: public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    bool m_first_alloc_finished = false;
    int *m_static_infer_rerun_marker;
    VarNode *m_first_run_var;
    CompNode::UnorderedMap<size_t> m_upper_bound;

    void print_log() {
        if (!m_first_run_var->owner_graph()->options().log_level)
            return;

        std::vector<std::pair<std::string, size_t>> log_msg_vec;
        for (auto &&i: m_upper_bound) {
            log_msg_vec.emplace_back(i.first.to_string(), i.second);
        }
        std::sort(log_msg_vec.begin(), log_msg_vec.end());
        std::string msg("determined workspace size limit upper bound "
                "and reallocate static memory; bounds(MiB):");
        for (auto &&i: log_msg_vec) {
            msg.push_back(' ');
            msg.append(ssprintf("%s=%.2f", i.first.c_str(),
                        i.second / 1024.0 / 1024));
        }
        mgb_log_debug("%s", msg.c_str());
    }

    void on_graph_compile(const cg::event::CompSeqOrderDetermined&) {
        m_first_alloc_finished = false;
        ++ *m_static_infer_rerun_marker;
        m_upper_bound.clear();
    }

    void on_static_mem_alloc(const cg::event::StaticMemAlloc &ev) {
        if (m_first_alloc_finished)
            return;

        if (!ev.need_realloc) {
            // indicates that alloc on all comp nodes finished
            m_first_alloc_finished = true;
            print_log();
            return;
        }

        if (m_upper_bound.empty()) {
            // m_upper_bound.empty() indicates first alloc run; since
            // need_realloc would be set to true, the mem allocator would update
            // static infer info; we modify m_static_infer_rerun_marker to force
            // re-computation of workspace sizes
            ++ *m_static_infer_rerun_marker;

            // we may get more workspace by coalescing free memory
            CompNode::try_coalesce_all_free_memory();
        }
        *ev.need_realloc = true;
        auto free = ev.comp_node.get_mem_status_bytes().second;
        free += m_first_run_var->owner_graph()->get_device_memory_size(
                ev.comp_node);
        if (free < ev.alloc_size) {
            mgb_log_warn(
                    "insufficient memory on %s: free_mem=%zu alloc_req=%zu; "
                    "set workspace limit to 0",
                    ev.comp_node.to_string().c_str(), free, ev.alloc_size);
            free = ev.alloc_size;
        }
        m_upper_bound[ev.comp_node] = free - ev.alloc_size;
    }

    public:
        Impl(ComputingGraph *graph) {
            auto first_run = std::make_shared<HostTensorND>(
                    CompNode::load("cpux"), dtype::Int32());
            m_static_infer_rerun_marker = first_run->resize({1}).ptr<int>();
            *m_static_infer_rerun_marker = 0;
            m_first_run_var =
                opr::Host2DeviceCopy::make(*graph, first_run).node();

            using namespace std::placeholders;
            graph->event().
                register_receiver_permanent<cg::event::StaticMemAlloc>(
                        std::bind(&Impl::on_static_mem_alloc, this, _1));
            graph->event().
                register_receiver_permanent<cg::event::CompSeqOrderDetermined>(
                        std::bind(&Impl::on_graph_compile, this, _1));
        }

        size_t get_workspace_limit(CompNode cn, size_t old_limit) {
            return std::min(old_limit, m_upper_bound.at(cn));
        }

        VarNode* first_run_var() const {
            return m_first_run_var;
        }

        bool is_prealloc_run() const { return !m_first_alloc_finished; }
};
MGB_TYPEINFO_OBJ_IMPL(WorkspaceLimitGetter::Impl);

WorkspaceLimitGetter::Impl*
WorkspaceLimitGetter::get_impl(ComputingGraph *graph) {
    auto container = graph->options().user_data.get_user_data<Impl>();
    mgb_assert(container.second == 1);
    return container.first[0];
}

size_t WorkspaceLimitGetter::get_workspace_limit(
        ComputingGraph *graph, CompNode cn, size_t old_limit) {
    if (graph->options().imperative_proxy_graph) {
        return old_limit;
    }
    if (!graph->options().seq_opt.enable_mem_reuse_alloc)
        return old_limit;
    return get_impl(graph)->get_workspace_limit(cn, old_limit);
}

bool WorkspaceLimitGetter::is_prealloc_run(ComputingGraph* graph) {
    if (graph->options().imperative_proxy_graph) {
        return false;
    }
    return graph->options().seq_opt.enable_mem_reuse_alloc &&
           get_impl(graph)->is_prealloc_run();
}

VarNode* WorkspaceLimitGetter::register_to_graph(ComputingGraph *graph) {
    if (graph->options().imperative_proxy_graph) {
        return nullptr;
    }
    auto maker = [graph](){
        return std::make_shared<Impl>(graph);
    };
    return graph->options().user_data.get_user_data_or_create<Impl>(
            maker)->first_run_var();
}
#endif // MGB_BUILD_SLIM_SERVING

/* ================== MegDNNDynOutMallocImpl ================== */

megdnn::TensorND MegDNNDynOutMallocImpl::alloc_output(
        size_t id, DType dtype, const TensorShape &shape,
        void * /*user_data*/) {
    auto ovar = m_opr->output(id);
    mgb_assert(dtype == ovar->dtype());
    ovar->shape_alloc(shape);
    return ovar->dev_tensor().as_megdnn();
}

void* MegDNNDynOutMallocImpl::alloc_workspace(size_t sz, void * /*user_data*/) {
    return m_cn.alloc_device(sz);
}

void MegDNNDynOutMallocImpl::free_workspace(void *ptr, void * /*user_data*/) {
    m_cn.free_device(ptr);
}

/* ================== MegDNNGraphDep ================== */
MegDNNGraphDep::MegDNNGraphDep(
        std::unique_ptr<megdnn::OperatorBase> opr) noexcept
        : m_opr{std::move(opr)} {}

MegDNNGraphDep::~MegDNNGraphDep() noexcept = default;

/* ================== WorkspaceSizeInfer ================== */
void mixin::WorkspaceSizeInfer::mixin_init_output_static_infer_desc_workspace(
        OperatorNodeBase &opr, bool need_limit) {
    using namespace cg::static_infer;
    auto &&mgr = opr.owner_graph()->static_infer_manager();

    auto out_wksp = opr.output().back();
    mgb_assert(out_wksp->dtype() == dtype::Byte() &&
            out_wksp->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
    auto infer_workspace = [&opr, this](TensorShape &dest, const InpVal &inp) {
        TensorShapeArray inp_shp(opr.input().size()),
                         out_shp(opr.output().size() - 1);
        auto iter = inp.val.begin();
        for (size_t i = 0; i < inp_shp.size(); ++ i)
            inp_shp[i] = ((iter ++)->shape());
        for (size_t i = 0; i < out_shp.size(); ++ i)
            out_shp[i] = ((iter ++)->shape());
        mgb_assert(iter <= inp.val.end());
        dest.ndim = 1;
        dest.shape[0] = get_workspace_size_bytes(inp_shp, out_shp);
        return true;
    };
    DepVal deps;
    deps.reserve(opr.input().size() + opr.output().size());
    for (auto i: opr.input())
        deps.push_back({i, DepType::SHAPE});
    for (size_t i = 0; i + 1 < opr.output().size(); ++ i) {
        auto ovar = opr.output(i);
        mgb_assert(mgr.get_infer_type(ovar).shape != InferType::NO_DESC,
                "output shape infer must be registered before calling "
                "init_output_static_infer_desc_workspace");
        deps.push_back({ovar, DepType::SHAPE});
    }
    if (need_limit) {
        auto var = intl::WorkspaceLimitGetter::register_to_graph(
                opr.owner_graph());
        if (var) {
            deps.push_back({var, DepType::VALUE});
        }
    }
    mgr.register_shape_infer(out_wksp,
            {SourceType::DEP, deps, infer_workspace});
}


/* ================== MegDNNOprHolder ================== */

MegDNNOprHolder::~MegDNNOprHolder() noexcept = default;

void MegDNNOprHolder::mixin_init_output_comp_node(OperatorNodeBase &self) {
    SingleCNOperatorNode::mixin_init_output_comp_node(self);
    create_megdnn_opr();
    mgb_assert(m_megdnn_opr);
    m_megdnn_opr->set_error_tracker(&self);
}

void MegDNNOprHolder::mixin_on_output_comp_node_stream_changed(
        OperatorNodeBase &self) {
    SingleCNOperatorNode::mixin_on_output_comp_node_stream_changed(self);
    create_megdnn_opr();
    mgb_assert(m_megdnn_opr);
    m_megdnn_opr->set_error_tracker(&self);
}

void MegDNNOprHolder::set_megdnn_opr(
        std::unique_ptr<megdnn::OperatorBase> self) {
    m_megdnn_opr = std::move(self);
}

void MegDNNOprHolder::record_megdnn_opr(
        std::unique_ptr<megdnn::OperatorBase> opr,
        cg::GraphExecutable::ExecDependencyArray& deps) {
    deps.emplace_back(std::make_unique<MegDNNGraphDep>(std::move(opr)));
}

void MegDNNOprHolder::record_megdnn_opr(
        cg::GraphExecutable::ExecDependencyArray& deps) {
    record_megdnn_opr(std::move(m_megdnn_opr), deps);
}

/* ================== MegDNNOprHolderBwdStaticInfer ================== */

MegDNNOprHolderBwdStaticInfer::~MegDNNOprHolderBwdStaticInfer() = default;

void MegDNNOprHolderBwdStaticInfer::mixin_setup_megdnn_bwd_output_infer(
        size_t oshp_idx, bool oshp_need_val) {
    mgb_assert(m_oshp_idx == BAD_OSHP_IDX);
    m_oshp_idx = oshp_idx;
    m_oshp_need_val = oshp_need_val;
}

void MegDNNOprHolderBwdStaticInfer::mixin_init_output_static_infer_desc_bwd(
        OperatorNodeBase &self) const {
    mgb_assert(self.output().size() == 2 && m_oshp_idx != BAD_OSHP_IDX);

    using namespace cg::static_infer;
    auto &&mgr = self.owner_graph()->static_infer_manager();

    // output shape
    mgr.register_shape_infer(self.output(0),
            ShapeInferDesc::make_identity(self.input(m_oshp_idx)));
}

void MegDNNOprHolderBwdStaticInfer::mixin_init_output_dtype(
        OperatorNodeBase &self) {
    mgb_assert(m_oshp_idx != BAD_OSHP_IDX);
    self.output(0)->dtype(self.input(m_oshp_idx)->dtype());
}

void MegDNNOprHolderBwdStaticInfer::mixin_update_node_prop(
        const OperatorNodeBase &self, NodeProp *prop) const {
    mgb_assert(m_oshp_idx != BAD_OSHP_IDX);
    if (!m_oshp_need_val) {
        using DT = NodeProp::DepType;
        SmallVector<DT> dep_types(self.input().size(), DT::DEV_VALUE);
        dep_types.at(m_oshp_idx) = DT::SHAPE;
        prop->reset_dep_type(self.input(), dep_types);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
