/**
 * \file src/core/impl/graph/var_node.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/var_node.h"
#include "./cg_impl.h"
#include "megbrain/graph/helper.h"
#include "megbrain/graph/operator_node.h"

using namespace mgb;
using namespace cg;

/* ===================== MemAllocPlan =====================  */

MGB_MUTEX MemAllocPlan::ReadonlyFwdList::list_mutex;

void MemAllocPlan::ReadonlyFwdList::reset() {
    MGB_LOCK_GUARD(list_mutex);
    m_prev = m_next = nullptr;
}

void MemAllocPlan::ReadonlyFwdList::insert_after(
        const MemAllocPlan& prev, MemAllocPlan* self) {
    MGB_LOCK_GUARD(list_mutex);
    mgb_assert(!m_prev && !m_next);
    auto next = prev.m_readonly_fwd_list.m_next;
    prev.m_readonly_fwd_list.m_next = self;
    m_prev = const_cast<MemAllocPlan*>(&prev);
    m_next = next;
    if (next) {
        next->m_readonly_fwd_list.m_prev = self;
    }
}

void MemAllocPlan::ReadonlyFwdList::remove_self() {
    MGB_LOCK_GUARD(list_mutex);
    if (m_prev) {
        if (m_next) {
            m_prev->m_readonly_fwd_list.m_next = m_next;
            m_next->m_readonly_fwd_list.m_prev = m_prev;
        } else {
            m_prev->m_readonly_fwd_list.m_next = nullptr;
        }
        m_prev = m_next = nullptr;
    }
}

MemAllocPlan::Chunk MemAllocPlan::sm_chunk_invalid_cond_exec_marker{nullptr};

MemAllocPlan::MemAllocPlan(VarNode* owner_var) : m_chunk_storage(owner_var) {}

MemAllocPlan& MemAllocPlan::assign(const MemAllocPlan& src) {
    mgb_assert(src.valid());
    m_layout = src.m_layout;
    m_layout.dtype = dtype();
    m_offset_byte = src.m_offset_byte;
    m_chunk = src.m_chunk;
    ++m_chunk->m_refcnt;
    return *this;
}

MemAllocPlan& MemAllocPlan::assign_for_forward(
        const MemAllocPlan& src, const SubTensorSpec& sub) {
    mgb_assert(valid() && src.valid() && m_layout.eq_shape(sub.layout()));
    ++(m_chunk = src.m_chunk)->m_refcnt;
    m_layout = sub.layout();
    // make layout strong-contig
    for (int i = static_cast<int>(m_layout.ndim) - 1; i >= 0; --i) {
        if (m_layout.shape[i] == 1) {
            m_layout.stride[i] = i + 1 < static_cast<int>(m_layout.ndim)
                                       ? m_layout.stride[i + 1] * m_layout.shape[i + 1]
                                       : 1;
        }
    }
    m_layout.dtype = dtype();
    m_offset_byte = src.m_offset_byte + sub.offset_byte();
    auto&& span = sub.layout().span();
    mgb_assert(
            m_offset_byte + span.high_byte <= m_chunk->size() &&
            static_cast<ptrdiff_t>(m_offset_byte) + span.low_byte >= 0);
    // Note: Multiple mem plans may be forwarded from the same mem plan. Here we
    // do not need to find the root mem plan. Instead, we just insert this node
    // to the linked list headed at the root node, obeying topological order,
    // but note that new nodes may be inserted into the middle of the list.
    m_readonly_fwd_list.insert_after(src, this);
    return *this;
}

MemAllocPlan& MemAllocPlan::reset_from_owner_var() {
    auto owner_var = m_chunk_storage.owner_var;
    m_layout.dtype = dtype();
    m_layout.format = format();
    m_layout.init_contiguous_stride(owner_var->shape());
    m_offset_byte = 0;
    m_chunk = &m_chunk_storage;
    auto chk = m_chunk;
    chk->m_refcnt = 1;
    chk->m_size = m_layout.span().dist_byte();
    chk->mem_alloc_status.set_invalid();
    mgb_assert(chk->m_refcnt.is_lock_free());

    // check size for not overflow
    mgb_assert(
            m_layout.total_nr_elems() <= m_layout.dtype.max_elements(),
            "var too large: %s", cg::dump_var_info({owner_var}).c_str());
    return *this;
}

MemAllocPlan& MemAllocPlan::release_chunk() {
    mgb_assert(valid());
    auto chk = m_chunk;
    bool need_consider = chk != &sm_chunk_invalid_cond_exec_marker;
    m_readonly_fwd_list.remove_self();
    if (need_consider && (!--chk->m_refcnt)) {
        auto&& dv = chk->owner_var->m_dev_tensor;
        mgb_assert(dv.storage().comp_node_valid());
        if (chk->size()) {
            mgb_assert(chk->mem_alloc_status.is_from_owner_var());
            chk->m_size = 0;
        }
        chk->mem_alloc_status.set_invalid();
        dv.storage({});
    }
    m_chunk = nullptr;
    return *this;
}

MemAllocPlan& MemAllocPlan::layout(const TensorLayout& dest, bool allow_shape_change) {
    mgb_assert(
            allow_shape_change || m_layout.eq_shape(dest),
            "disallowed shape change: %s vs %s",
            m_layout.TensorShape::to_string().c_str(),
            dest.TensorShape::to_string().c_str());
    m_layout = dest;
    m_layout.dtype = dtype();
    return *this;
}

#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> MemAllocPlan::to_json() const {
    auto cvt_layout = [](const TensorLayout& layout) {
        auto shape = json::Array::make(), stride = json::Array::make();
        for (size_t i = 0; i < layout.ndim; i++) {
            shape->add(json::Number::make(layout.shape[i]));
            stride->add(json::Number::make(layout.stride[i]));
        }
        return json::Object::make(
                {{"shape", shape},
                 {"stride", stride},
                 {"dtype", json::String::make(layout.dtype.name())}});
    };

    return json::Object::make(
            {{"mem_chunk_id", json::String::make(m_chunk->id_str())},
             {"layout", cvt_layout(m_layout)},
             {"offset_byte", json::Number::make(m_offset_byte)}});
}
#endif

std::string MemAllocPlan::Chunk::id_str() const {
    return "chk" + std::to_string(owner_var->id());
}

/* ===================== MemAllocPlan::Chunk =====================  */
#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> MemAllocPlan::Chunk::to_json() const {
    std::shared_ptr<json::Value> dev_ptr;
    if (owner_var->dev_tensor_valid()) {
        dev_ptr = json::NumberInt::make(
                reinterpret_cast<size_t>(owner_var->dev_tensor().raw_ptr()));
    } else {
        dev_ptr = json::Null::make();
    }
    return json::Object::make(
            {{"node_type", json::String::make("mem_chunk")},
             {"id", json::String::make(id_str())},
             {"size", json::Number::make(size())},
             {"owner_var", json::String::make(owner_var->id_str())},
             {"dev_ptr", dev_ptr}});
}
#endif

/* ===================== VarNode =====================  */

const std::string& VarNode::name() const {
    return m_name.valid() ? m_name.val() : owner_opr()->name();
}

VarNode& VarNode::name(std::string name) {
    m_name = std::move(name);
    m_has_name_set = true;
    return *this;
}

const DeviceTensorND& VarNode::dev_tensor() const {
    mgb_assert(dev_tensor_valid());
    return m_dev_tensor;
}

DeviceTensorND& VarNode::mutable_dev_tensor() {
    mgb_assert(dev_tensor_valid() && contain_flag(Flag::NO_SYS_MEM_ALLOC));
    return m_dev_tensor;
}

VarNode& VarNode::dtype(DType dtype) {
    mgb_assert(dtype.valid() && !m_dev_tensor.dtype().valid());
    m_dev_tensor.dtype(dtype);
    return *this;
}

VarNode& VarNode::format(TensorFormat format) {
    mgb_assert(format == m_dev_tensor.format() || m_dev_tensor.format().is_default());
    m_dev_tensor.format(format);
    return *this;
}

bool VarNode::set_fwd_in2out_readonly(VarNode* input, const SubTensorSpec& sub) {
    if (owner_graph()->options().imperative_proxy_graph) {
        return false;
    }
    return ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .fwd_in2out_readonly(input, sub, this);
}

VarNode& VarNode::set_fwd_in2out_writable(VarNode* input) {
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .fwd_in2out_writable(input, this);
    return *this;
}

VarNode& VarNode::set_fwd_in2out_writable_force(VarNode* input) {
    mgb_assert(!owner_graph()->options().imperative_proxy_graph);
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .fwd_in2out_writable_force(input, this);
    return *this;
}

VarNode& VarNode::add_layout_constraint(LayoutConstraintCallback callback) {
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .add_layout_constraint(this, std::move(callback));
    return *this;
}

VarNode& VarNode::add_layout_constraint_contiguous() {
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .add_layout_constraint_level(
                    this, VarNodeMemManager::LayoutConstraintLevel::CONTIG);
    return *this;
}

VarNode& VarNode::add_layout_constraint_monotone() {
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .add_layout_constraint_level(
                    this, VarNodeMemManager::LayoutConstraintLevel::MONOTONE);
    return *this;
}

VarNode& VarNode::shape(const TensorShape& shape) {
    if (!m_shape.eq_shape(shape)) {
        mgb_assert(
                m_allow_shape_change,
                "invalid var shape change: "
                "dest=%s var=%s",
                shape.to_string().c_str(), dump_var_info({this}).c_str());
        m_shape = shape;
        for (auto&& i : m_shape_update_callback)
            i.second(this);
    }

#if MGB_ENABLE_DEBUG_UTIL
    static size_t log_limit =
            MGB_GETENV("MGB_LOG_VAR_SIZE_MB")
                    ? std::stold(MGB_GETENV("MGB_LOG_VAR_SIZE_MB")) * (1024 * 1024)
                    : 0;
    if (log_limit) {
        auto size = dtype().size(shape.total_nr_elems());
        static size_t max_size = 0;
        if (size >= log_limit) {
            bool updated = false;
            if (size > max_size) {
                max_size = size;
                updated = true;
            }
            mgb_log("var exceeds log limit: %s; size=%.3fMiB%s",
                    cg::dump_var_info({this}).c_str(), size / (1024.0 * 1024),
                    updated ? " (with maxsize updated)" : "");
        }
    }
#endif

    return *this;
}

VarNode& VarNode::shape_alloc(const TensorShape& shape, size_t size_req) {
    mgb_assert(
            shape.ndim,
            "got empty shape in shape_alloc: "
            "var=%s owner_opr=%s{%s}",
            cname(), owner_opr()->cname(), owner_opr()->dyn_typeinfo()->name);
    mgb_assert(
            contain_flag(Flag::NO_SYS_MEM_ALLOC),
            "shape_alloc() could only be used for vars with"
            " NO_SYS_MEM_ALLOC flag; actual var: %s",
            cg::dump_var_info({this}).c_str());
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .var_alloc_with_shape(this, shape, size_req);
    return *this;
}

bool VarNode::reset_dev_tensor_from_other_var(VarNode* src_var) {
    mgb_assert(contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC));
    if (src_var->owner_graph() == owner_graph()) {
        // this is actually readonly forwarding in the same graph
        mgb_assert(
                src_var->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE) ||
                        !is_static_var_storage(src_var),
                "dynamic storage on src is required for dynamic readonly "
                "forwarding: vars=%s",
                dump_var_info({src_var, this}).c_str());
        auto&& trait = ComputingGraphImpl::downcast(owner_graph())
                               ->var_node_mem_manager()
                               .get_var_node_mem_trait_at(src_var);
        if (trait.seq_force_update_dest ||
            !src_var->dev_tensor().layout().is_contiguous()) {
            shape_alloc(src_var->shape())
                    .dev_tensor()
                    .copy_from_fixlayout(src_var->dev_tensor());
            return false;
        }
    }
    shape(src_var->shape());
    m_mem_plan.assign(src_var->m_mem_plan);
    assign_dev_tensor_from_tensor(src_var->dev_tensor());
    return true;
}

VarNode& VarNode::reset_dev_tensor_from_tensor(const DeviceTensorND& value) {
    mgb_assert(contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC));
    mgb_assert(
            value.comp_node() == comp_node(),
            "attempt to reset var on %s from a value on %s",
            comp_node().to_string().c_str(), value.comp_node().to_string().c_str());
    shape(value.shape());
    auto&& chk = m_mem_plan.reset_from_owner_var().chunk();
    assign_dev_tensor_from_tensor(value);
    chk.mem_alloc_status.set_from_owner_var();
    return *this;
}

void VarNode::assign_dev_tensor_from_tensor(const DeviceTensorND& value) {
    mgb_assert(
            (value.layout().is_contiguous() || value.empty()) &&
            m_dev_tensor.dtype() == value.dtype() &&
            m_dev_tensor.format() == value.format());
    if (value.empty()) {
        mgb_assert(value.shape_valid() && value.layout().is_empty());
        bool allow_empty = contain_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
        auto&& recv = owner_graph()->var_receiver_in_current_comp_seq(this);
        mgb_throw_if(
                !allow_empty || !recv.is_empty_allowed(), GraphError,
                "assign empty tensor to var %s, but allowed=%d, receiver=%s",
                cg::dump_var_info({this}).c_str(), allow_empty,
                recv.to_string().c_str());
    }
    if (cg::is_static_var_shape(this)) {
        mgb_assert(
                shape().eq_shape(value.shape()),
                "shape mismatch for static inferrable var when setting dev "
                "tensor: var=%s new_shape=%s",
                cg::dump_var_info({this}).c_str(), value.shape().to_string().c_str());
    }
    m_dev_tensor.reset(value.storage(), value.layout());
    m_dev_tensor.comp_node(comp_node());
    m_prev_dev_ptr = value.raw_ptr();
    mgb_assert(dev_tensor_valid());
}

VarNode& VarNode::add_rt_force_dynamic_mem_alloc_imply_chain(VarNode* dest) {
    mgb_assert(
            dest && dest->owner_graph() == owner_graph() &&
            (!contain_flag(Flag::FLAG_FREEZED) ||
             !dest->contain_flag(Flag::FLAG_FREEZED)));
    m_rt_force_dynamic_mem_alloc_imply_chain.push_back(dest);
    return *this;
}

VarNode& VarNode::comp_node(const CompNode& cn) {
    mgb_assert(
            cn.valid() &&
            (!m_comp_node.valid() || m_comp_node.mem_node() == cn.mem_node()));
    m_comp_node = cn;
    if (m_cn_sync_manager) {
        m_cn_sync_manager->comp_node(cn);
    }
    return *this;
}

#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> VarNode::dump_static_infer_info_to_json() const {
    using namespace cg::static_infer;
    auto&& mgr = static_cast<cg::ComputingGraphImpl*>(owner_graph())
                         ->static_infer_manager_impl();
    auto get_dep_type = [](const DepType& type) -> std::string {
        switch (type) {
#define cb(name)        \
    case DepType::name: \
        return #name;
            cb(SHAPE) cb(VALUE)
#undef cb
                    default : mgb_throw(MegBrainError, "unknown dep type");
        }
    };
    auto get_infer_type = [](const InferType::Flag& type) {
        switch (type) {
#define cb(name)                \
    case InferType::Flag::name: \
        return json::String::make(#name);
            cb(NO_DESC) cb(CONST) cb(RT_STATIC) cb(MISSING_INP)
#undef cb
                    default : mgb_throw(MegBrainError, "unknown infer type");
        }
    };
    auto make_tag = [&](const DepType& type) {
        VarNode* self = const_cast<VarNode*>(this);
        auto c_deps = mgr.get_deps({self, type});
        auto deps = json::Array::make();
        for (auto&& i : c_deps) {
            mgb_assert(i.dest);
            deps->add(json::Object::make(
                    {{"var", json::String::make(i.dest->id_str())},
                     {"dep_type", json::String::make(get_dep_type(i.type))}}));
        }
        auto infer_type_handle = mgr.get_infer_type(self);
        auto inferred_result = json::Null::make();
        auto infer_type = type == DepType::SHAPE ? infer_type_handle.shape
                                                 : infer_type_handle.value;
        if (infer_type != InferType::Flag::NO_DESC) {
            if (type == DepType::SHAPE) {
                if (auto shape = mgr.infer_shape_fallible(self)) {
                    auto inferred_shape = json::Array::make();
                    for (size_t i = 0; i < shape->ndim; ++i) {
                        inferred_shape->add(json::Number::make((*shape)[i]));
                    }
                    inferred_result = inferred_shape;
                }
            } else {
                if (auto p = mgr.infer_value_fallible(self)) {
                    auto&& dev = *p;
                    if (dev.shape().ndim == 1 && dev.shape(0) < TensorShape::MAX_NDIM &&
                        mgb_likely(dev.comp_node() == CompNode::default_cpu())) {
                        MGB_TRY {
                            size_t nr_elems = dev.shape(0);
                            auto&& dtype = dev.dtype();
                            void* vptr = dev.raw_ptr();
                            double data[nr_elems];
                            HostTensorND contig;
                            if (!dev.layout().is_contiguous()) {
                                // both src and dst are placed on default cpu,
                                // no need for sync
                                contig.copy_from(dev);
                                mgb_assert(contig.layout().is_contiguous());
                                vptr = contig.raw_ptr();
                            }
                            static_cast_dtype(data, dtype, vptr, nr_elems);
                            auto inferred_value = json::Array::make();
                            for (size_t i = 0; i < nr_elems; ++i) {
                                inferred_value->add(json::Number::make(data[i]));
                            }
                            inferred_result = inferred_value;
                        }
                        MGB_CATCH(ConversionError&, {});
                    } else {
                        inferred_result = json::String::make("Large Array");
                    }
                }
            }
        }
        return json::Object::make(
                {{"node_type", json::String::make("static_infer_tag")},
                 {"infer_type", get_infer_type(infer_type)},
                 {"inferred_result", inferred_result},
                 {"deps", deps}});
    };
    return json::Object::make({
#define TAG(type) {get_dep_type(type), make_tag(type)}
            TAG(DepType::SHAPE), TAG(DepType::VALUE)
#undef TAG
    });
}

std::shared_ptr<json::Value> VarNode::to_json() const {
    auto get_var = [](VarNode* p) -> std::shared_ptr<json::Value> {
        if (p)
            return json::String::make(p->id_str());
        return json::Null::make();
    };

    auto&& trait = ComputingGraphImpl::downcast(owner_graph())
                           ->var_node_mem_manager()
                           .get_var_node_mem_trait(this);
    auto flag = json::Array::make();
    {
        // add flags
        size_t flag_checked = static_cast<size_t>(Flag::FLAG_FREEZED);
#define CHK(v)                                            \
    do {                                                  \
        if (contain_flag(Flag::v)) {                      \
            flag->add(json::String::make(#v));            \
            flag_checked |= static_cast<size_t>(Flag::v); \
        }                                                 \
    } while (0)
        CHK(NO_SYS_MEM_ALLOC);
        CHK(NO_ALLOC_IF_UNUSED);
        CHK(NO_SYS_STATIC_MEM_ALLOC);
        CHK(NO_MEM_RECLAIM);
        CHK(RT_FORCE_DYNAMIC_MEM_ALLOC);
        CHK(VOLATILE_CONTENT);
        CHK(ALLOW_EMPTY_SHAPE);
        CHK(PERSISTENT_DEVICE_VALUE);
        CHK(DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC);
        CHK(DISALLOW_VAR_SANITY_CHECK);
        CHK(MEMORY_NO_NEED);
#undef CHK

        mgb_assert(flag_checked == static_cast<size_t>(m_flag));
    }

    auto rst = json::Object::make(
            {{"node_type", json::String::make("var")},
             {"id", json::String::make(id_str())},
             {"name", json::String::make(name())},
             {"mem_readonly_fwd_src", get_var(trait.readonly_src)},
             {"force_update_src", get_var(trait.force_update_src)},
             {"mem_plan",
              m_mem_plan.valid() ? m_mem_plan.to_json() : json::Null::make()},
             {"comp_node", json::String::make(comp_node().to_string())},
             {"dev_ptr", json::Null::make()},
             {"prev_dev_ptr",
              json::NumberInt::make(reinterpret_cast<size_t>(m_prev_dev_ptr))},
             {"flag", flag},
             {"static_infer_tags", dump_static_infer_info_to_json()}});

    if (m_prev_dev_ptr) {
        (*rst)["prev_dev_ptr_end"] = json::NumberInt::make(
                reinterpret_cast<size_t>(m_prev_dev_ptr) +
                m_mem_plan.layout().span().high_byte);
    }
    if (dev_tensor_valid()) {
        (*rst)["dev_ptr"] =
                json::NumberInt::make(reinterpret_cast<size_t>(m_dev_tensor.raw_ptr()));
    }
    return rst;
}
#endif

MemAllocPlan& VarNode::init_mem_plan(const DeviceTensorND* fixed_alloc) {
    ComputingGraphImpl::downcast(owner_graph())
            ->var_node_mem_manager()
            .init_single_var_mem_plan(this, fixed_alloc);
    return m_mem_plan;
}

VarNode& VarNode::add_flag(Flag flag) {
    modify_flag(flag, m_flag | flag);
    return *this;
}

void VarNode::modify_flag(Flag delta, Flag new_flag) {
    if (contain_flag(Flag::FLAG_FREEZED)) {
        mgb_assert(
                (delta & (Flag::NO_MEM_RECLAIM | Flag::NO_SYS_STATIC_MEM_ALLOC |
                          Flag::RT_FORCE_DYNAMIC_MEM_ALLOC)) == delta ||
                (new_flag & Flag::MEMORY_NO_NEED));

        mgb_assert(
                !ComputingGraphImpl::downcast(owner_graph())
                         ->var_node_mem_manager()
                         .optimize_started(),
                "could not modify var flags after optimization started");
    }
    mgb_assert(
            !(new_flag & Flag::RT_FORCE_DYNAMIC_MEM_ALLOC) ||
                    !(new_flag & Flag::NO_SYS_MEM_ALLOC),
            "RT_FORCE_DYNAMIC_MEM_ALLOC conflicts with NO_SYS_MEM_ALLOC");
    mgb_assert(
            !(new_flag & Flag::NO_ALLOC_IF_UNUSED) ||
                    !(new_flag & Flag::NO_SYS_MEM_ALLOC),
            "NO_ALLOC_IF_UNUSED conflicts with NO_SYS_MEM_ALLOC");
    mgb_assert(
            !(new_flag & Flag::DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC) ||
                    (new_flag & Flag::NO_MEM_RECLAIM),
            "DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC must be added after "
            "NO_MEM_RECLAIM");
    m_flag = new_flag;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
