/**
 * \file src/core/impl/graph/static_infer_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#define LOG_INFER_RESULT 0

#include "./static_infer_impl.h"
#include "./cg_impl.h"
#include "./impl_common.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/helper.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/graph/var_node.h"
#include "megbrain/utils/shared_set.h"

#if LOG_INFER_RESULT
#include "megbrain/tensor_iter.h"
#endif

#include <cstring>
#include <deque>

using namespace mgb;
using namespace cg;
using namespace static_infer;

namespace {

constexpr size_t INFER_VALUE_SIZE_THRESH_FOR_WARNING = 1024,
                 INFER_VALUE_CHECK_UNCHANGE_MAX_SIZE = TensorLayout::MAX_NDIM;

constexpr bool is_static_infer_type(InferType::Flag t) {
    return t & (InferType::RT_STATIC | InferType::CONST);
}

#if MGB_ENABLE_EXCEPTION
[[noreturn]] void update_rethrow_exc(VarNode* var, MegBrainError& exc) {
    if (var && !exc.extra_info()) {
        OperatorNodeExcExtraInfo::record(var->owner_opr(), exc);
    }
    throw;
}
#endif

}  // namespace

/* ===================== nested class decls ===================== */

MGB_DEFINE_CLS_WITH_SUPER(StaticInferManagerImpl::TagTraitBase, TagHandler) // {
    const bool m_is_const;

protected:
    InferType::Flag m_infer_type = InferType::NO_DESC;

    //! added each time do_infer() is called and returns
    //! InferResult::CHANGED
    size_t m_inp_element_version = 0;

    TagTraitBase(Tag tag, bool is_const) : Super(tag), m_is_const{is_const} {}

public:
    using TagTraitArray = SmallVector<TagTraitBase*>;

    /*!
     * \brief whether this tag is TagConstShapeTrait
     *
     * This is not the same as InferType::CONST, which also includes const
     * value but is lazily inferred.
     */
    bool is_const() const { return m_is_const; }

    InferType::Flag infer_type() const { return m_infer_type; }

    //! version of most recent infer result
    size_t infer_result_version() const { return m_inp_element_version; }

    size_t update_infer_result_version() override {
        infer(false, false);
        return m_inp_element_version;
    }

    /*!
     * \brief get inferred value for pure nodes
     *
     * Note: inferencing would be skipped if m_inp_element_synced is true
     *
     * \param recomp_mutable_srcnode whether to re-compute mutable src
     *      nodes; if this is true, then this tag must be a mutable src
     *      (i.e. calling infer() on an intermediate trait with
     *       recomp_mutable_srcnode being true is not allowed).
     * \param allow_fail whether to allow returning nullptr result
     * \return inferred value, or nullptr if failed
     */
    const InpElement* infer(bool recomp_mutable_srcnode, bool allow_fail);

    /*!
     * \brief core implementation for infer(), without handling exceptions
     *
     * If infer result changes, all traits that depend on this one would be
     * marked as out-of-sync.
     *
     * \param[out] cur_active_var current variable, used for backtracing
     */
    virtual const InpElement* infer_withoutexc(
            VarNode** cur_active_var, bool recomp_mutable_srcnode) = 0;

    virtual const TagTraitArray& deps() const = 0;

    //! convert to TagTraitMutableBase; return nullptr on failure
    inline TagTraitMutableBase* as_mutable();

    //! assert this is mutable and convert to TagTraitMutableBase
    inline TagTraitMutableBase* as_mutable_safe();
};

/*!
 * \brief TagConstShapeTrait is used when the shape of Tag is const
 *
 * This is used to reduce memory usage and shorten the inference chain.
 */
MGB_DEFINE_CLS_WITH_SUPER(
        StaticInferManagerImpl::TagConstShapeTrait final, TagTraitBase) // {
    struct InferResultCache {
        Spinlock mtx;
#if __DEPLOY_ON_XP_SP2__
        ThinHashMap<size_t, InpElement> storage;
#else
        ThinHashMap<std::thread::id, InpElement> storage;
#endif
    };
    static TagTraitArray sm_empty_deps;
    static InferResultCache sm_result_cache;

public:
    TagConstShapeTrait(Tag tag) : Super(tag, true) {
        m_infer_type = InferType::CONST;
        m_inp_element_version = 1;
    }

    TagHandlerType handler_type() const override { return TagHandlerType::SHAPE; }

    void sync_from_var() override {
        mgb_throw(InternalError, "sync_from_var() called on const shape");
    }

    const InpElement* infer_withoutexc(
            VarNode** cur_active_var, bool recomp_mutable_srcnode) override {
        InpElement* ret;
        {
            // thread_local not supported on ios; so we us a manual impl
            MGB_LOCK_GUARD(sm_result_cache.mtx);
#if __DEPLOY_ON_XP_SP2__
            ret = &sm_result_cache.storage[0];
#else
            ret = &sm_result_cache.storage[std::this_thread::get_id()];
#endif
        }
        ret->m_shape = &tag()->shape();
        return ret;
    }

    TagTraitArray& deps() const override { return sm_empty_deps; }
};
StaticInferManagerImpl::TagConstShapeTrait::TagTraitArray
        StaticInferManagerImpl::TagConstShapeTrait::sm_empty_deps;
StaticInferManagerImpl::TagConstShapeTrait::InferResultCache
        StaticInferManagerImpl::TagConstShapeTrait::sm_result_cache;

//! non-const tag trait that requires inference
MGB_DEFINE_CLS_WITH_SUPER(
        StaticInferManagerImpl::TagTraitMutableBase, TagTraitBase) // {
public:
    TagTraitMutableBase(Tag tag) : Super(tag, false) {}

    /*!
     * \brief tags on which this tag depends (i.e. required to infer this
     *      tag)
     */
    const TagTraitArray& deps() const final { return m_deps; }

    /*!
     * \brief sync shape/value from corresponding var, used for
     *      dependents without shape_desc
     */
    void sync_from_var() override;

    const SharedSet<TagHandler*, TagHandlerSet>& missing_inp() {
        if (!m_initialized && !m_missing_input) {
            mgb_assert(m_infer_type == InferType::NO_DESC);
            m_missing_input.insert(this);
        }
        return m_missing_input;
    }

    /*!
     * \brief add an extra receiver to this trait; so when this trait
     *      changes, *ptr* would be marked out of sync
     */
    void add_extra_receiver(TagTraitMutableBase* ptr) {
        auto rst = m_receivers.insert(ptr);
        mgb_assert(rst.second);
    }

    /*!
     * \brief add an extra dependency
     *
     * Extra deps can only exist due to implicitly computed value through
     * sub graph, and only should be added by SubgraphStaticInferHelperImpl
     */
    void add_extra_dep(TagTraitBase* t) {
        mgb_assert(tag()->owner_graph() == t->tag()->owner_graph());
        m_deps.push_back(t);
    }

    void remove_extra_receiver(TagTraitMutableBase* ptr) {
        auto cnt = m_receivers.erase(ptr);
        mgb_assert(cnt == 1);
    }

    //! whether previous inference succeeds
    bool prev_infer_succeed() const { return m_infer_withoutexc_ret; }

    //! original deps given in the InferDesc by the caller
    virtual const DepVal& raw_deps() = 0;

protected:
    //! current infer result, to be used by dependents
    InpElement m_inp_element;

    enum class InferResult { UNCHANGED, CHANGED, FAILED };

    /*!
     * \brief infer the shape or value and update m_inp_element
     * \return whether its shape or value is actually updated
     */
    virtual InferResult do_infer(const InpVal& inp) = 0;

    /*!
     * \brief set the shape or value from corresponding VarNode
     * \return whether its shape or value is actually updated
     */
    virtual InferResult do_sync_from_var() = 0;

    /*!
     * \brief initialize deps and infer_type
     */
    void init(SourceType src_type, StaticInferManagerImpl* mgr);

    bool is_mutable_src() const {
        return m_deps.empty() && m_infer_type == InferType::RT_STATIC;
    }

    /*!
     * \brief whether init() has been called (i.e. whether infer desc is
     *      set)
     */
    bool initialized() const { return m_initialized; }

private:
    bool m_initialized = false;

    //! whether current m_inp_element reflects newest input value
    bool m_inp_element_synced = false;

    InpElement* m_infer_withoutexc_ret = nullptr;

    //! record previous run_id to skip calling infer() if input is the same
    size_t m_prev_inp_run_id = 0;

    TagTraitArray m_deps;

    ThinHashSet<TagTraitMutableBase*> m_receivers;

    //! all missing inputs
    SharedSet<TagHandler*, TagHandlerSet> m_missing_input;

    //! recursively set m_inp_element_synced of this and all receivers to
    //! false
    void reset_inp_element_synced();

    const InpElement* infer_withoutexc(
            VarNode** cur_active_var, bool recomp_mutable_srcnode) override final;
};

//! mutable shape inference
MGB_DEFINE_CLS_WITH_SUPER(
        StaticInferManagerImpl::TagShapeTrait final, TagTraitMutableBase) // {
    TensorShape m_shape;
    ShapeInferDesc m_desc;

    const DepVal& raw_deps() override { return m_desc.deps; }

    TagHandlerType handler_type() const override { return TagHandlerType::SHAPE; }

    InferResult set_shape(const TensorShape& shp);
    InferResult do_infer(const InpVal& inp) override;

    InferResult do_sync_from_var() override { return set_shape(tag()->shape()); }

public:
    using Super::Super;

    void init(const ShapeInferDesc& desc, StaticInferManagerImpl* mgr) {
        m_desc = desc;
        Super::init(desc.src_type, mgr);
    }
};

//! mutable value inference
MGB_DEFINE_CLS_WITH_SUPER(
        StaticInferManagerImpl::TagValueTrait final, TagTraitMutableBase) // {
    bool m_log_printed = false;

    //!< used for detection src value change
    TensorLayout m_prev_layout;
    DeviceTensorStorage m_prev_value;

    DeviceTensorND m_cur_value;
    ValueInferDesc m_desc;

    const DepVal& raw_deps() override { return m_desc.deps; }

    TagHandlerType handler_type() const override { return TagHandlerType::VALUE; }

    /*!
     * \brief called after finishing writing to get_writable_value()
     */
    InferResult update_value();

    InferResult do_infer(const InpVal& inp) override;

    InferResult do_sync_from_var() override {
        // strictly speaking the sync should be implemented by CompNode::Event,
        // however m_cur_value is on cpu::default, so we can just sync in this
        // caller thread
        m_cur_value.copy_from(tag()->dev_tensor().sync());
        return update_value();
    }

public:
    TagValueTrait(Tag tag)
            : Super{tag}, m_cur_value{CompNode::default_cpu(), tag->dtype()} {}

    void init(const ValueInferDesc& desc, StaticInferManagerImpl* mgr) {
        m_desc = desc;
        Super::init(desc.src_type, mgr);
    }
};

struct StaticInferManagerImpl::TagTraitContainer {
    TagTraitBase* shape;
    TagValueTrait* value;

    TagTraitBase* select(DepType type) {
        if (type == DepType::VALUE)
            return value;
        mgb_assert(type == DepType::SHAPE);
        return shape;
    }

    InferType get_infer_type() {
        return {shape ? shape->infer_type() : InferType::NO_DESC,
                value ? value->infer_type() : InferType::NO_DESC};
    }
};

/* ===================== misc ===================== */

const DeviceTensorND& InpElement::value() const {
    mgb_assert(m_value, "value not available");
    return *m_value;
}

/* ===================== TagTraitBase ===================== */

StaticInferManagerImpl::TagTraitMutableBase* StaticInferManagerImpl::TagTraitBase::
        as_mutable() {
    return is_const() ? nullptr : static_cast<TagTraitMutableBase*>(this);
}

StaticInferManagerImpl::TagTraitMutableBase* StaticInferManagerImpl::TagTraitBase::
        as_mutable_safe() {
    mgb_assert(!is_const());
    return static_cast<TagTraitMutableBase*>(this);
}

const InpElement* StaticInferManagerImpl::TagTraitBase::infer(
        bool recomp_mutable_srcnode, bool allow_fail) {
    VarNode* cur_var = nullptr;
    MGB_TRY {
        auto ret = infer_withoutexc(&cur_var, recomp_mutable_srcnode);
        if (!ret && !allow_fail) {
            // find the first var that causes infer failure
            cur_var = nullptr;
            for (auto trait = this->as_mutable_safe();;) {
                if (trait->deps().empty()) {
                    cur_var = trait->tag();
                    break;
                }
                mgb_assert(!trait->prev_infer_succeed());
                bool found = false;
                for (auto i : trait->deps()) {
                    auto imut = i->as_mutable();
                    if (imut && !imut->prev_infer_succeed()) {
                        found = true;
                        trait = imut;
                        break;
                    }
                }
                mgb_assert(found);
            }
            mgb_throw(
                    GraphError,
                    "failed to perform static inference for var%s\n"
                    "NOTE: this is caused by var%s",
                    cg::dump_var_info({tag()}).c_str(),
                    cg::dump_var_info({cur_var}).c_str());
        }
        return ret;
    }
    MGB_CATCH(MegBrainError & exc, {
        if (!cur_var) {
            cur_var = tag();
        }
        update_rethrow_exc(cur_var, exc);
    })
}

/* ===================== TagTraitDepIter ===================== */
/*!
 * \brief iterate over the dependencies of traits in topological order
 *
 * Note:
 *  If \p cb_pre is empty, a default impl checking the VisitedSet would be used.
 *  If \p cb_post is empty, no action would be taken when visiting of a trait
 *  finishs.
 */
class StaticInferManagerImpl::TagTraitDepIter {
public:
    using VisitedSet = ThinHashSet<TagTraitBase*>;
    /*!
     * callback for before visiting a trait; it must return a bool indicating
     * whether this tag should be visited
     */
    using CallbackPre = thin_function<bool(VisitedSet& visited, TagTraitBase*)>;

    //! callback for after visiting a trait
    using CallbackPost = thin_function<void(TagTraitBase*)>;

    explicit TagTraitDepIter(CallbackPre cb_pre, CallbackPost cb_post)
            : m_cb_pre{std::move(cb_pre)}, m_cb_post{std::move(cb_post)} {
        if (!m_cb_pre) {
            m_cb_pre = [](VisitedSet& visited, TagTraitBase* trait) {
                return visited.insert(trait).second;
            };
        }
        if (!m_cb_post) {
            m_cb_post = [](TagTraitBase*) {};
        }
    }

    void add(TagTraitBase* trait);

private:
    struct Frame {
        TagTraitBase* trait;
        TagTraitBase* const* deps;
        TagTraitBase* const* deps_end;
    };

    SmallVector<Frame, 1024> m_stack;
    CallbackPre m_cb_pre;
    CallbackPost m_cb_post;
    ThinHashSet<TagTraitBase*> m_visited;

    void push_stack(TagTraitBase* trait);
};

void StaticInferManagerImpl::TagTraitDepIter::add(TagTraitBase* trait) {
    push_stack(trait);
    while (!m_stack.empty()) {
        auto&& frame = m_stack.back();
        if (frame.deps == frame.deps_end) {
            m_cb_post(frame.trait);
            m_stack.pop_back();
        } else {
            auto next = *(frame.deps++);
            push_stack(next);
        }
    }
}

void StaticInferManagerImpl::TagTraitDepIter::push_stack(TagTraitBase* trait) {
    if (m_cb_pre(m_visited, trait)) {
        auto&& deps = trait->deps();
        m_stack.push_back({trait, deps.data(), deps.data() + deps.size()});
    }
}

/* ===================== TagTraitMutableBase ===================== */

void StaticInferManagerImpl::TagTraitMutableBase::init(
        SourceType src_type, StaticInferManagerImpl* mgr) {
    mgb_assert(!m_initialized, "can not overwrite infer desc");
    m_initialized = true;

    if (src_type == SourceType::CONSTANT) {
        mgb_assert(raw_deps().empty());
        m_infer_type = InferType::CONST;
        return;
    }

    if (src_type == SourceType::MUTABLE) {
        mgb_assert(raw_deps().empty());
        m_infer_type = InferType::RT_STATIC;
        return;
    }

    mgb_assert(src_type == SourceType::DEP && !raw_deps().empty());

    for (auto&& i : raw_deps()) {
        auto dst0 = mgr->get_tag_trait_for_dep(i);

        m_deps.push_back(dst0);
        if (dst0->is_const()) {
            m_infer_type = std::max(m_infer_type, InferType::CONST);
            continue;
        }

        auto dst = static_cast<TagTraitMutableBase*>(dst0);
        dst->m_receivers.insert(this);

        // compute infer type and missing_inp
        if (!dst->m_initialized) {
            // dst has no infer desc
            mgb_assert(dst->m_infer_type == InferType::NO_DESC);
            m_infer_type = InferType::MISSING_INP;
            m_missing_input.merge_from(dst->missing_inp());
        } else {
            mgb_assert(dst->m_infer_type != InferType::NO_DESC);
            m_infer_type = std::max(m_infer_type, dst->infer_type());
            if (dst->infer_type() == InferType::MISSING_INP)
                m_missing_input.merge_from(dst->missing_inp());
        }
    }

    mgb_assert(m_infer_type != InferType::NO_DESC);
    if (m_infer_type != InferType::MISSING_INP)
        mgb_assert(!m_missing_input);
}

const InpElement* StaticInferManagerImpl::TagTraitMutableBase::infer_withoutexc(
        VarNode** cur_active_var, bool recomp_mutable_srcnode) {
    InpVal inp_val;

    auto infer_single_core =
            [&inp_val, cur_active_var](TagTraitMutableBase* trait) -> InpElement* {
        inp_val.run_id = 0;
        inp_val.val.clear();

        // all dependencies should have been processed due to topological iter
        // order, so we only check if any of them fails
        for (auto&& dep : trait->m_deps) {
            const InpElement* cur_inp;
            if (dep->is_const()) {
                cur_inp = dep->infer_withoutexc(cur_active_var, false);
            } else {
                auto dt = static_cast<TagTraitMutableBase*>(dep);
                cur_inp = dt->m_infer_withoutexc_ret;
                if (!cur_inp) {
                    return nullptr;
                }
            }
            inp_val.val.push_back(*cur_inp);
            inp_val.run_id += dep->infer_result_version();
        }

        if (!trait->deps().empty() && inp_val.run_id == trait->m_prev_inp_run_id) {
            // inputs unchanged, and middle nodes are required to be pure
            return &trait->m_inp_element;
        }
        *cur_active_var = trait->tag();
        auto rst = trait->do_infer(inp_val);
        if (rst == InferResult::FAILED) {
            // intermediate traits should never fail (already checked in
            // do_infer())
            mgb_assert(trait->deps().empty());
            return nullptr;
        }

        trait->m_prev_inp_run_id = inp_val.run_id;
        if (rst == InferResult::CHANGED) {
            ++trait->m_inp_element_version;
            trait->reset_inp_element_synced();
        }
        return &trait->m_inp_element;
    };
    auto infer_single = [infer_single_core](TagTraitBase* trait_) {
        auto trait = static_cast<TagTraitMutableBase*>(trait_);
        trait->m_infer_withoutexc_ret = infer_single_core(trait);
        trait->m_inp_element_synced = true;
    };

    if (recomp_mutable_srcnode) {
        mgb_assert(is_mutable_src());
        infer_single(this);
        return m_infer_withoutexc_ret;
    }

    if (m_inp_element_synced) {
        return m_infer_withoutexc_ret;
    }

    auto cb_pre = [](TagTraitDepIter::VisitedSet&, TagTraitBase* trait_) {
        auto trait = trait_->as_mutable();
        // m_inp_element_synced would be set to true after processing the trait,
        // so it can be used as the visit mark
        return trait && !trait->m_inp_element_synced;
    };
    TagTraitDepIter dep_iter{cb_pre, infer_single};
    dep_iter.add(this);
    return m_infer_withoutexc_ret;
}

void StaticInferManagerImpl::TagTraitMutableBase::sync_from_var() {
    mgb_assert(!m_initialized && m_infer_type == InferType::NO_DESC);
    auto rst = do_sync_from_var();
    mgb_assert(rst != InferResult::FAILED);
    if (rst == InferResult::CHANGED) {
        ++m_inp_element_version;
        reset_inp_element_synced();
    }
}

void StaticInferManagerImpl::TagTraitMutableBase::reset_inp_element_synced() {
    if (!m_inp_element_synced) {
        return;
    }
    m_inp_element_synced = false;
    SmallVector<TagTraitMutableBase*, 1024> stack{this};
    while (!stack.empty()) {
        auto top = stack.back();
        stack.pop_back();
        for (auto i : top->m_receivers) {
            if (i->m_inp_element_synced) {
                i->m_inp_element_synced = false;
                stack.push_back(i);
            }
        }
    }
}

/* ===================== TagShapeTrait ===================== */

StaticInferManagerImpl::TagShapeTrait::InferResult StaticInferManagerImpl::
        TagShapeTrait::set_shape(const TensorShape& shp) {
    mgb_assert(shp.ndim || tag()->contain_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE));
    m_inp_element.m_shape = &m_shape;
    if (shp.eq_shape(m_shape))
        return InferResult::UNCHANGED;
#if LOG_INFER_RESULT
    mgb_log_debug(
            "shape changed: %s: %s", cg::dump_var_info({tag()}).c_str(),
            shp.to_string().c_str());
#endif
    m_shape = shp;
    return InferResult::CHANGED;
}

StaticInferManagerImpl::TagShapeTrait::InferResult StaticInferManagerImpl::
        TagShapeTrait::do_infer(const InpVal& inp) {
    if (!initialized()) {
        if (!m_shape.ndim) {
            mgb_log_debug(
                    "uninitialized shape during static infer: var=%s",
                    cg::dump_var_info({tag()}).c_str());
            return InferResult::FAILED;
        }
        return InferResult::UNCHANGED;
    }
    TensorShape dest;
    bool succ = m_desc.infer_func(dest, inp);
    if (!succ) {
        mgb_assert(is_mutable_src(), "infer failed for non-mutable src tag");
        return InferResult::FAILED;
    }
    return set_shape(dest);
}

/* ===================== TagValueTrait ===================== */

StaticInferManagerImpl::TagValueTrait::InferResult StaticInferManagerImpl::
        TagValueTrait::update_value() {
    m_inp_element.m_value = &m_cur_value;
    mgb_assert(
            m_cur_value.comp_node() == CompNode::default_cpu() &&
            m_cur_value.layout().ndim && m_cur_value.dtype() == tag()->dtype());

    auto span = m_cur_value.layout().span();
    if (span.dist_elem() >= INFER_VALUE_SIZE_THRESH_FOR_WARNING && !m_log_printed) {
        mgb_log_debug(
                "compute static_infer_value() for %s: "
                "span dist too large (%zu)",
                cg::dump_var_info({tag()}).c_str(), span.dist_byte());
        m_log_printed = true;
    }

    // check value change for src nodes and small mid nodes
    if (deps().empty() ||
        m_cur_value.shape().total_nr_elems() <= INFER_VALUE_CHECK_UNCHANGE_MAX_SIZE) {
        if (!m_cur_value.layout().is_contiguous_allow_brdcst()) {
            DeviceTensorND tmp;
            tmp.copy_from(m_cur_value);
            std::swap(m_cur_value, tmp);
        }
        auto&& cur_storage = m_cur_value.storage();
        auto sz = m_cur_value.layout().span().dist_byte();
        if (m_prev_layout.ndim && m_prev_layout.eq_layout(m_cur_value.layout())) {
            mgb_assert(sz <= m_prev_value.size());
            if (!memcmp(cur_storage.ptr(), m_prev_value.ptr(), sz))
                return InferResult::UNCHANGED;
        }
        m_prev_layout = m_cur_value.layout();
        m_prev_value.comp_node(cur_storage.comp_node()).ensure_size(sz);
        memcpy(m_prev_value.ptr(), cur_storage.ptr(), sz);
    } else {
        m_prev_layout.ndim = 0;
    }

#if LOG_INFER_RESULT
    auto&& val = m_cur_value;
    auto vstr = ssprintf("shape=%s value={", val.shape().to_string().c_str());
    for (float v : tensor_iter_valonly(val))
        vstr.append(ssprintf("%.3g, ", v));
    vstr.pop_back();
    vstr.back() = '}';
    mgb_log_debug(
            "value changed: %s: %s", cg::dump_var_info({tag()}).c_str(), vstr.c_str());
#endif
    return InferResult::CHANGED;
}

StaticInferManagerImpl::TagValueTrait::InferResult StaticInferManagerImpl::
        TagValueTrait::do_infer(const InpVal& inp) {
    if (!initialized()) {
        if (m_cur_value.empty()) {
            mgb_log_debug(
                    "uninitialized value during static infer: var=%s",
                    cg::dump_var_info({tag()}).c_str());
            return InferResult::FAILED;
        }
        return InferResult::UNCHANGED;
    }
    bool succ = m_desc.infer_func(m_cur_value, inp);
    if (!succ) {
        mgb_assert(
                is_mutable_src(), "infer failed for non-mutable src tag: var: %s",
                cg::dump_var_info({tag()}).c_str());
        return InferResult::FAILED;
    }
    return update_value();
}

/* ===================== StaticInferManagerImpl ===================== */

StaticInferManagerImpl::~StaticInferManagerImpl() noexcept {
    m_mem_pool_shape_trait.disable_freelist();
    m_mem_pool_value_trait.disable_freelist();
    for (auto&& i : m_dtor_callbacks)
        i.second();
    for (auto&& i : ComputingGraphImpl::downcast(m_owner_graph)->all_oprs()) {
        for (auto j : i->output()) {
            clear_tag_handler(j);
        }
    }
}

void StaticInferManagerImpl::clear_tag_handler(Tag tag) {
    auto&& container = get_tag_trait_container(tag);
    if (auto s = container.shape) {
        if (s->is_const()) {
            m_mem_pool_const_shape_trait.free(static_cast<TagConstShapeTrait*>(s));
        } else {
            m_mem_pool_shape_trait.free(static_cast<TagShapeTrait*>(s));
        }
        container.shape = nullptr;
    }
    if (container.value) {
        m_mem_pool_value_trait.free(container.value);
        container.value = nullptr;
    }
}

StaticInferManagerImpl::TagTraitContainer& StaticInferManagerImpl::
        get_tag_trait_container(Tag tag) {
    static_assert(
            sizeof(tag->m_static_infer_trait) == sizeof(TagTraitContainer) &&
                    alignof(std::remove_reference<
                            decltype(tag->m_static_infer_trait)>::type) ==
                            alignof(TagTraitContainer),
            "bad size");
    return *aliased_ptr<TagTraitContainer>(&tag->m_static_infer_trait);
}

void StaticInferManagerImpl::register_shape_infer(
        Tag dest, const ShapeInferDesc& desc) {
    mgb_assert(dest->owner_opr() == m_register_allowed_opr);
    for (auto&& i : desc.deps)
        mgb_assert(dest->owner_graph() == i.dest->owner_graph());

    auto&& t = get_tag_trait_container(dest);
    mgb_assert(!t.shape, "shape desc already inserted");
    auto ptr = m_mem_pool_shape_trait.alloc_unique(dest);
    ptr->init(desc, this);
    if (ptr->infer_type() == InferType::CONST &&
        !dest->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
        // infer const shapes immediately
        auto r = ptr->infer(false, false);
        mgb_assert(r && r->m_shape);
        dest->shape(r->shape());
        dest->m_allow_shape_change = false;
        t.shape = m_mem_pool_const_shape_trait.alloc(dest);
    } else {
        t.shape = ptr.release();
    }
}

void StaticInferManagerImpl::register_value_infer(
        Tag dest, const ValueInferDesc& desc) {
    mgb_assert(dest->owner_opr() == m_register_allowed_opr);
    for (auto&& i : desc.deps)
        mgb_assert(dest->owner_graph() == i.dest->owner_graph());

    auto&& t = get_tag_trait_container(dest);
    mgb_assert(t.shape, "shape desc not inserted before value desc");
    mgb_assert(!t.value, "value infer already registered");
    t.value = m_mem_pool_value_trait.alloc(dest);
    t.value->init(desc, this);
}

InferType StaticInferManagerImpl::get_infer_type(Tag dest) {
    return get_tag_trait_container(dest).get_infer_type();
}

const TensorShape& StaticInferManagerImpl::infer_shape(Tag dest) {
    return *do_infer_shape(dest, false);
}

const TensorShape* StaticInferManagerImpl::infer_shape_fallible(Tag dest) {
    return do_infer_shape(dest, true);
}

const DeviceTensorND& StaticInferManagerImpl::infer_value(Tag dest) {
    return *do_infer_value(dest, false);
}

const DeviceTensorND* StaticInferManagerImpl::infer_value_fallible(Tag dest) {
    return do_infer_value(dest, true);
}

const StaticInferManagerImpl::TagHandlerSet& StaticInferManagerImpl::get_missing_inp(
        TagHandler* dest_) {
    auto dest = static_cast<TagTraitBase*>(dest_)->as_mutable_safe();
    mgb_assert(dest->infer_type() & (InferType::NO_DESC | InferType::MISSING_INP));
    auto ptr = dest->missing_inp().get();
    mgb_assert(ptr);
    return *ptr;
}

StaticInferManagerImpl::TagHandler* StaticInferManagerImpl::get_tag_handler_for_shape(
        Tag tag) {
    auto&& c = get_tag_trait_container(tag);
    if (!c.shape) {
        c.shape = m_mem_pool_shape_trait.alloc(tag);
    }
    return c.shape;
}

StaticInferManagerImpl::TagHandler* StaticInferManagerImpl::get_tag_handler_for_value(
        Tag tag) {
    auto&& c = get_tag_trait_container(tag);
    if (!c.value) {
        c.value = m_mem_pool_value_trait.alloc(tag);
    }
    return c.value;
}

StaticInferManagerImpl::TagTraitBase* StaticInferManagerImpl::get_tag_trait_for_dep(
        const DepElement& dep) {
    TagHandler* ret;
    switch (dep.type) {
        case DepType::SHAPE:
            ret = get_tag_handler_for_shape(dep.dest);
            break;
        case DepType::VALUE:
            ret = get_tag_handler_for_value(dep.dest);
            break;
        default:
            mgb_assert(0, "bad dep type");
    }
    return static_cast<TagTraitBase*>(ret);
}

DepVal StaticInferManagerImpl::get_rt_static_source_deps(const DepElement& dest) {
    auto trait_base = get_tag_trait_container(dest.dest).select(dest.type);
    if (!trait_base || trait_base->is_const())
        return {};

    auto trait = static_cast<TagTraitMutableBase*>(trait_base);

    mgb_assert(is_static_infer_type(trait->infer_type()));

    DepVal result;
    auto cb_pre = [&](TagTraitDepIter::VisitedSet& visited, TagTraitBase* trait) {
        if (!trait->is_const() && visited.insert(trait).second) {
            if (trait->deps().empty() && trait->infer_type() == InferType::RT_STATIC) {
                result.push_back({trait->tag(), trait->handler_type()});
                return false;
            }
            return true;
        }
        return false;
    };
    TagTraitDepIter iter{cb_pre, {}};
    iter.add(trait);
    return result;
}

const TensorShape* StaticInferManagerImpl::do_infer_shape(Tag dest, bool allow_fail) {
    MGB_LOCK_GUARD(m_mtx);
    MGB_TRY {
        auto&& container = get_tag_trait_container(dest);
        mgb_assert(
                container.shape,
                "infer desc for var has not been added for infer_shape: %s",
                cg::dump_var_info({dest}).c_str());
        auto ret = container.shape->infer(false, allow_fail);
        if (!ret) {
            mgb_assert(allow_fail);
            return nullptr;
        }
        return &ret->shape();
    }
    MGB_CATCH(MegBrainError & exc, { update_rethrow_exc(dest, exc); })
}

const DeviceTensorND* StaticInferManagerImpl::do_infer_value(
        Tag dest, bool allow_fail) {
    MGB_LOCK_GUARD(m_mtx);
    MGB_TRY {
        auto&& container = get_tag_trait_container(dest);
        mgb_assert(
                container.value,
                "infer desc for var has not been added for infer_value: %s",
                cg::dump_var_info({dest}).c_str());
        auto ret = container.value->infer(false, allow_fail);
        if (!ret) {
            mgb_assert(allow_fail);
            return nullptr;
        }
        return &ret->value();
    }
    MGB_CATCH(MegBrainError & exc, { update_rethrow_exc(dest, exc); })
}

void StaticInferManagerImpl::update_mutable_src_shape(Tag dest) {
    MGB_LOCK_GUARD(m_mtx);
    MGB_TRY {
        auto&& container = get_tag_trait_container(dest);
        auto handle = container.shape;
        if (handle && handle->infer_type() == InferType::RT_STATIC &&
            handle->deps().empty()) {
            dest->shape(handle->infer(true, false)->shape());
        }
    }
    MGB_CATCH(MegBrainError & exc, { update_rethrow_exc(dest, exc); })
}

DepVal StaticInferManagerImpl::get_deps(const DepElement& elem) {
    auto trait_base = get_tag_trait_container(elem.dest).select(elem.type);
    if (!trait_base || trait_base->is_const())
        return {};

    return trait_base->as_mutable_safe()->raw_deps();
}

/* ===================== CompSeqManager ===================== */

class CompSeqManager::VersionedTagTrait {
    TagTraitBase* const m_trait;
    size_t m_version = 0;

public:
    VersionedTagTrait(TagTraitBase* trait) : m_trait{trait} {}

    /*!
     * \brief re-infer and assign shape
     * \return <whether version changed, whether shape changed>
     */
    std::pair<bool, bool> update(bool recomp_mutable_srcnode);

    TagTraitBase* trait() const { return m_trait; }
};

std::pair<bool, bool> CompSeqManager::VersionedTagTrait::update(
        bool recomp_mutable_srcnode) {
    auto rst = m_trait->infer(recomp_mutable_srcnode, false);
    auto version = m_trait->infer_result_version();

    if (version != m_version) {
        bool shp = false;
        if (m_trait->handler_type() == TagHandlerType::SHAPE) {
            m_trait->tag()->shape(rst->shape());
            shp = true;
        }
        m_version = version;
        return {true, shp};
    }

    return {false, false};
}

CompSeqManager::CompSeqManager(ComputingGraph* graph) : m_owner_graph(graph) {}

CompSeqManager::~CompSeqManager() noexcept = default;

void CompSeqManager::add_dest(CompSeqExtraInfo& info, TagTraitBase* dest) {
    if (!m_added.insert(dest).second)
        return;

    auto&& queue = m_add_dest_queue;
    queue.clear();
    queue.push_back(dest);

    while (!queue.empty()) {
        auto qh = queue.front();
        queue.pop_front();
        mgb_assert(qh->tag()->owner_graph() == m_owner_graph);

        for (auto i : qh->deps()) {
            if (m_added.insert(i).second)
                queue.push_back(i);
        }

        switch (qh->infer_type()) {
            case InferType::CONST:
                if (!qh->is_const()) {
                    // shape already updated for qh being const
                    m_static_infer_const_needed.emplace_back(qh);
                }
                break;
            case InferType::NO_DESC:
                // record this as a missing input
                if (qh->handler_type() == TagHandlerType::SHAPE)
                    info.missing_for_shape.insert(qh->tag());
                else {
                    mgb_assert(qh->handler_type() == TagHandlerType::VALUE);
                    info.missing_for_value.insert(qh->tag());
                }
                break;
            case InferType::RT_STATIC:
                if (qh->deps().empty()) {
                    m_static_srcnode.emplace_back(qh);
                } else {
                    m_static_mid.emplace_back(qh);
                }
            case InferType::MISSING_INP:
                // its missing inputs have been recorded, and this tag would be
                // inferred on demand when the operator asks for its value
                break;
            default:
                mgb_throw(MegBrainError, "bad infer type");
        }
    }
}

void CompSeqManager::reset_dest(CompSeqExtraInfo& info) {
    m_static_first_run = true;
    m_added.clear();
    m_static_infer_const_needed.clear();
    m_static_srcnode.clear();
    m_static_mid.clear();
    info.missing_for_shape.clear();
    info.missing_for_value.clear();

    for (auto&& i : info.infer_dest) {
        mgb_assert(i->tag()->owner_graph() == m_owner_graph);
        add_dest(info, static_cast<TagTraitBase*>(i));
    }

    info.rt_static_infer_src.clear();
    for (auto&& i : m_static_srcnode) {
        auto trait = i.trait();
        if (trait->infer_type() & InferType::RT_STATIC) {
            info.rt_static_infer_src.push_back({trait->tag(), trait->handler_type()});
        }
    }
}

bool CompSeqManager::update_static_check_shape_change() {
    if (m_static_first_run) {
        for (auto&& i : m_static_infer_const_needed)
            i.update(false);
    }
    bool src_changed = false, shape_changed = false;
    for (auto&& i : m_static_srcnode) {
        auto cur = i.update(true);
        src_changed |= cur.first;
        shape_changed |= cur.second;
    }
    if (!src_changed && !m_static_first_run)
        return false;

    for (auto&& i : m_static_mid) {
        shape_changed |= i.update(false).second;
    }
    m_static_first_run = false;
    return shape_changed;
}

/* ===================== SubgraphStaticInferHelperImpl  ===================== */

/*
 * The basic idea is to manage deps of vars in subgraph by this helper class,
 * and the deps would NOT be known by StaticInferManagerImpl (so the tags appear
 * as MUTABLE or CONST sources in their corresponding graphs).
 *
 * This helper is necessary (i.e. static infer manager could be shared by parent
 * and sub graphs) because a trait may be statically inferable in sub graph but
 * not so in parent graph.
 */
class StaticInferManagerImpl::SubgraphStaticInferHelperImpl final
        : public SubgraphStaticInferHelper {
    using TagTraitArray = TagTraitBase::TagTraitArray;
    using RegisterSubgrahInferCallback = thin_function<TagTraitBase*(
            StaticInferManagerImpl& mgr, SourceType src_type,
            const TagTraitArray& par_deps)>;
    typedef bool (SubgraphStaticInferHelperImpl::*RegisterHelperPtr)(
            Tag, const DepVal&, RegisterSubgrahInferCallback);

    //! par graph dependency traits of a subgraph trait
    struct SubgraphTraitDepInPar {
        bool only_static_dep = false;
        SharedSet<TagTraitBase*> static_deps;
    };

    bool m_par_destructed = false;

    //! traits registered as extra receiver in parent graph; used deregstering
    //! the sub graph
    std::vector<std::pair<TagTraitMutableBase*, TagTraitMutableBase*>>
            m_registered_in_par_graph_receiver;

    ComputingGraphImpl *m_sub_graph = nullptr, *m_par_graph = nullptr;

    ThinHashMap<TagTraitBase*, SubgraphTraitDepInPar> m_sub_trait_dep_in_par;

    void check_graph_par(VarNode* var) {
        if (mgb_unlikely(!m_par_graph)) {
            m_par_graph = ComputingGraphImpl::downcast(var->owner_graph());
            mgb_assert(m_par_graph != m_sub_graph);

            auto cb = [this]() { m_par_destructed = true; };

            auto ins = m_par_graph->static_infer_manager_impl().m_dtor_callbacks.insert(
                    {this, cb});
            mgb_assert(ins.second);

        } else {
            mgb_assert(m_par_graph == var->owner_graph());
        }
    }

    void check_graph_sub(VarNode* var) {
        if (mgb_unlikely(!m_sub_graph)) {
            m_sub_graph = ComputingGraphImpl::downcast(var->owner_graph());
            mgb_assert(m_sub_graph != m_par_graph);
        } else {
            mgb_assert(m_sub_graph == var->owner_graph());
        }
    }

    /*!
     * \brief helper to implement registering infer func for a var in subgraph
     * \param user_deps deps given by user
     * \param callback register proxy infer func in sub manager, and should
     *      return tag trait for dest
     * \return true
     */
    bool helper_register_infer_sub(
            Tag dest, const DepVal& user_deps, RegisterSubgrahInferCallback callback);

    /*!
     * \brief helper to implement registering infer func for a var in par graph
     *
     * The infer func would be registered only if all deps are statically
     * inferable from par.
     *
     * See helper_register_infer_sub for more details.
     *
     * \return whether infer func is registered
     */
    bool helper_register_infer_par(
            Tag dest, const DepVal& user_deps, RegisterSubgrahInferCallback callback);

    bool call_register_for_shape(
            Tag dest, const ShapeInferDesc& desc, RegisterHelperPtr helper);

    bool call_register_for_value(
            Tag dest, const ValueInferDesc& desc, RegisterHelperPtr helper);

    /*!
     * \brief check whether a trait in subgraph only has static deps in par
     *      graph
     */
    const SubgraphTraitDepInPar& get_sub_trait_dep_in_par(TagTraitBase* trait);

    static InpVal prepare_inp_val(const TagTraitArray& deps);

    static bool infer_shape_raw(
            const TagTraitArray& deps, const ShapeInferDesc::infer_func_t& func,
            TensorShape& dest, const InpVal&);

    static bool infer_value_raw(
            const TagTraitArray& deps, const ValueInferDesc::infer_func_t& func,
            DeviceTensorND& dest, const InpVal&);

public:
    ~SubgraphStaticInferHelperImpl() {
        if (m_par_destructed || !m_par_graph)
            return;

        for (auto&& i : m_registered_in_par_graph_receiver)
            i.first->remove_extra_receiver(i.second);
        auto cnt =
                m_par_graph->static_infer_manager_impl().m_dtor_callbacks.erase(this);
        mgb_assert(cnt == 1);
    }

    void register_shape_infer_sub(Tag dest, const ShapeInferDesc& desc) override {
        call_register_for_shape(
                dest, desc, &SubgraphStaticInferHelperImpl::helper_register_infer_sub);
    }

    void register_value_infer_sub(Tag dest, const ValueInferDesc& desc) override {
        call_register_for_value(
                dest, desc, &SubgraphStaticInferHelperImpl::helper_register_infer_sub);
    }

    bool register_shape_infer_par(Tag dest, const ShapeInferDesc& desc) override {
        return call_register_for_shape(
                dest, desc, &SubgraphStaticInferHelperImpl::helper_register_infer_par);
    }

    bool register_value_infer_par(Tag dest, const ValueInferDesc& desc) override {
        return call_register_for_value(
                dest, desc, &SubgraphStaticInferHelperImpl::helper_register_infer_par);
    }
};

bool StaticInferManagerImpl::SubgraphStaticInferHelperImpl::helper_register_infer_sub(
        Tag dest, const DepVal& user_deps, RegisterSubgrahInferCallback callback) {
    check_graph_sub(dest);
    mgb_assert(!user_deps.empty());

    bool is_const = true, is_static = true;
    TagTraitArray deps;  // dependency in par graph
    for (auto&& i : user_deps) {
        check_graph_par(i.dest);

        auto&& par_mgr = m_par_graph->static_infer_manager_impl();

        InferType::Flag infer_type;
        {
            auto t = par_mgr.get_infer_type(i.dest);
            if (i.type == DepType::SHAPE) {
                infer_type = t.shape;
            } else {
                mgb_assert(i.type == DepType::VALUE);
                infer_type = t.value;
            }
        }
        is_static &= is_static_infer_type(infer_type);
        is_const &= ((infer_type & InferType::CONST) != 0);
        deps.push_back(par_mgr.get_tag_trait_for_dep(i));
    }

    auto&& sub_mgr = m_sub_graph->static_infer_manager_impl();

    auto dest_trait = callback(
            sub_mgr, is_const ? SourceType::CONSTANT : SourceType::MUTABLE, deps);

    auto&& dep_info = m_sub_trait_dep_in_par[dest_trait];
    dep_info.only_static_dep = is_static;
    if (is_static) {
        for (auto i : deps)
            dep_info.static_deps.insert(i);
    }
    if (!is_const) {
        auto non_const_dt = dest_trait->as_mutable_safe();
        for (auto i0 : deps) {
            if (auto i = i0->as_mutable()) {
                i->add_extra_receiver(non_const_dt);
                m_registered_in_par_graph_receiver.emplace_back(i, non_const_dt);
            }
        }
    }

    return true;
}

bool StaticInferManagerImpl::SubgraphStaticInferHelperImpl::helper_register_infer_par(
        Tag dest, const DepVal& user_deps, RegisterSubgrahInferCallback callback) {
    mgb_assert(m_sub_graph && m_par_graph);
    check_graph_par(dest);

    auto&& sub_mgr = m_sub_graph->static_infer_manager_impl();
    auto&& par_mgr = m_par_graph->static_infer_manager_impl();

    TagTraitArray deps;
    bool is_const = true;

    TagTraitArray extra_par_deps;  // deps in user_deps in par graph

    for (auto&& i : user_deps) {
        auto iog = i.dest->owner_graph();
        mgb_assert(iog == m_sub_graph || iog == m_par_graph);
        TagTraitBase* cur_trait;
        if (iog == m_sub_graph) {
            cur_trait = sub_mgr.get_tag_trait_for_dep(i);
            auto&& dep_info = get_sub_trait_dep_in_par(cur_trait);
            if (!dep_info.only_static_dep)
                return false;
            for (auto i : dep_info.static_deps)
                extra_par_deps.push_back(i);
        } else {
            cur_trait = par_mgr.get_tag_trait_for_dep(i);
            extra_par_deps.push_back(cur_trait);
        }
        is_const &= ((cur_trait->infer_type() & InferType::CONST) != 0);
        deps.push_back(cur_trait);
    }

    auto dest_trait = callback(
            par_mgr, is_const ? SourceType::CONSTANT : SourceType::MUTABLE, deps);

    if (!is_const) {
        auto non_const_dt = dest_trait->as_mutable_safe();
        for (auto i0 : extra_par_deps) {
            if (auto i = i0->as_mutable()) {
                i->add_extra_receiver(non_const_dt);
                m_registered_in_par_graph_receiver.emplace_back(i, non_const_dt);
                non_const_dt->add_extra_dep(i);
            }
        }
    }

    return true;
}

bool StaticInferManagerImpl::SubgraphStaticInferHelperImpl::call_register_for_shape(
        Tag dest, const ShapeInferDesc& desc, RegisterHelperPtr helper) {
    mgb_assert(desc.src_type == SourceType::DEP);

    auto callback = [&](StaticInferManagerImpl& mgr, SourceType src_type,
                        const TagTraitArray& deps) -> TagTraitBase* {
        using namespace std::placeholders;
        auto f = std::bind(
                &SubgraphStaticInferHelperImpl::infer_shape_raw, deps, desc.infer_func,
                _1, _2);

        mgr.register_shape_infer(dest, {src_type, {}, f});
        return mgr.get_tag_trait_container(dest).shape;
    };

    return (this->*helper)(dest, desc.deps, callback);
}

bool StaticInferManagerImpl::SubgraphStaticInferHelperImpl::call_register_for_value(
        Tag dest, const ValueInferDesc& desc, RegisterHelperPtr helper) {
    mgb_assert(desc.src_type == SourceType::DEP);

    auto callback = [&](StaticInferManagerImpl& mgr, SourceType src_type,
                        const TagTraitArray& deps) -> TagTraitBase* {
        using namespace std::placeholders;
        auto f = std::bind(
                &SubgraphStaticInferHelperImpl::infer_value_raw, deps, desc.infer_func,
                _1, _2);

        mgr.register_value_infer(dest, {src_type, {}, f});
        return mgr.get_tag_trait_container(dest).value;
    };

    return (this->*helper)(dest, desc.deps, callback);
}

InpVal StaticInferManagerImpl::SubgraphStaticInferHelperImpl::prepare_inp_val(
        const TagTraitArray& deps) {
    mgb_assert(!deps.empty());
    InpVal finp;
    for (auto i : deps) {
        auto t = i->infer(false, true);
        if (!t) {
            finp.val.clear();
            return finp;
        }
        finp.val.push_back(*t);
        finp.run_id += i->infer_result_version();
    }
    return finp;
}

bool StaticInferManagerImpl::SubgraphStaticInferHelperImpl::infer_shape_raw(
        const TagTraitArray& deps, const ShapeInferDesc::infer_func_t& func,
        TensorShape& dest, const InpVal&) {
    auto finp = prepare_inp_val(deps);
    if (finp.val.empty())
        return false;
    auto succ = func(dest, finp);
    mgb_assert(succ);
    return succ;
}

bool StaticInferManagerImpl::SubgraphStaticInferHelperImpl::infer_value_raw(
        const TagTraitArray& deps, const ValueInferDesc::infer_func_t& func,
        DeviceTensorND& dest, const InpVal&) {
    auto finp = prepare_inp_val(deps);
    if (finp.val.empty())
        return false;
    auto succ = func(dest, finp);
    mgb_assert(succ);
    return succ;
}

const StaticInferManagerImpl::SubgraphStaticInferHelperImpl::SubgraphTraitDepInPar&
StaticInferManagerImpl::SubgraphStaticInferHelperImpl::get_sub_trait_dep_in_par(
        TagTraitBase* trait) {
    auto iter = m_sub_trait_dep_in_par.find(trait);
    if (iter != m_sub_trait_dep_in_par.end())
        return iter->second;

    auto&& rst = m_sub_trait_dep_in_par[trait];
    if (trait->deps().empty()) {
        rst.only_static_dep = trait->infer_type() == InferType::CONST;
    } else {
        rst.only_static_dep = true;
        for (auto i : trait->deps()) {
            if (!get_sub_trait_dep_in_par(i).only_static_dep) {
                rst.only_static_dep = false;
                break;
            }
        }

        if (rst.only_static_dep) {
            for (auto i : trait->deps()) {
                auto&& t = m_sub_trait_dep_in_par.at(i);
                rst.static_deps.merge_from(t.static_deps);
            }
        }
    }
    return rst;
}

std::unique_ptr<SubgraphStaticInferHelper> SubgraphStaticInferHelper::make() {
    return std::unique_ptr<SubgraphStaticInferHelper>(
            new StaticInferManagerImpl::SubgraphStaticInferHelperImpl);
}

/* ===================== StaticInferUpdaterImpl ===================== */
class StaticInferManagerImpl::StaticInferUpdaterImpl final : public StaticInferUpdater {
    StaticInferManagerImpl* m_mgr = nullptr;
    bool m_build_done = false;
    SmallVector<TagTraitMutableBase*> m_src, m_dst;

    void build() {
        auto cb_pre = [this](TagTraitDepIter::VisitedSet& visited,
                             TagTraitBase* trait) {
            if (!trait->is_const() && visited.insert(trait).second) {
                if (trait->deps().empty() &&
                    trait->infer_type() == InferType::RT_STATIC) {
                    m_src.push_back(static_cast<TagTraitMutableBase*>(trait));
                    return false;
                }
                return true;
            }
            return false;
        };
        TagTraitDepIter dep_iter{cb_pre, {}};
        for (auto i : m_dst) {
            dep_iter.add(i);
        }
    }

public:
    StaticInferUpdater& add_dest(const DepElement& dest) override {
        mgb_throw_if(
                m_build_done, GraphError,
                "add_dest() can not be called after update()");
        auto mgr = static_cast<StaticInferManagerImpl*>(
                &dest.dest->owner_graph()->static_infer_manager());
        if (!m_mgr) {
            m_mgr = mgr;
        } else {
            mgb_throw_if(
                    m_mgr != mgr, GraphError,
                    "computing graph in StaticInferUpdater changes");
        }

        auto trait_base = mgr->get_tag_trait_container(dest.dest).select(dest.type);
        if (trait_base && trait_base->is_const()) {
            // ignore const infer types
            return *this;
        }

        mgb_throw_if(
                !trait_base || trait_base->infer_type() != InferType::RT_STATIC,
                GraphError, "StaticInferUpdater dest is not RT_STATIC type");
        m_dst.push_back(trait_base->as_mutable_safe());
        return *this;
    }

    void update() override {
        if (!m_build_done) {
            build();
            m_build_done = true;
        }
        for (auto i : m_src) {
            i->infer(true, false);
        }
        for (auto i : m_dst) {
            i->infer(false, false);
        }
    }
};

std::unique_ptr<StaticInferUpdater> StaticInferUpdater::make() {
    return std::make_unique<StaticInferManagerImpl::StaticInferUpdaterImpl>();
}

/* ===================== others ===================== */
ShapeInferDesc ShapeInferDesc::make_identity(VarNode* src) {
    auto infer_shape = [](TensorShape& dest, const InpVal& inp) {
        dest = inp.val.at(0).shape();
        return true;
    };
    return {SourceType::DEP, {{src, DepType::SHAPE}}, infer_shape};
}

ShapeInferDesc ShapeInferDesc::make_const(const TensorShape& shp) {
    auto infer_shape = [shp](TensorShape& dest, const InpVal&) {
        dest = shp;
        return true;
    };
    return {SourceType::CONSTANT, {}, infer_shape};
}

ValueInferDesc ValueInferDesc::make_identity(VarNode* src) {
    auto infer_value = [](DeviceTensorND& dest, const InpVal& inp) {
        dest = inp.val.at(0).value();
        return true;
    };
    return {SourceType::DEP, {{src, DepType::VALUE}}, infer_value};
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
