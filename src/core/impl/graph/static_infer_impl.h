/**
 * \file src/core/impl/graph/static_infer_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <deque>
#include "megbrain/graph/static_infer.h"
#include "megbrain/utils/mempool.h"

namespace mgb {
namespace cg {

struct CompSeqExtraInfo;
class OperatorNodeBase;
class ComputingGraph;

namespace static_infer {

class CompSeqManager;

class StaticInferManagerImpl final : public StaticInferManager {
public:
    class StaticInferUpdaterImpl;
    class SubgraphStaticInferHelperImpl;
    using TagHandlerType = DepType;

    StaticInferManagerImpl(ComputingGraph* graph) : m_owner_graph{graph} {}

    ~StaticInferManagerImpl() noexcept;

    /*!
     * \brief represents shape or value of a tag
     */
    class TagHandler;
    using TagHandlerSet = ThinHashSet<TagHandler*>;

    void register_shape_infer(Tag dest, const ShapeInferDesc& desc) override;
    void register_value_infer(Tag dest, const ValueInferDesc& desc) override;

    InferType get_infer_type(Tag dest) override;

    const TensorShape& infer_shape(Tag dest) override;
    const TensorShape* infer_shape_fallible(Tag dest) override;

    const DeviceTensorND& infer_value(Tag dest) override;
    const DeviceTensorND* infer_value_fallible(Tag dest) override;

    DepVal get_rt_static_source_deps(const DepElement& dest) override;

    /*!
     * \brief get a tag handler for shape inference
     */
    TagHandler* get_tag_handler_for_shape(Tag tag);

    /*!
     * \brief get a tag handler for value inference
     */
    TagHandler* get_tag_handler_for_value(Tag tag);

    /*!
     * \brief clear registered handler for a tag; this is only used in error
     *      handling in opr creation
     */
    void clear_tag_handler(Tag tag);

    /*!
     * \brief set the operator that is allowd to call register_*_infer
     *      methods; set to null to disable calling such methods
     * \return original register_allowed_opr
     */
    OperatorNodeBase* set_register_allowed_opr(OperatorNodeBase* opr) {
        auto ret = m_register_allowed_opr;
        m_register_allowed_opr = opr;
        return ret;
    }

    /*!
     * \brief get all source missing inputs needed to statically infer a
     *      tag
     * \return set of missing inputs; the pointer is always available
     */
    const TagHandlerSet& get_missing_inp(TagHandler* dest);

    /*!
     * \brief update mutable src tag's shape explictly which only used by
            eager eval
     */
    void update_mutable_src_shape(Tag tag);

    /*!
     * \brief get original deps given in the InferDesc which is registered
     * by register_shape_infer or register_value_infer
     *
     * Note: the \p elem with DepType::SHAPE and InferType::CONST shows no
     * deps since the StaticInferManagerImpl folds the infererence chain of
     * the const var shape
     */
    DepVal get_deps(const DepElement& elem);

private:
    friend class CompSeqManager;

    class TagTraitBase;
    class TagConstShapeTrait;
    class TagTraitMutableBase;
    class TagShapeTrait;
    class TagValueTrait;
    class TagTraitDepIter;
    struct TagTraitContainer;

    ComputingGraph* const m_owner_graph;
    MGB_RECURSIVE_MUTEX m_mtx;

    //! callbacks to be invoked in destructor
    ThinHashMap<void*, thin_function<void()>> m_dtor_callbacks;

    MemPool<TagConstShapeTrait> m_mem_pool_const_shape_trait;
    MemPool<TagShapeTrait> m_mem_pool_shape_trait;
    MemPool<TagValueTrait> m_mem_pool_value_trait;

    OperatorNodeBase* m_register_allowed_opr = nullptr;

    const TensorShape* do_infer_shape(Tag dest, bool allow_fail);
    const DeviceTensorND* do_infer_value(Tag dest, bool allow_fail);

    TagTraitBase* get_tag_trait_for_dep(const DepElement& dep);
    static TagTraitContainer& get_tag_trait_container(Tag tag);
};

class StaticInferManagerImpl::TagHandler {
    Tag const m_tag;

public:
    TagHandler(Tag tag) : m_tag(tag) {}

    virtual ~TagHandler() = default;

    //! type of this handler impl
    virtual TagHandlerType handler_type() const = 0;

    /*!
     * \brief get corresponding tag for this tag handler
     */
    Tag tag() const { return m_tag; }

    /*!
     * \brief sync shape/value from corresponding var, used for
     *      missing input sources
     */
    virtual void sync_from_var() = 0;

    /*!
     * \brief compute newest result and get current result version
     */
    virtual size_t update_infer_result_version() = 0;
};

/*!
 * \brief helper for static inference for a computing sequence
 */
class CompSeqManager {
    ComputingGraph* m_owner_graph;
    using TagTraitBase = StaticInferManagerImpl::TagTraitBase;
    using TagHandlerType = StaticInferManagerImpl::TagHandlerType;

    class VersionedTagTrait;

    std::vector<VersionedTagTrait>
            m_static_infer_const_needed,  //!< const infer type, checked in first run
            m_static_srcnode,             //!< to be checked in each run
            m_static_mid;                 //!< nodes to be updated if src changed

    ThinHashSet<TagTraitBase*> m_added;  //!< nodes already added by add_dest()

    std::deque<TagTraitBase*> m_add_dest_queue;

    bool m_static_first_run = false;

    void add_dest(CompSeqExtraInfo& info, TagTraitBase* dest);

public:
    CompSeqManager(ComputingGraph* graph);
    ~CompSeqManager() noexcept;

    /*!
     * \brief called by graph compiler to set needed tags
     *
     * input: info.infer_dest
     * outputs: info.missing_for_shape, info.missing_for_value,
     *          infer.rt_static_infer_src
     */
    void reset_dest(CompSeqExtraInfo& info);

    /*!
     * \brief re-compute tags in reset_dest() that are statically
     *      inferable and assign shape descs to to var->shape()
     * \return whether any shape changes
     */
    bool update_static_check_shape_change();
};

}  // namespace static_infer
}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
