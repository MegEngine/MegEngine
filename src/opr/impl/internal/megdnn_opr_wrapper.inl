/**
 * \file src/opr/impl/internal/megdnn_opr_wrapper.inl
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

namespace mgb {
namespace opr {

namespace intl {
    /*!
     * \brief template that can be specialized so inputs of an operator could be
     *      modified in-place
     *
     * Invoked by MEGDNN_OPR_INIT* macros
     *
     * \tparam Opr an megbrain opr final class
     */
    template<class Opr>
    struct MegDNNOprInitInputsModifier {
        static inline void apply(const typename Opr::Param &param,
                std::initializer_list<SymbolVar*> inputs) {
            MGB_MARK_USED_VAR(param);
            MGB_MARK_USED_VAR(inputs);
        }
    };

    /*!
     * \brief template that can be specialized to be called in opr constructor
     *
     * Invoked by MEGDNN_OPR_INIT* macros
     */
    template<class Opr>
    struct MegDNNOprInitPostCtor {
        static inline void apply(cg::OperatorNodeBase &opr) {
            MGB_MARK_USED_VAR(opr);
        }
    };

    //! get megdnn Workspace object from a workspace var
    megdnn::Workspace get_megdnn_workspace_from_var(VarNode *var);

    /*!
     * \brief A UserData object associated with the computing graph to get
     *      maximal usable workspace.
     *
     * It works by first limit workspace to 0 and alloc to get free memory, and
     * assume workspace can use all free memory.
     * It would produce a var node, which should be taken as a value dep for
     * workspace static infer functors so memory manager can re-allocate.
     */
    class WorkspaceLimitGetter {
        class Impl;
        static Impl* get_impl(ComputingGraph *graph);
        public:
            /*!
             * \brief get usable workspace size in bytes for a comp node
             *
             * Can only be called after is_prealloc_run() returns false
             *
             * \param old_limit workspace limit set by user, which would be an
             *      upper bound for the return value
             */
            static size_t get_workspace_limit(
                    ComputingGraph *graph, CompNode cn, size_t old_limit);

            //! return whether current is pre-allocation so workspace should
            //! return 0
            static bool is_prealloc_run(ComputingGraph *graph);

            /*!
             * \brief register WorkspaceLimitGetter in a graph
             * \return an var to be added as extra value dep for workspace
             *      infer; it would be null if WorkspaceLimitGetter is disabled
             *      at compile time
             */
            static VarNode* register_to_graph(ComputingGraph *graph);
    };

    /*!
     * a template that can be specialized to indicate whether
     * WorkspaceLimitGetter is needed for an operator class
     *
     * \tparam MegDNNOpr a megdnn opr class
     */
    template<class MegDNNOpr>
    struct AutoAddWorkspaceNeedLimitGetter {
        static constexpr bool val = false;
    };

    /*!
     * \brief implement megdnn::DynOutMallocPolicy using memory management
     *      system in megbrain
     */
    class MegDNNDynOutMallocImpl final: public megdnn::DynOutMallocPolicy {
        cg::OperatorNodeBase *m_opr;
        CompNode m_cn;

        public:
            MegDNNDynOutMallocImpl(cg::OperatorNodeBase *opr, CompNode cn):
                m_opr{opr}, m_cn{cn}
            {}

            megdnn::TensorND alloc_output(
                    size_t id, DType dtype, const TensorShape &shape,
                    void *user_data) override;

            void* alloc_workspace(size_t sz, void *user_data) override;
            void free_workspace(void *ptr, void *user_data) override;
    };

    /* ======================= MegDNNOprMethInvoker ======================= */
namespace {

    template<int nr_in, int nr_out>
    struct _MegDNNOprMethInvoker;

    template<class Opr>
    using MegDNNOprMethInvoker =
        _MegDNNOprMethInvoker<Opr::NR_INPUTS, Opr::NR_OUTPUTS>;

#define _NR_INPUTS 1
#define _NR_OUTPUTS 1
#define _FOREACH_IO(_i, _o) _i(0), _o(0)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 1
#define _NR_OUTPUTS 2
#define _FOREACH_IO(_i, _o) _i(0), _o(0), _o(1)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 1
#define _NR_OUTPUTS 3
#define _FOREACH_IO(_i, _o) _i(0), _o(0), _o(1), _o(2)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 2
#define _NR_OUTPUTS 1
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _o(0)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 2
#define _NR_OUTPUTS 2
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _o(0), _o(1)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 3
#define _NR_OUTPUTS 1
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _i(2), _o(0)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 3
#define _NR_OUTPUTS 2
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _i(2), _o(0), _o(1)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 3
#define _NR_OUTPUTS 3
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _i(2), _o(0), _o(1), _o(2)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 4
#define _NR_OUTPUTS 1
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _i(2), _i(3), _o(0)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 5
#define _NR_OUTPUTS 2
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _i(2), _i(3), _i(4), _o(0), _o(1)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

#define _NR_INPUTS 5
#define _NR_OUTPUTS 3
#define _FOREACH_IO(_i, _o) _i(0), _i(1), _i(2), _i(3), _i(4), _o(0), _o(1), _o(2)
#include "./megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl"

} // anonymous namespace

    /* ======================= MegDNNOprWrapperFwd ======================= */

    template<class MegDNNOpr>
    void MegDNNOprWrapperFwd<MegDNNOpr>::init_output_static_infer_desc() {
        Super::set_nr_managed_outputs(this->output().size() - 1);
        Super::init_output_static_infer_desc();
        this->init_output_static_infer_desc_workspace(
                AutoAddWorkspaceNeedLimitGetter<MegDNNOpr>::val);
    }

    template<class MegDNNOpr>
    void MegDNNOprWrapperFwd<MegDNNOpr>::scn_do_execute() {
        MegDNNOprMethInvoker<MegDNNOpr>::exec(this->megdnn_opr(), this);
    }

    template<class MegDNNOpr>
    size_t MegDNNOprWrapperFwd<MegDNNOpr>::get_workspace_size_bytes(
                const TensorShapeArray &input_shapes,
                const TensorShapeArray &output_shapes) const {
        return this->mixin_get_workspace_size_bytes_by_megdnn(
                *this, input_shapes, output_shapes);
    }

    template<class MegDNNOpr>
    void MegDNNOprWrapperFwd<MegDNNOpr>::get_output_var_shape(
                const TensorShapeArray &inp_shape,
                TensorShapeArray &out_shape) const {
        MegDNNOprMethInvoker<MegDNNOpr>::deduce_layout(
                this->megdnn_opr(), this, inp_shape, out_shape);
    }

    /* ======================= MegDNNOprWrapperBwd ======================= */

    template<class MegDNNOpr>
    void MegDNNOprWrapperBwd<MegDNNOpr>::init_output_static_infer_desc() {
        this->mixin_init_output_static_infer_desc_bwd(*this);
        this->init_output_static_infer_desc_workspace(
                AutoAddWorkspaceNeedLimitGetter<MegDNNOpr>::val);
    }

    template<class MegDNNOpr>
    void MegDNNOprWrapperBwd<MegDNNOpr>::scn_do_execute() {
        MegDNNOprMethInvoker<MegDNNOpr>::exec(this->megdnn_opr(), this);
    }

    template<class MegDNNOpr>
    size_t MegDNNOprWrapperBwd<MegDNNOpr>::get_workspace_size_bytes(
                const TensorShapeArray &input_shapes,
                const TensorShapeArray &output_shapes) const {
        return this->mixin_get_workspace_size_bytes_by_megdnn(
                *this, input_shapes, output_shapes);
    }

    template<class MegDNNOpr>
    typename MegDNNOprWrapperBwd<MegDNNOpr>::Super::NodeProp*
    MegDNNOprWrapperBwd<MegDNNOpr>::do_make_node_prop() const {
        auto prop = Super::do_make_node_prop();
        this->mixin_update_node_prop(*this, prop);
        return prop;
    }

} // nmamespace intl


namespace mixin {
    /* ======================= MegDNNOprHolderImpl ======================= */

    template<class MegDNNOpr, bool add_workspace, class OprHolder>
    size_t MegDNNOprHolderImpl<MegDNNOpr, add_workspace, OprHolder>::
    mixin_get_workspace_size_bytes_by_megdnn(
            const OperatorNodeBase &opr,
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const {
        static_assert(add_workspace, "must add_workspace");
        return intl::MegDNNOprMethInvoker<MegDNNOpr>::get_workspace_in_bytes(
                this->megdnn_opr(), &opr, input_shapes, output_shapes);
    }
}


} // namespace opr
} // namespace mgb


//! generate opr constructor, with 1 arg
#define MEGDNN_OPR_CTOR_INIT1(_name, _node_name, ...) \
_name::_name(VarNode *i0, \
        const Param &param, const OperatorNodeConfig &config): \
    Super(OperatorNodeBaseCtorParam{ \
            i0->owner_graph(), config, _node_name, {i0}} ,##__VA_ARGS__) \
{ \
    init_megdnn_opr(*this, param); \
    add_input({i0}); \
    intl::MegDNNOprInitPostCtor<_name>::apply(*this); \
}
//! generate opr constructor and ::make, with 1 arg
#define MEGDNN_OPR_INIT1(_name, _node_name, ...) \
MEGDNN_OPR_CTOR_INIT1(_name, _node_name ,##__VA_ARGS__) \
SymbolVar _name::make(SymbolVar i0, \
        const Param &param, const OperatorNodeConfig &config) { \
    intl::MegDNNOprInitInputsModifier<_name>::apply(param, {&i0}); \
    return i0.insert_single_output_opr<_name>( \
            i0.node(), param, config); \
}

//! generate opr constructor, with 2 args
#define MEGDNN_OPR_CTOR_INIT2(_name, _node_name, ...) \
_name::_name(VarNode *i0, VarNode *i1, \
        const Param &param, const OperatorNodeConfig &config): \
    Super(OperatorNodeBaseCtorParam{ \
            i0->owner_graph(), config, _node_name, {i0}} ,##__VA_ARGS__) \
{ \
    init_megdnn_opr(*this, param); \
    add_input({i0, i1}); \
    intl::MegDNNOprInitPostCtor<_name>::apply(*this); \
}
//! generate opr constructor and ::make, with 2 args
#define MEGDNN_OPR_INIT2(_name, _node_name, ...) \
MEGDNN_OPR_CTOR_INIT2(_name, _node_name ,##__VA_ARGS__) \
SymbolVar _name::make(SymbolVar i0, SymbolVar i1, \
        const Param &param, const OperatorNodeConfig &config) { \
    intl::MegDNNOprInitInputsModifier<_name>::apply(param, {&i0, &i1}); \
    return i0.insert_single_output_opr<_name>( \
            i0.node(), i1.node(), param, config); \
}

//! generate opr constructor, with 3 args
#define MEGDNN_OPR_CTOR_INIT3(_name, _node_name, ...) \
_name::_name(VarNode *i0, VarNode *i1, VarNode *i2, \
        const Param &param, const OperatorNodeConfig &config): \
    Super(OperatorNodeBaseCtorParam{ \
            i0->owner_graph(), config, _node_name, {i0}} ,##__VA_ARGS__) \
{ \
    init_megdnn_opr(*this, param); \
    add_input({i0, i1, i2}); \
    intl::MegDNNOprInitPostCtor<_name>::apply(*this); \
}
//! generate opr constructor and ::make, with 3 args
#define MEGDNN_OPR_INIT3(_name, _node_name, ...) \
MEGDNN_OPR_CTOR_INIT3(_name, _node_name ,##__VA_ARGS__) \
SymbolVar _name::make(SymbolVar i0, SymbolVar i1, SymbolVar i2, \
        const Param &param, const OperatorNodeConfig &config) { \
    intl::MegDNNOprInitInputsModifier<_name>::apply(param, {&i0, &i1, &i2}); \
    return i0.insert_single_output_opr<_name>( \
            i0.node(), i1.node(), i2.node(), param, config); \
}

//! generate opr constructor, with 4 args
#define MEGDNN_OPR_CTOR_INIT4(_name, _node_name, ...) \
_name::_name(VarNode *i0, VarNode *i1, VarNode *i2, VarNode *i3, \
        const Param &param, const OperatorNodeConfig &config): \
    Super(OperatorNodeBaseCtorParam{ \
            i0->owner_graph(), config, _node_name, {i0}} ,##__VA_ARGS__) \
{ \
    init_megdnn_opr(*this, param); \
    add_input({i0, i1, i2, i3}); \
    intl::MegDNNOprInitPostCtor<_name>::apply(*this); \
}
//! generate opr constructor and ::make, with 4 args
#define MEGDNN_OPR_INIT4(_name, _node_name, ...) \
MEGDNN_OPR_CTOR_INIT4(_name, _node_name ,##__VA_ARGS__) \
SymbolVar _name::make(SymbolVar i0, SymbolVar i1, SymbolVar i2, SymbolVar i3, \
        const Param &param, const OperatorNodeConfig &config) { \
    intl::MegDNNOprInitInputsModifier<_name>::apply( \
            param, {&i0, &i1, &i2, &i3}); \
    return i0.insert_single_output_opr<_name>( \
            i0.node(), i1.node(), i2.node(), i3.node(), param, config); \
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
