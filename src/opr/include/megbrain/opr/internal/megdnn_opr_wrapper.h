/**
 * \file src/opr/include/megbrain/opr/internal/megdnn_opr_wrapper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/opr/internal/mixin_base.h"

#include "megdnn/handle.h"

namespace mgb {
namespace opr {

namespace intl {
    //! get megdnn handle from comp node
    megdnn::Handle *get_megdnn_handle(CompNode comp_node);
    std::shared_ptr<megdnn::Handle> get_megdnn_handle_shared(CompNode comp_node);

    /*!
     * \brief get global megdnn operator asscoated with a computing node
     * \tparam Opr megdnn operator class, must be one of:
     *      * AddUpdate
     *      * Relayout
     *      * Checksum
     */
    template<typename Opr>
    Opr* get_megdnn_global_opr(CompNode comp_node);

    template<class Obj>
    class UniqPtrWithCN: public std::unique_ptr<Obj> {
        CompNode m_cn;

        public:
            UniqPtrWithCN() = default;

            template<class RObj>
            UniqPtrWithCN(UniqPtrWithCN<RObj> && o):
                std::unique_ptr<Obj>(std::move(o)),
                m_cn(o.comp_node())
            {
            }

            UniqPtrWithCN(std::unique_ptr<Obj> ptr, CompNode cn):
                std::unique_ptr<Obj>{std::move(ptr)}, m_cn{cn}
            {}

            CompNode comp_node() const {
                return m_cn;
            }
    };

    //! create megdnn opr from megdnn handle in a CompNode
    template<class Opr>
    UniqPtrWithCN<Opr> create_megdnn_opr(CompNode comp_node) {
        return {get_megdnn_handle(comp_node)->create_operator<Opr>(),
            comp_node};
    }

    /*!
     * \brief get temporary storage for oprs
     *
     * temp storage differs from workspace because the temp storage might
     * depends on runtime layout / pointer address
     */
    DeviceTensorStorage& get_temp_storage(ComputingGraph& graph,
                                          CompNode comp_node);

    /*!
     * \brief like get_temp_storage() but returns a DeviceTensorND instead
     * Note that if \p graph is nullptr, a new tensor would be returned
     */
    DeviceTensorND get_temp_tensor(ComputingGraph* graph, CompNode comp_node,
                                   const TensorLayout& layout);

} // namespace intl

namespace mixin {
    //! utility functions for megdnn opr
    namespace megdnn_utils {

        //! add input layout constraint to require all inputs to be contiguous
        void add_input_layout_constraint_contig(OperatorNodeBase &opr);

        //! called in constructor to add output vars
        void add_output_vars(
                OperatorNodeBase &opr, size_t nr_output, bool add_workspace);
    }

    /*!
     * \brief mixin for infer workspace size based on input and output shapes
     *
     * workspace must be the last output var
     */
    class WorkspaceSizeInfer: public cg::OperatorNodeMixinBase {
        protected:
            virtual size_t get_workspace_size_bytes(
                    const TensorShapeArray &input_shapes,
                    const TensorShapeArray &output_shapes) const = 0;

            /*!
             * \brief register static infer desc for workspace size
             * \param need_limit whether WorkspaceLimitGetter is needed
             */
            void mixin_init_output_static_infer_desc_workspace(
                    OperatorNodeBase &opr, bool need_limit);

            ~WorkspaceSizeInfer() = default;
    };

    //! hold a megdnn self and call create_megdnn_opr() when necessary
    class MegDNNOprHolder: public cg::mixin::SingleCNOperatorNode {
        public:
            //! call create_opr() internally.
            void mixin_init_output_comp_node(OperatorNodeBase &self);

            //! recreate operator when stream changes
            void mixin_on_output_comp_node_stream_changed(
                    OperatorNodeBase &self);

            static void record_megdnn_opr(
                    std::unique_ptr<megdnn::OperatorBase> opr,
                    cg::GraphExecutable::ExecDependencyArray& deps);

        protected:
            ~MegDNNOprHolder() noexcept;

            //! create actual megdnnn operator
            virtual void create_megdnn_opr() = 0;

            megdnn::OperatorBase* megdnn_opr() const {
                return m_megdnn_opr.get();
            }

            void set_megdnn_opr(std::unique_ptr<megdnn::OperatorBase> opr);

            //! record the megdnn opr owned by this opr to ExecDependencyArray
            void record_megdnn_opr(
                    cg::GraphExecutable::ExecDependencyArray& deps);

        private:
            std::unique_ptr<megdnn::OperatorBase> m_megdnn_opr;
    };

    class MegDNNOprHolderBwdStaticInfer: public MegDNNOprHolder {
        static constexpr size_t BAD_OSHP_IDX = -1;
        size_t m_oshp_idx = BAD_OSHP_IDX;
        bool m_oshp_need_val = 0;

        protected:

            ~MegDNNOprHolderBwdStaticInfer();


            //! initialize output shape desc for output(0)
            void mixin_init_output_static_infer_desc_bwd(
                    OperatorNodeBase &self) const;

            /*!
             * \brief set how to infer output shape; must be called in
             *      constructor
             * \param oshp_idx index of input var to provide output shape
             * \param oshp_need_val whether device value is needed for the input
             *      var that provides output shape; if this is false, oshp_idx
             *      must be output().size() - 1.
             */
            void mixin_setup_megdnn_bwd_output_infer(
                    size_t oshp_idx, bool oshp_need_val);

            void mixin_init_output_dtype(OperatorNodeBase &self);

            void mixin_update_node_prop(const OperatorNodeBase &self,
                    NodeProp *prop) const;

    };

    /*!
     * \brief implements create_megdnn_opr() and param init for a particular
     * MegDNNOpr
     *
     * WARNING: remember to add contiguous input layout constraint when directly
     * using this mixin
     */
    template<class MegDNNOpr,
        bool add_workspace = true, class OprHolder = MegDNNOprHolder>
    class MegDNNOprHolderImpl: public OprHolder {
        public:
            using Param = typename MegDNNOpr::Param;

            const Param &param() const {
                return m_param;
            }

            /*!
             * called in opr constructor to initialize
             *
             * 1. add output vars as specified by MegDNNOpr::NR_OUTPUTS
             * 2. add workspace output var
             * 3. add hash for m_param
             */
            void init_megdnn_opr(OperatorNodeBase &opr, const Param &param) {
                megdnn_utils::add_output_vars(opr, MegDNNOpr::NR_OUTPUTS,
                        add_workspace);
                m_param = param;
                if (!std::is_empty<Param>::value)
                    opr.add_equivalence_component<PODHash<Param>>(&m_param);
            }

            MegDNNOpr* megdnn_opr() const {
                return static_cast<MegDNNOpr*>(MegDNNOprHolder::megdnn_opr());
            }

        protected:
            ~MegDNNOprHolderImpl() = default;

            //! default impl calls megdnn::handle::create_operator()
            void create_megdnn_opr() override {
                auto opr = intl::get_megdnn_handle(this->mixin_comp_node())->
                    template create_operator<MegDNNOpr>();
                opr->param() = m_param;
                MegDNNOprHolder::set_megdnn_opr(std::move(opr));
            }

            size_t mixin_get_workspace_size_bytes_by_megdnn(
                    const OperatorNodeBase &opr,
                    const TensorShapeArray &input_shapes,
                    const TensorShapeArray &output_shapes) const;

        private:
            Param m_param;
    };

} // namespace mixin


namespace intl {
    class MegDNNGraphDep final : public cg::GraphExecutable::ExecDependency {
        std::unique_ptr<megdnn::OperatorBase> m_opr;

    public:
        MegDNNGraphDep(std::unique_ptr<megdnn::OperatorBase> opr) noexcept;
        ~MegDNNGraphDep() noexcept;
    };

    /*!
     * \brief glue class with workspace infer
     */
    template<class Base, class MixinImpl = mixin::WorkspaceSizeInfer>
    class WorkspaceSizeInfer: public mixin::CheckBase<Base>::Base,
                              public MixinImpl {
        protected:
            using Base::Base;

            void init_output_static_infer_desc_workspace(bool need_limit) {
                this->mixin_init_output_static_infer_desc_workspace(
                        *this, need_limit);
            }
    };


    template<class MegDNNOpr>
    class MegDNNOprWrapperFwdBase {
        public:
            using Holder = mixin::MegDNNOprHolderImpl<MegDNNOpr>;
            using Base = cg::SingleCNOperatorNode<
                cg::OutshapePureByInshapeOpr<>, Holder>;
    };

    //! base opr class for normal megdnn forward oprs that utilize megdnn's
    //! shape infer mechanism
    template<class MegDNNOpr>
    MGB_DEFINE_CLS_WITH_SUPER(MegDNNOprWrapperFwd,
            WorkspaceSizeInfer<
            typename MegDNNOprWrapperFwdBase<MegDNNOpr>::Base>) // {

        protected:
            using Super::Super;

            void add_input_layout_constraint() override {
                mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
            }

            void init_output_static_infer_desc() override;
            size_t get_workspace_size_bytes(
                    const TensorShapeArray &input_shapes,
                    const TensorShapeArray &output_shapes) const override;
            void scn_do_execute() override;
            void get_output_var_shape(
                    const TensorShapeArray &inp_shape,
                    TensorShapeArray &out_shape) const override final;

            void record_execute_deps(
                    cg::GraphExecutable::ExecDependencyArray& deps) override {
                this->record_megdnn_opr(deps);
            }
    };

    template<class MegDNNOpr>
    class MegDNNOprWrapperBwdBase {
        public:
            using BwdInfer = mixin::MegDNNOprHolderBwdStaticInfer;
            using Holder = mixin::MegDNNOprHolderImpl<
                MegDNNOpr, true, BwdInfer>;
            using Base = cg::SingleCNOperatorNode<
                cg::OperatorNodeBase, Holder>;
    };

    /*!
     * \brief helper for implementing backward operators whose output shape is
     *      one of its input shapes
     */
    template<class MegDNNOpr>
    MGB_DEFINE_CLS_WITH_SUPER(MegDNNOprWrapperBwd,
            WorkspaceSizeInfer<
            typename MegDNNOprWrapperBwdBase<MegDNNOpr>::Base>) // {
        protected:
            MegDNNOprWrapperBwd(
                    const OperatorNodeBaseCtorParam &base_param,
                    size_t oshp_idx, bool oshp_need_val):
                Super(base_param)
            {
                this->mixin_setup_megdnn_bwd_output_infer(
                        oshp_idx, oshp_need_val);
            }

            size_t get_workspace_size_bytes(
                    const TensorShapeArray &input_shapes,
                    const TensorShapeArray &output_shapes) const override;

            void add_input_layout_constraint() override {
                mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
            }

            void init_output_static_infer_desc() override;

            void scn_do_execute() override;

            void init_output_dtype() override {
                this->mixin_init_output_dtype(*this);
            }

            void record_execute_deps(
                    cg::GraphExecutable::ExecDependencyArray& deps) override {
                this->record_megdnn_opr(deps);
            }

            typename Super::NodeProp* do_make_node_prop() const override;
    };

} // namespace intl
} // namespace opr
} // namespace mgb

//! define a megdnn opr wrapper class with 1 input for forward
#define MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(_name) \
MGB_DEFINE_OPR_CLASS(_name, intl::MegDNNOprWrapperFwd<megdnn::_name>) \
    public: \
        _name(VarNode *p0, const Param &param, \
                const OperatorNodeConfig &config); \
        static SymbolVar make(SymbolVar p0, const Param &param = {}, \
                const OperatorNodeConfig &config = {}); \
}

//! define a megdnn opr wrapper class with 2 inputs for forward
#define MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD2(_name) \
MGB_DEFINE_OPR_CLASS(_name, intl::MegDNNOprWrapperFwd<megdnn::_name>) \
    public: \
        _name(VarNode *p0, VarNode *p1, const Param &param, \
                const OperatorNodeConfig &config); \
        static SymbolVar make(SymbolVar p0, SymbolVar p1, \
                const Param &param = {}, \
                const OperatorNodeConfig &config = {}); \
}

//! define a megdnn opr wrapper class with 3 inputs for grad
#define MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(_name, _extra...) \
MGB_DEFINE_OPR_CLASS(_name, intl::MegDNNOprWrapperBwd<megdnn::_name>) \
    _extra \
    public: \
        _name(VarNode *p0, VarNode *p1, VarNode *p2, const Param &param, \
                const OperatorNodeConfig &config); \
        static SymbolVar make(SymbolVar p0, SymbolVar p1, SymbolVar p2, \
                const Param &param = {}, \
                const OperatorNodeConfig &config = {}); \
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
