/**
 * \file src/opr/include/megbrain/opr/internal/out_shape_by_sym_var.h
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

namespace mgb {
namespace opr {

namespace mixin {

/*!
 * \brief mixin for operators whose output shape depends on the value of a
 *      symbol var input
 *
 * assuming single output.
 */
class OutshapeBySymvarOpr: public cg::OperatorNodeMixinBase {
    using NodeProp = cg::OperatorNodeBase::NodeProp;

    protected:
        ~OutshapeBySymvarOpr();

        /*!
         * \brief static shape infer is added here
         */
        void mixin_init_output_static_infer_desc(OperatorNodeBase &opr);

        /*!
         * \brief enable inferring output shape from other symbol var, and
         *      initialize; should be called in constructor after all inputs and
         *      outputs are added
         *
         * \param nr_shape_inp number of inputs whose shapes are required to
         *      infer output shape; must be placed at the beginning of inputs
         * \param hostval_inp_start starting index of inputs whose values are
         *      needed to infer output shape, and their device value must not
         *      be needed (i.e. they should not be involved in actual
         *      computing); they must be placed at the end of inputs
         */
        void mixin_outshape_by_symvar_enable(
                OperatorNodeBase &opr,
                size_t nr_shape_inp, size_t hostval_inp_start);

        /*!
         * \brief struct containing information needed for inferring output
         *      shape
         */
        struct ShapeInferInfo {
            //! shapes for the inputs [:nr_shape_inp]
            TensorShapeArray shape_inp_shp;

            //! values for the inputs [hostval_inp_start:]
            std::vector<const DeviceTensorND*> shpval_inp_val;
        };

        /*!
         * \brief implemented by subclasses to compute output shape
         */
        virtual void outshape_by_symvar_do_get_output_shape(
                TensorShape &dest, const ShapeInferInfo &shpinfo) = 0;

        /*!
         * \brief called by subclass to get ShapeInferInfo eagerly; usually used
         *      in graph execution for NO_SYS_MEM_ALLOC vars
         */
        const ShapeInferInfo& mixin_outshape_by_symvar_get_shape_infer_info(
                const OperatorNodeBase &opr) const;

        /*!
         * \brief update node prop to set dependency type
         */
        void mixin_outshape_by_symvar_reset_node_dep_type(
                const OperatorNodeBase &opr, NodeProp *prop) const;

    private:
        bool m_enable_out_shape_by_symbol_var = false;
        size_t m_nr_shape_inp = -1, m_hostval_inp_start = -1;
        mutable ShapeInferInfo m_shape_infer_info;
};

/*!
 * \brief OutshapeBySymvarOpr on single comp node
 */
template<class SCNBase = cg::mixin::SingleCNOperatorNode>
class OutshapeBySymvarSCNOpr: public OutshapeBySymvarOpr, public SCNBase {
    protected:
        using NodeProp = cg::OperatorNodeBase::NodeProp;

        ~OutshapeBySymvarSCNOpr() = default;

        NodeProp* mixin_do_make_node_prop(const OperatorNodeBase &opr) const {
            auto prop = SCNBase::mixin_do_make_node_prop(opr);
            this->mixin_outshape_by_symvar_reset_node_dep_type(opr, prop);
            return prop;
        }
};

}

namespace intl {

//! glue class for mixin::OutshapeBySymvarOpr
template<class Base = cg::OperatorNodeBase,
         class MixinImpl = mixin::OutshapeBySymvarOpr>
class OutshapeBySymvarOpr: public mixin::CheckBase<Base>::Base,
                           public MixinImpl {
    protected:
        using Base::Base;
        using ShapeInferInfo = mixin::OutshapeBySymvarOpr::ShapeInferInfo;
        using NodeProp = typename Base::NodeProp;
        using ExecEnv = typename Base::ExecEnv;

        void init_output_static_infer_desc() override {
            this->mixin_init_output_static_infer_desc(*this);
        }

        //! see mixin_outshape_by_symvar_enable for docs
        void outshape_by_symvar_enable(
                size_t nr_shape_inp, size_t hostval_inp_start) {
            this->mixin_outshape_by_symvar_enable(
                    *this, nr_shape_inp, hostval_inp_start);
        }

        const ShapeInferInfo& outshape_by_symvar_get_shape_infer_info() const {
            return this->mixin_outshape_by_symvar_get_shape_infer_info(*this);
        }

        void outshape_by_symvar_reset_node_dep_type(NodeProp *prop) const {
            this->mixin_outshape_by_symvar_reset_node_dep_type(*this, prop);
        }
};

using OutshapeBySymvarOprBase = OutshapeBySymvarOpr<>;
template<class SCNBase = cg::mixin::SingleCNOperatorNode>
using OutshapeBySymvarSCNOpr = OutshapeBySymvarOpr<
    cg::SingleCNOperatorNodeBaseT<mixin::OutshapeBySymvarSCNOpr<SCNBase>>,
    cg::mixin::EmptyMixinImpl
>;
using OutshapeBySymvarSCNOprBase = OutshapeBySymvarSCNOpr<>;

} // intl
} // opr
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

