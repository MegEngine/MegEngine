/**
 * \file src/opr/include/megbrain/opr/basic_arith.h
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
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/param_defs.h"

#include "megdnn/oprs/general.h"

#include <bitset>

namespace mgb {
namespace opr {

namespace intl {
    using ElemwiseBase = cg::SingleCNOperatorNode<
            cg::OutshapePureByInshapeOpr<>,
            mixin::MegDNNOprHolderImpl<megdnn::Elemwise, false>>;

    //! helper for dtype promotion of a list of vars
    class BatchedDTypePromotion final : NonCopyableObj {
        const VarNodeArrayView& m_orig_vars;
        bool m_changed = false, m_finalized = false;
        VarNodeArray m_cvt_vars;
        Maybe<VarNodeArrayView> m_cvt_vars_view;
        DType m_final_dtype;

    public:
        explicit BatchedDTypePromotion(const VarNodeArrayView& vars);

        //! get currently promoted dtype
        DType get_dtype() const { return m_final_dtype; }

        //! change the target dtype
        void set_dtype(DType dtype);

        //! get converted vars
        const VarNodeArrayView& get_vars();
    };
}

/*!
 * \brief element-wise arithmetic operators
 *
 * Actual arithmetic operation is determined by mode.
 *
 * The operands are broadcasted automatically on dimensions of shape one to
 * match shapes of each other; it works like broadcasting in numpy.
 */
MGB_DEFINE_OPR_CLASS(Elemwise, intl::ElemwiseBase) // {
    using ModeTrait = megdnn::Elemwise::ModeTrait;

    public:
        using Mode = Param::Mode;

        Elemwise(const ModeTrait &mode_trait,
                const VarNodeArrayView &inputs, Param param,
                const OperatorNodeConfig &config);

        static SymbolVar make(
                const VarNodeArrayView &inputs,
                Param param, const OperatorNodeConfig &config = {});

        static TensorShape get_output_var_shape(Mode mode,
                const TensorShapeArray &input_shapes);

        /*!
         * \brief compute the result directly on device tensors
         *
         * All inputs must have the same dtype.
         *
         * \param opr the megdnn operator to be used; a new operator would be
         *      created if it is null
         */
        static void perform(Mode mode,
                DeviceTensorND &dest,
                const SmallVector<DeviceTensorND> &inputs,
                intl::UniqPtrWithCN<megdnn::Elemwise>& opr);


        using TensorLayoutPtrArray = SmallVector<TensorLayout*>;

        /*!
         * \brief collectively collapse consecutive axes with contiguous and
         *      same shape in all layous together before collective collapse all
         *      layouts should be broadcast to the same dim.
         *
         * \param layouts the layouts to be collectively collapsed
         *
         */
        static TensorLayoutArray collective_collapse(
                const TensorLayoutArray& layouts);

        //! like collective_collapse(), but modify the layouts inplace
        static void collective_collapse_inplace(
                const TensorLayoutPtrArray& layouts);

        /*!
         * \brief wapper for broadcast and collective collapse
         *
         * \param[in,out] inp_layouts input layouts that would be
         *      broadcasted into \p target_layout and then collapsed together
         * \param[in,out] target_layout broadcasted target layout; it would be
         *      collapsed together with inputs
         */
        static void broadcast_collective_collapse(
                const TensorLayoutPtrArray& inp_layouts,
                TensorLayout *target_layout);

        /*!
         * \brief whether an input var could be broadcasted to match other
         *      inputs
         *
         * Used in grad
         */
        auto&& input_broadcastable() const {
            return m_input_broadcastable;
        }

        /*!
         * \brief sum a list of gradient vars with possible optimizations
         * \param wrt the var to take grad with
         * \param[in,out] grads vars to be summed; it is also an output param,
         *      which would contain all the intermediate results for summing
         */
        static VarNode* sum_grad_list(VarNode *wrt, VarNodeArray &grads);

        //! whether input layouts mismatch ever happened for fused oprs; this
        //! method is public for debug purpose
        bool fuse_badlayout_warn_printed() const {
            return m_fuse_badlayout_warn_printed;
        }

    private:
        bool m_fuse_badlayout_warn_printed = false;
        std::bitset<8> m_input_broadcastable;

        void mem_plan_fwd_in2out_writable() override;
        void scn_do_execute() override;

        void get_output_var_shape(
                const TensorShapeArray &inp_shape,
                TensorShapeArray &out_shape) const override;

        void init_output_static_infer_desc() override;

        static void call_megdnn_opr_exec(
                CompNode comp_node,
                megdnn::TensorNDArray &inp, const megdnn::TensorND &out,
                megdnn::Elemwise *opr,
                Elemwise *caller);

        void record_execute_deps(ExecDependencyArray& deps) override;
        void add_input_layout_constraint() override;
};

namespace intl {
    using TypeCvtBase = cg::OutshapePureByInshapeOpr<
        cg::SingleCNOperatorNodeBaseT<
            mixin::MegDNNOprHolderImpl<megdnn::TypeCvt, false>>,
        cg::mixin::IOSameShapeOperatorNode
    >;
}

MGB_DEFINE_OPR_CLASS(TypeCvt, intl::TypeCvtBase) // {
    public:
        TypeCvt(VarNode *inp, DType dest_type,
                const OperatorNodeConfig &config);

        static SymbolVar make(
                SymbolVar input, DType dest_type,
                const OperatorNodeConfig &config = {});

        static void perform(DeviceTensorND &dest,
                DType dest_type, const DeviceTensorND &src,
                intl::UniqPtrWithCN<megdnn::TypeCvt> &opr);

        using Param = DType;
        Param param() const {
            return output(0)->dtype();
        }

    private:
        void mem_plan_fwd_in2out_writable() override;
        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        NodeProp* do_make_node_prop() const override;
        void record_execute_deps(ExecDependencyArray& deps) override;
        void add_input_layout_constraint() override;
};


/*!
 * \brief update a SharedDeviceTensor by adding to it
 *
 * dest := dest * alpha + delta * beta + bias
 *
 * dest must be produced by SharedDeviceTensor
 *
 * Note that if alpha == 0, beta == 1 and bias == 0, then dest would be
 * overwritten directly (so it could contain any value before updating, even
 * INF or NAN)
 *
 * Attention: AddUpdate will not be executed if disable flag is set to 1,
 * this is used for dynamic param-updating.
 */
MGB_DEFINE_OPR_CLASS(AddUpdate,
        cg::SingleCNOperatorNodeBaseT<mixin::MegDNNOprHolder>) // {
    public:
        using SharedScalar = std::shared_ptr<DTypeScalar>;

        class SharedScalarOrImm {
            SharedScalar m_ss;

            public:

                SharedScalarOrImm(const SharedScalar &v):
                    m_ss{v}
                {}

                template<typename ctype,
                    typename = typename ctype_enable_if<ctype>::type>
                SharedScalarOrImm(ctype v):
                    m_ss{std::make_shared<DTypeScalar>(v)}
                {}

                auto &&get() const {
                    return m_ss;
                }
        };

        struct Param {
            SharedScalar alpha, beta, bias, disable;
            Param(const SharedScalarOrImm& alpha_ = 1.f,
                    const SharedScalarOrImm& beta_ = 1.f,
                    const SharedScalarOrImm& bias_ = 0.f,
                    const SharedScalarOrImm& disable_ = 0):
                alpha{alpha_.get()}, beta{beta_.get()},
                bias{bias_.get()}, disable{disable_.get()}
            {}
        };

        const Param& param() const {
            return m_param;
        }

        AddUpdate(VarNode *dest, VarNode *delta,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar dest, SymbolVar delta,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
    private:
        const Param m_param;

        NodeProp* do_make_node_prop() const override;
        void create_megdnn_opr() override;

        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        void record_execute_deps(ExecDependencyArray& deps) override;
};

/*!
 * \brief reduce to given shape or along a specific axis
 *
 * Mode specifies the actual arithmetic; and exactly one of *axis* and
 * *target_shape* must be provided, to specify output shape.
 */
MGB_DEFINE_OPR_CLASS(Reduce, intl::DynamicOutputIfInputDynamic<
        intl::OutshapeBySymvarSCNOpr<mixin::MegDNNOprHolder>>) //  {

    public:
        using Param = megdnn::param::Reduce;
        using Mode = Param::Mode;

        Reduce(VarNode *inp, VarNode *target_shape, const Param &param,
                const OperatorNodeConfig &config);
        ~Reduce();

        const Param& param() const {
            return m_param;
        }

        static SymbolVar make(
                SymbolVar src, Param param,
                SymbolVar target_shape = {},
                const OperatorNodeConfig &config = {});

        static void perform(Mode mode, DeviceTensorND& dest,
                            DeviceTensorND& workspace,
                            const DeviceTensorND& input,
                            const TensorShape& target_shape,
                            intl::UniqPtrWithCN<megdnn::Reduce>& opr,
                            const Param::DataType data_type=Param::DataType::DEFAULT);

    private:
        class KernScheduler;
        class OutTensorShapeExtender;

        const Param m_param;
        const std::unique_ptr<KernScheduler> m_kern_scheduler;

        //! if m_kern_param empty, just forward to output
        bool m_mem_fwd_success = false;

        //! whether target shape is symbolic (rather than axis)
        bool m_is_symtshp = false;

        inline void init_kern_sched_shape(
                const TensorShape &ishp, const TensorShape &oshp);

        OprEventCallback get_opr_event_callback() override final;

        void outshape_by_symvar_do_get_output_shape(
                TensorShape &dest, const ShapeInferInfo &shpinfo)
            override final;

        void mem_plan_fwd_in2out_readonly() override final;
        void add_input_layout_constraint() override final;
        void scn_do_execute() override final;
        void init_output_static_infer_desc() override final;

        void create_megdnn_opr() override;
        void record_execute_deps(ExecDependencyArray& deps) override;
};

/*!
 * \brief pow with constant exponent
 *
 * Note: this is considered as a fused opr in the optimization passes.
 * Elemwise::Mode::POW  is the canonical form. The user should construct the
 * graph with only Elemwise::Mode::POW, and this opr should only be inserted by
 * the optimizer.
 */
MGB_DEFINE_OPR_CLASS(PowC, intl::MegDNNOprWrapperFwd<megdnn::PowC>) // {
    void add_input_layout_constraint() override;
    void init_output_static_infer_desc() override;
    void mem_plan_fwd_in2out_writable() override;

public:
    PowC(VarNode* inp, const Param& param, const OperatorNodeConfig& config);
    static SymbolVar make(SymbolVar inp, const Param& param = {},
                          const OperatorNodeConfig& config = {});
};

} // namespace opr
} // namespace mgb



// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
