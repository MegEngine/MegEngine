/**
 * \file src/opr/include/megbrain/opr/tensor_manip.h
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
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/internal/indexing_helper.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"

#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

/*!
 * \brief get the shape of a var and store in a 1-dim tensor
 *
 * For multiple inputs, shape would be the broadcasted shape.
 *
 * \param axis output shape of a single axis
 */
MGB_DEFINE_OPR_CLASS(GetVarShape, cg::SingleCNOperatorNodeBase) // {
    class ShapeDevValueExecDep;
    public:
        using Param = megdnn::param::OptionalAxisV1;

        GetVarShape(const VarNodeArrayView &inp, Param axis,
                const OperatorNodeConfig &config);

        static SymbolVar make(
                SymbolVar inp, Param axis = {},
                const OperatorNodeConfig &config = {}) {
            return make(SymbolVarArray({inp}), axis, config);
        }

        //! get broadcasted shape
        static SymbolVar make(
                const VarNodeArrayView &inp,
                Param axis = {}, const OperatorNodeConfig &config = {});

        Param param() const {
            return m_axis;
        }

    private:
        const Param m_axis;

        //! cached shape, to avoid h2d copy when shape not changed
        bool m_cached_shape_dev_v_synced = false;
        TensorShape m_cached_shape;
        TensorShapeArray m_src_shapes;
        DeviceTensorND m_cached_shape_cpu_v{CompNode::default_cpu()},
                       m_cached_shape_dev_v;

        //! update m_cached_shape from m_src_shapes
        void update_cached_shape();

        //! update m_cached_shape for static infer
        void update_for_static_infer(const cg::static_infer::InpVal &inp);

        NodeProp* do_make_node_prop() const override;
        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        void record_execute_deps(ExecDependencyArray& deps) override;
};

namespace intl {

/*!
 * \brief base class for reshape and broadcast
 */
MGB_DEFINE_CLS_WITH_SUPER(ReshapeBrdcastHelper,
        ReadonlyFwdHelper<OutshapeBySymvarSCNOprBase>) // {
    bool m_incompatible_inp_layout = false;

    void mem_plan_fwd_in2out_readonly() override final;
    void outshape_by_symvar_do_get_output_shape(
                TensorShape &dest,
                const ShapeInferInfo &shpinfo) override final;
    void scn_do_execute() override final;
    void add_input_layout_constraint() override final;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;

    protected:
        using Super::Super;

        void reshapebrdcast_init(VarNode *inp, VarNode *tshp);

        /*!
         * \brief get dest layout
         *
         * Invalid TensorLayout can be returned if reshape fails
         */
        virtual Maybe<TensorLayout> reshapebrdcast_get_dest_layout(
                const TensorLayout &src, const TensorShape &tshape) const = 0;

        /*!
         * \brief whether output shape depends on input shape; if true,
         *      reshapebrdcast_get_dest_layout() would be called to get output
         *      shape; otherwise output shape would be value of input(1)
         */
        virtual bool reshapebrdcast_output_shape_need_input_shape() const = 0;
};

} // namespace intl

/*!
 * \brief reshape a tensor in-place, without changing total span
 * \param unspec_axis the axis that shape is not specified in input, but should
 *      be calculated from total number of elements and other dims in dest shape
 */
MGB_DEFINE_OPR_CLASS(Reshape, intl::ReshapeBrdcastHelper) // {
    public:
        using Param = megdnn::param::OptionalAxisV1;

        Reshape(VarNode *inp, VarNode *tshp, Param unspec_axis,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar inp,
                SymbolVar tshp, Param unspec_axis = {},
                const OperatorNodeConfig &config = {});

        static SymbolVar make(SymbolVar inp,
                const TensorShape &target_shape, Param unspec_axis = {},
                const OperatorNodeConfig &config = {}) {
            return make(inp, cg::var_from_tensor_shape(inp, target_shape),
                    unspec_axis, config);
        }

        Param param() const {
            return m_unspec_axis;
        }
    private:
        Param m_unspec_axis;

        Maybe<TensorLayout> reshapebrdcast_get_dest_layout(
                const TensorLayout &src, const TensorShape &tshape)
            const override;

        bool reshapebrdcast_output_shape_need_input_shape() const override;
};

/*!
 * \brief broadcast tensor value along axes whose shape is 1
 */
MGB_DEFINE_OPR_CLASS(Broadcast, intl::ReshapeBrdcastHelper) // {
    Maybe<TensorLayout> reshapebrdcast_get_dest_layout(
            const TensorLayout &src, const TensorShape &tshape) const override;

    bool reshapebrdcast_output_shape_need_input_shape() const override;

    public:
        Broadcast(VarNode *inp, VarNode *tshp,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar inp, SymbolVar tshp,
                const OperatorNodeConfig &config = {});

        static SymbolVar make(SymbolVar inp,
                const TensorShape &target_shape,
                const OperatorNodeConfig &config = {}) {
            return make(inp, cg::var_from_tensor_shape(inp, target_shape),
                    config);
        }

        // used for serialization

        using Param = megdnn::param::Empty;

        Param param() const {
            return {};
        }

        static SymbolVar make(SymbolVar inp, SymbolVar tshp,
                const Param &, const OperatorNodeConfig &config) {
            return make(inp, tshp, config);
        }
};

namespace intl {

/*!
 * \brief base class for oprs that manipulate axis
 */
MGB_DEFINE_CLS_WITH_SUPER(AxisManipOprBase,
        ReadonlyFwdHelper<cg::SingleCNOperatorNodeBase>) // {
    void mem_plan_fwd_in2out_readonly() override final;
    void scn_do_execute() override final;
    void init_output_static_infer_desc() override final;
    NodeProp* do_make_node_prop() const override;

    protected:
        using Super::Super;
        virtual TensorLayout axis_manip_get_output_layout(
                const TensorLayout &inp_layout) const = 0;

        void axis_manip_init(VarNode* inp);
};

}

/*!
 * \brief dimshuffle a tensor in-place, without changing total span
 * \param pattern non-negative intergers refer to corresponding dimension;
 *      -1 refers to new dimension
 * \param ndim number of input dimensions; 0 to be inferred from pattern
 *
 * Note that dimensions with shape-1 could be dropped
 */
MGB_DEFINE_OPR_CLASS(Dimshuffle, intl::AxisManipOprBase) // {
    std::vector<int> m_pattern;
    size_t m_inp_ndim;

    TensorLayout axis_manip_get_output_layout(
            const TensorLayout &inp_layout) const override;

    public:
        Dimshuffle(VarNode *inp, const std::vector<int> &pattern, size_t ndim,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar inp,
                const std::vector<int> &pattern,
                size_t ndim = 0,
                const OperatorNodeConfig &config = {});

        VarNode* grad(size_t wrt_idx, const VarNodeArray &out_grad) const;

        // used for serialization
        struct Param {
            static constexpr uint32_t TAG = param_tag::DIMSHUFFLE;
            uint32_t pattern_len;
            int32_t pattern[TensorShape::MAX_NDIM];
            uint32_t ndim;
        };
        static SymbolVar make(SymbolVar inp, const Param &param,
                const OperatorNodeConfig &config) {
            return make(inp, {param.pattern, param.pattern + param.pattern_len},
                    param.ndim, config);
        }
        Param param() const {
            Param ret;
            ret.pattern_len = m_pattern.size();
            std::copy(m_pattern.begin(), m_pattern.end(), ret.pattern);
            ret.ndim = m_inp_ndim;
            return ret;
        }
};

/*!
 * \brief add or remove an axis with shape 1
 *
 * All the axis descs would be processed in order
 */
MGB_DEFINE_OPR_CLASS(AxisAddRemove, intl::AxisManipOprBase) // {
    public:
        struct AxisDesc {
            enum class Method {
                //! add a dim with shape 1, just before axis
                ADD_1,
                //! remove this axis, which must be shape 1
                REMOVE
            };
            Method method;
            indexing::AxisNum axis;

            static AxisDesc make_add(indexing::AxisNum axis) {
                AxisDesc r;
                r.axis = axis;
                r.method = Method::ADD_1;
                return r;
            }

            static AxisDesc make_remove(indexing::AxisNum axis) {
                AxisDesc r;
                r.axis = axis;
                r.method = Method::REMOVE;
                return r;
            }
        };

        AxisAddRemove(VarNode *inp, const std::vector<AxisDesc> &desc,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar inp,
                const std::vector<AxisDesc> &desc,
                const OperatorNodeConfig &config = {});

        // used for serialization
        struct Param {
            static constexpr uint32_t TAG = param_tag::AXIS_ADD_REMOVE,
                             MAX_DESC_SIZE = TensorShape::MAX_NDIM * 2;
            uint32_t nr_desc;
            AxisDesc desc[MAX_DESC_SIZE];
        };
        static SymbolVar make(SymbolVar inp, const Param &param,
                const OperatorNodeConfig &config) {
            return make(inp, {param.desc, param.desc + param.nr_desc},
                    config);
        }
        Param param() const {
            mgb_assert(m_desc.size() <= Param::MAX_DESC_SIZE);
            Param ret;
            ret.nr_desc = m_desc.size();
            std::copy(m_desc.begin(), m_desc.end(), ret.desc);
            return ret;
        }

    private:
        std::vector<AxisDesc> m_desc;

        TensorLayout axis_manip_get_output_layout(
                const TensorLayout &inp_layout) const override;
};

namespace intl {

MGB_DEFINE_CLS_WITH_SUPER(
        ModifySubtensorImplHelper, FancyIndexingHelper) // {

    void init_output_static_infer_desc() override final;
    void scn_do_execute() override final;

    /*!
     * \brief implement the actual modifycation
     *
     * Note that this method may be used both for exec and static value infer
     *
     * \param sub a view of the dest subtensor on target tensor
     */
    virtual void modify(DeviceTensorND &sub, const DeviceTensorND &val) = 0;

    protected:
        using Super::Super;
};

} // intl

/*!
 * \brief get subtensor in a python-like way
 */
MGB_DEFINE_OPR_CLASS(Subtensor,
        intl::ReadonlyFwdHelper<intl::FancyIndexingHelper>) // {

    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void mem_plan_fwd_in2out_readonly() override;
    void init_rt_force_dynamic_mem_alloc_imply_chain() override;

    public:
        Subtensor(VarNode *inp, const IndexDesc &desc,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar inp, const IndexDesc &desc,
                const OperatorNodeConfig &config = {});
};

/*!
 * \brief replace the value of subtensor by another tensor
 */
MGB_DEFINE_OPR_CLASS(SetSubtensor, intl::ModifySubtensorImplHelper) // {

    void modify(DeviceTensorND &sub, const DeviceTensorND &val) override;

    public:
        MGB_DECL_FANCY_INDEXING_OPR_MODIFY(SetSubtensor);
};

/*!
 * \brief increase the value of subtensor by another tensor
 */
MGB_DEFINE_OPR_CLASS(IncrSubtensor, intl::ModifySubtensorImplHelper) // {
    void modify(DeviceTensorND &sub, const DeviceTensorND &val) override;

    public:
        MGB_DECL_FANCY_INDEXING_OPR_MODIFY(IncrSubtensor);
};

class IndexAt {
    public:
        /*!
         * \brief helper for Subtensor with only index
         * \param index list of pairs of (axis, index)
         */
        static SymbolVar make(SymbolVar inp,
                const std::vector<std::pair<size_t, SymbolVar>> &index,
                const OperatorNodeConfig &config = {});
};


/*!
 * \brief split a tensor along one axis, possibly to different computing nodes
 *
 * Note that the computing nodes could be specified in one of the following
 * ways:
 * 1. If omitted in OperatorNodeConfig, it would be inferred from input
 * 2. Specify one comp_node in OperatorNodeConfig, and all output would reside
 *      on this comp_node
 * 3. Specify comp_node for each output in OperatorNodeConfig
 */
MGB_DEFINE_OPR_CLASS(Split, intl::OutshapeBySymvarOprBase) // {
    public:
        struct Options {
            enum class Method {
                SPECIFY,    //!< specify output sizes
                CALLBACK    //!< output sizes obtained from callback
            };
            Method method;
            size_t nr_part = 0;
            int axis = 0;

            using callback_t = thin_function<std::vector<size_t>(
                    size_t tot_size)>;
            callback_t callback;
            SymbolVarArray partition;

            /*!
             * \brief make split option by splitting into average parts
             */
            static Options make_average(int axis, size_t nr_part);

            static Options make_partition(int axis,
                    const SymbolVarArray &partition);
            static Options make_partition(SymbolVar inp, int axis,
                    const std::vector<size_t> &partition);

            static Options make_callback(int axis, size_t nr_part,
                    callback_t callback);
        };

        Split(VarNode* inp,
                const Options &opt, const OperatorNodeConfig &config);

        static SymbolVarArray make(SymbolVar inp,
                Options opt, const OperatorNodeConfig &config = {});

        const Options& options() const {
            return m_opt;
        }
    private:
        struct OutputSpec {
            TensorShape shape; //! recent inferred shape
            bool mem_fwd_success = false;
            SubTensorSpec subspec;
        };
        bool m_readonly_fwd_called = false;
        std::vector<OutputSpec> m_output_spec;
        Options m_opt;
        size_t m_output_shape_version = 0;

        void init_output_comp_node() override;

        NodeProp* do_make_node_prop() const override;

        void do_execute(ExecEnv &env) override;

        void init_output_static_infer_desc() override;
        void outshape_by_symvar_do_get_output_shape(
                TensorShape &dest, const ShapeInferInfo &shpinfo) override;

        void mem_plan_fwd_in2out_readonly() override;

        void add_input_layout_constraint() override;

        bool infer_shape(size_t out_idx, TensorShape &dest,
                const cg::static_infer::InpVal &inp);

        void on_mem_status_changed();
        OprEventCallback get_opr_event_callback() override final;

        void init_subspec(bool memfwd);

        void on_output_comp_node_stream_changed() override;
        void init_rt_force_dynamic_mem_alloc_imply_chain() override;
};

/*!
 * \brief concat a tensor
 *
 * To concat to a different computing node, specify the destination in
 * OperatorNodeConfig
 */
MGB_DEFINE_OPR_CLASS(Concat, cg::SingleCNOutshapePureByInshapeOprBase) // {
    public:
        using Param = megdnn::param::Axis;
        Concat(const VarNodeArrayView &inp, int axis,
                const OperatorNodeConfig &config);

        static SymbolVar make(
                const VarNodeArrayView &inp, int axis,
                const OperatorNodeConfig &config = {});

        //! for desrialization
        static SymbolVar make(
                const VarNodeArrayView &inp, const Param &param,
                const OperatorNodeConfig &config) {
            return make(inp, static_cast<int>(param.axis), config);
        }

        //! get axis for this concat
        int axis() const {
            return m_axis;
        }

        Param param() const {
            return m_axis;
        }

    private:
        int m_axis;

        void scn_do_execute() override;

        NodeProp* do_make_node_prop() const override;

        void init_output_static_infer_desc() override;
        void add_input_layout_constraint() override;
        void init_output_comp_node() override;

        void get_output_var_shape(
                const TensorShapeArray &inp_shape,
                TensorShapeArray &out_shape) const override;
};

/*!
 * \brief Opr used to pack parameter, all input node must in same device, dtype
 *      and shape is not needed to be same
 * \param offsets: size of 2 * inputs.size()
 *      offsets[i * 2] and offsets[i * 2 + 1] means
 *      the begin and the end of inputs[i]'s offsets in output
 * \param offsets_val: offsets value on cpu
 */
MGB_DEFINE_OPR_CLASS(ParamPackConcat, cg::SingleCNOperatorNodeBase) // {
    //! input pointer buffer
    SmallVector<void*> m_inp_ptr;
    std::vector<dt_int32> m_offsets;
    intl::UniqPtrWithCN<megdnn::ParamPackConcat> m_opr;

    void add_input_layout_constraint() override;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    void on_output_comp_node_stream_changed() override;

public:
    using Param = megdnn::param::Empty;

    Param param() const {
        return {};
    }

    ParamPackConcat(VarNodeArray& inp, VarNode* offsets,
                    const std::vector<dt_int32> offsets_val,
                    const OperatorNodeConfig& config);
    static SymbolVar make(const SmallVector<SymbolVar>& inp,
                          const SymbolVar& offsets,
                          const std::vector<dt_int32> offsets_val,
                          const OperatorNodeConfig& config = {});

    static SymbolVar make(const SmallVector<SymbolVar>& inp,
                          const SymbolVar& offsets,
                          const std::vector<dt_int32> offsets_val, const Param&,
                          const OperatorNodeConfig& config) {
        return make(inp, offsets, offsets_val, config);
    }

    const std::vector<dt_int32>& get_offsets() const {
        return m_offsets;
    }
};

/*!
 * \brief Opr used to split parameter
 * \param offsets: size of 2 * outputs.size()
 *      offsets[i * 2] and offsets[i * 2 + 1] means
 *      the begin and the end of output[i]'s offsets in input
 * \param offsets_val: offsets value on cpu
 * \param shapes: shape of each output
 */
MGB_DEFINE_OPR_CLASS(ParamPackSplit, cg::SingleCNOperatorNodeBase) // {
    TensorShapeArray m_shapes;
    std::vector<dt_int32> m_offsets;

    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    bool infer_shape(size_t index, TensorShape &dest,
            const cg::static_infer::InpVal &inp);
    void init_output_dtype() override;
    void mem_plan_fwd_in2out_readonly() override;
    void add_input_layout_constraint() override;

public:
    ParamPackSplit(VarNode* src, const std::vector<dt_int32> offsets,
                   TensorShapeArray& shapes, const OperatorNodeConfig& config);

    static SymbolVarArray make(const SymbolVar& src,
                               const std::vector<dt_int32> offsets,
                               TensorShapeArray shapes,
                               const OperatorNodeConfig& config = {});

    const std::vector<dt_int32>& get_offsets() const {
        return m_offsets;
    }

    const TensorShapeArray& get_output_shapes() const {
        return m_shapes;
    }

    void init_rt_force_dynamic_mem_alloc_imply_chain() override;
};

/*!
 * \brief change the tensor layout to adapt to new format
 *
 * See docs of megdnn params for more details
 */
MGB_DEFINE_OPR_CLASS(RelayoutFormat,
                     intl::MegDNNOprWrapperFwd<megdnn::RelayoutFormat>)  // {
    public:
        RelayoutFormat(VarNode* src, const Param &param,
                const OperatorNodeConfig &config);
        static SymbolVar make(SymbolVar src, const Param &param,
                const OperatorNodeConfig &config = {});
        void init_output_format() override final;
};

/*!
 * \brief change conv weights layout base on winograd transform.
 *
 * See docs of megdnn params for more details
 */
MGB_DEFINE_OPR_CLASS(WinogradFilterPreprocess,
                     intl::MegDNNOprWrapperFwd<megdnn::WinogradFilterPreprocess>)
    public:
        WinogradFilterPreprocess(VarNode* p0, const Param& param,
                const OperatorNodeConfig& config);
        static SymbolVar make(SymbolVar p0, const Param& param = {},
                const OperatorNodeConfig& config = {});
        void init_output_dtype() override final;
};

} // opr
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
