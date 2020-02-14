/**
 * \file src/opr/include/megbrain/opr/internal/indexing_helper.h
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
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/internal/mixin_base.h"

namespace mgb {
namespace opr {
namespace indexing {

    /*!
     * \brief axis number that can be either positive or negative; negative
     *      meaning counting from the end
     */
    class AxisNum {
        int m_num = 0;

        public:
            AxisNum() = default;

            AxisNum(int num):
                m_num{num}
            {
            }

            //! get actual axis for given ndim, with boundary check
            size_t get(size_t ndim) const;

            int get_raw() const {
                return m_num;
            }

            bool operator == (const AxisNum &rhs) const {
                return m_num == rhs.m_num;
            }

            bool operator != (const AxisNum &rhs) const {
                return m_num != rhs.m_num;
            }
    };

    /*!
     * \brief index on one axis
     */
    struct AxisIndexer {
        AxisNum axis;

        //! index a sub interval in a similar manner to python's indexing
        SymbolVar begin, end, step;

        //! get a single slice, so the corresponding axis would be removed
        SymbolVar idx;

        //! index an axis on an interval
        static AxisIndexer make_interval(
                AxisNum axis,
                Maybe<SymbolVar> begin, Maybe<SymbolVar> end,
                Maybe<SymbolVar> step);

        //! index an axis with scalar or vector indexer
        static AxisIndexer make_index(AxisNum axis, SymbolVar idx);

        /*!
         * \brief return true if axis of *lhs* is larger than (i.e. with smaller
         *      stride in contiguous case) axis of *rhs*
         */
        static bool cmp_by_axis_rev(
                const AxisIndexer &lhs, const AxisIndexer &rhs) {
            auto a0 = lhs.axis.get_raw(), a1 = rhs.axis.get_raw();
            return (a0 < 0) == (a1 < 0) ? a0 > a1 : a0 < 0;
        }
    };
    using IndexDesc = std::vector<AxisIndexer>;

} // namespace indexing

namespace intl {

/*!
 * \brief base class for fancy indexing oprs
 *
 * Currently there are two families of such oprs: Subtensor and MultiAxisVec
 */
MGB_DEFINE_CLS_WITH_SUPER(
        FancyIndexingHelper, cg::SingleCNOperatorNodeBase) // {

    public:
        using AxisIndexer = indexing::AxisIndexer;
        using IndexDesc = indexing::IndexDesc;
        using InputTensorReplacer = thin_function<
            DeviceTensorND(const TensorShape &shape)>;

        //! original index desc given by user, sorted as descending axis num
        const IndexDesc& index_desc() const {
            return m_index_desc;
        }

        //! input tensor replacer func
        const InputTensorReplacer& input_tensor_replacer() const {
            return m_input_tensor_replacer;
        }

    protected:
        //! non-null for input vars that correspond to AxisIndexer with valid
        //! AxisIndexer::idx; null if interval AxisIndexer is used
        std::vector<const AxisIndexer*> m_input2idxonly_axis_indexer;

        /*!
         * \param data input data
         * \param value value for subtensor modifier; nullptr for subtensor
         *      getter
         * \param require_scalar_index whether indexers are required to be scalar,
         *      so we can use memory forwarding; true for Subtensor oprs, and
         *      false for IndexingMultiAxisVec oprs
         * \param input_tensor_replacer set a callback function that replaces
         *      the input tensor during scn_do_execute. This can be non-empty
         *      only when *value* is not null; it would be given the shape of
         *      input tensor and should return a tensor to be modified. It is
         *      currently only used for optimizing grad sum in loop. If this
         *      callback is set, then the input/output vars are changed as
         *      following:
         *          1. only shape of data input would be used (i.e. its dep type
         *             changed to shape) to validate shape of tensor given by
         *             the replacer;
         *          2. data output would be empty and marked as volatile. The
         *             tensor given by the callback would be directly modified.
         */
        FancyIndexingHelper(
                const OperatorNodeBaseCtorParam &opr,
                VarNode *data, VarNode *value, const IndexDesc &index_desc,
                bool require_scalar_index,
                const InputTensorReplacer &input_tensor_replacer = {});


        /*!
         * \brief get a SubTensorSpec using value infer result for input vars
         *
         * Note that if require_scalar_index is true, then the SubTensorSpec is
         * directly ready for producing output; otherwise the derived opr needs
         * to perform further computing.
         */
        SubTensorSpec fancy_indexing_make_sub_spec(
                const TensorLayout &inp_layout);

        /*!
         * \brief get a SubTensorSpec using value infer result in static infer
         *      func
         * \param fake_single_idx whether to use a const value to replace
         *      indexing on an axis, so shape inference can work with unknown
         *      indexing value
         */
        SubTensorSpec fancy_indexing_make_sub_spec(
                const TensorLayout &inp_layout,
                const cg::static_infer::InpVal &infer_inp,
                size_t infer_inp_start, bool fake_single_idx = false);

        /*!
         * \brief get (data, value) pairs to implement modification by indexing
         *      opr; data on given sub should be modified by value
         *
         * Must be called from scn_do_execute.
         */
        std::pair<DeviceTensorND, DeviceTensorND>
        fancy_indexing_get_tensors_for_modify_in_scn_do_execute();

        //! see notes on the constructor
        bool has_input_tensor_replacer() const {
            return static_cast<bool>(m_input_tensor_replacer);
        }

        NodeProp* do_make_node_prop() const override;

    private:
        const size_t m_idx_inp_start;
        const bool m_require_scalar_index, m_is_assign_opr;

        IndexDesc m_index_desc;

        //! number of AxisIndexer with valid AxisIndexer::idx
        size_t m_nr_axis_single_idx = 0;

        //! current infer result for indexing var values, for arg passing to
        //! do_make_sub_spec
        std::vector<const DeviceTensorND*> m_value_infer_result;

        InputTensorReplacer m_input_tensor_replacer;

        //! get a SubTensorSpec from m_value_infer_result
        SubTensorSpec do_make_sub_spec(const TensorLayout &inp_layout) const;

        void init(const IndexDesc &index_desc);

        //! writable forward inp[0] to out[0] if value in ctor is not null
        void mem_plan_fwd_in2out_writable() override final;
};

} // namespace intl

} // namespace opr
} // namespace mgb

#define MGB_DECL_FANCY_INDEXING_OPR_GET(_opr) \
    _opr(VarNode* inp, const IndexDesc& desc, \
            const OperatorNodeConfig& config); \
    static SymbolVar make(SymbolVar inp, \
            const IndexDesc& desc, \
            const OperatorNodeConfig& config = {})

#define MGB_IMPL_FANCY_INDEXING_OPR_GET(_opr, _name, _require_scalar_index, \
        ctor_body...) \
_opr::_opr(VarNode *inp, const IndexDesc &desc, \
        const OperatorNodeConfig &config): \
    Super({inp->owner_graph(), config, _name, {inp}}, \
            inp, nullptr, desc, _require_scalar_index) \
{ \
    ctor_body; \
} \
SymbolVar _opr::make(SymbolVar inp, const IndexDesc &desc, \
        const OperatorNodeConfig &config) { \
    return inp.insert_single_output_opr<_opr>(inp.node(), desc, config); \
} \
MGB_DYN_TYPE_OBJ_FINAL_IMPL(_opr)

#define MGB_DECL_FANCY_INDEXING_OPR_MODIFY(_opr) \
    _opr(VarNode *inp, VarNode *value, \
            const IndexDesc &desc, \
            const OperatorNodeConfig &config, \
            const InputTensorReplacer &input_tensor_replacer); \
    static SymbolVar make(SymbolVar inp, SymbolVar value, \
            const IndexDesc &desc, \
            const OperatorNodeConfig &config = {}, \
            const InputTensorReplacer &input_tensor_replacer = {})

#define MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(_opr, _name, _require_scalar_index) \
_opr::_opr(VarNode *inp, VarNode *value, const IndexDesc &desc, \
        const OperatorNodeConfig &config, \
        const InputTensorReplacer &input_tensor_replacer): \
    Super({inp->owner_graph(), config, _name, {inp, value}}, \
            inp, value, desc, _require_scalar_index, input_tensor_replacer) \
{ \
} \
SymbolVar _opr::make(SymbolVar inp, SymbolVar value, const IndexDesc &desc, \
        const OperatorNodeConfig &config, \
        const InputTensorReplacer &input_tensor_replacer) { \
    return inp.insert_single_output_opr<_opr>( \
            inp.node(), value.node(), desc, config, input_tensor_replacer); \
} \
MGB_DYN_TYPE_OBJ_FINAL_IMPL(_opr)

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

