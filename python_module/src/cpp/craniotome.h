/**
 * \file python_module/src/cpp/craniotome.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief extend megbrain operators in python
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph/operator_node.h"
#include "./megbrain_wrap.h"

using TensorShapeVec = std::vector<std::vector<size_t>>;
using SymbolVarArray = mgb::SymbolVarArray;

namespace mgb {
namespace opr {
class Craniotome;
}  // namespace opr
}  // namespace mgb

class CraniotomeDesc: public mgb::Hashable {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    mutable PyObject *m_py_self = nullptr;

    bool is_same_st(const mgb::Hashable &rhs) const override;

    size_t hash() const override;

    public:
        struct NodeFlag {
            static constexpr uint32_t
                DYNAMIC_OUTPUT_SHAPE = 1 << 0,
                DISALLOW_DUPLICATE = 1 << 1,
                ALLOW_EMPTY_OUTPUT = 1 << 2,
                DISABLE_SYS_MEM_ALLOC = 1 << 3;
        };
        virtual ~CraniotomeDesc() = default;

        mgb::opr::Craniotome* owner_opr = nullptr;

        //! get final py object that implements this interface
        PyObject* py_self() const ;

        //! store self in \p result which is a list
        virtual void _setup_self(PyObject *result) const = 0;

        virtual bool _is_same(PyObject *rhs) const = 0;

        virtual uint32_t _node_flag() const = 0;

        virtual size_t _hash() const = 0;

        virtual std::string _get_opr_type_name() = 0;

        virtual size_t _get_nr_outputs() = 0;

        virtual void _execute(
                const std::vector<CompGraphCallbackValueProxy> &inputs,
                std::vector<SharedND> &outputs) = 0;

        /*!
         * \brief infer output shape if DYNAMIC_OUTPUT_SHAPE is not set
         */
        virtual TensorShapeVec _infer_shape(
                const TensorShapeVec &inp_shape) = 0;

        virtual SymbolVarArray _grad(
                size_t wrt_idx,
                const SymbolVarArray &inputs,
                const SymbolVarArray &outputs,
                const SymbolVarArray &out_grad) = 0;

        virtual size_t _get_nr_dev_comp_order_deps() = 0;

        mgb::thin_function<SymbolVarArray()> _get_all_io_vars;

        /*!
         * \brief get output dtypes from input dtypes
         * \param[in] input_dtypes python list of input
         * \param[out] result initialized as an empty python list, and should
         *      be filled with output dtypes
         * \return whether user has set the dtype
         */
        virtual bool _init_output_dtype(
                PyObject *input_dtypes, PyObject *result) = 0;

        /*!
         * \brief get computing graph when no input var is provided
         */
        virtual CompGraph _get_comp_graph() = 0;

        /*!
         * \brief copy this CraniotomeDesc
         *
         * The implementation must call _set_copy_result() to return the result;
         * this is used to bypass some swig issues.
         */
        virtual void _copy() const = 0;
        mutable mgb::thin_function<void(CraniotomeDesc*)> _set_copy_result;

        /*!
         * \brief setup params for serialization
         * \param output an allocated list. One or two elements should be
         *      inserted in it after this function returns: the first element
         *      should be a string, indicating the id to be passed to
         *      opr_maker_loader; the second element, if exists, must be a byte
         *      object containing extra param that should be written to file.
        */
        virtual void _setup_serialize_params(PyObject *output) const = 0;

        /*!
         * \brief callback invoked when the graph is compiled or when func is
         *      destructed
         *
         * If the graph is compiled but not executed, this function might not be
         * called
         *
         * \param used_outputs an array indices indicating the used output vars;
         *      this argument being empty means that the previously compiled
         *      func is destructed
         */
        virtual void _on_graph_compile_or_func_del(
                const std::vector<size_t>& used_outputs) = 0;
};


namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(Craniotome, cg::SingleCNOutshapePureByInshapeOprBase) // {
    class FuncDelCallbackInvoker;
    using NodeFlag = CraniotomeDesc::NodeFlag;

    bool m_on_graph_compile_called = false;
    const uint32_t m_node_flag;

    //! DEV_COMP_ORDER inputs are at the tail of input array; this is the
    //! number of DEV_VALUE inputs, and also the index of the first
    //! DEV_COMP_ORDER input
    size_t m_nr_dev_value_inp;

    std::unique_ptr<CraniotomeDesc> m_desc;

    //! previously inferred shape; used when there is no input and
    //! m_is_dynamic_output_shape is set to true
    Maybe<TensorShapeArray> m_prev_inferred_shape;

    void scn_do_execute() override;
    void get_output_var_shape(const TensorShapeArray &inp_shape,
            TensorShapeArray &out_shape) const override;

    void add_input_layout_constraint() override;

    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    NodeProp* do_make_node_prop() const override;

    bool output_no_sys_mem_alloc() const {
        return m_node_flag & (NodeFlag::DYNAMIC_OUTPUT_SHAPE |
                              NodeFlag::DISABLE_SYS_MEM_ALLOC);
    }

    public:
        Craniotome(mgb::ComputingGraph *graph,
                std::unique_ptr<CraniotomeDesc> desc,
                const VarNodeArray &inputs, const OperatorNodeConfig &config);

        ~Craniotome() noexcept;

        static SymbolVarArray make(
                std::unique_ptr<CraniotomeDesc> desc,
                const SymbolVarArray &inputs,
                const OperatorNodeConfig &config = {});

        const CraniotomeDesc& desc() const {
            return *m_desc;
        }

        size_t nr_dev_value_inp() const {
            return m_nr_dev_value_inp;
        }
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
