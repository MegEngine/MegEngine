/*
 * $File: craniotome.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%{
#include "craniotome.h"
%}

typedef std::vector<std::vector<size_t>> TensorShapeVec;
%template(_VectorTensorShape) std::vector<std::vector<size_t>>;

%feature("director") CraniotomeDesc;
class CraniotomeDesc {

    public:
        virtual ~CraniotomeDesc() = default;

        virtual void _setup_self(PyObject *result) const = 0;

        virtual bool _is_same(PyObject *rhs) const = 0;

        virtual uint32_t _node_flag() const = 0;

        virtual size_t _hash() const = 0;

        virtual std::string _get_opr_type_name() = 0;

        virtual size_t _get_nr_outputs() = 0;

        virtual void _execute(
                const std::vector<CompGraphCallbackValueProxy> &inputs,
                std::vector<SharedND> &outputs) = 0;

        virtual TensorShapeVec _infer_shape(
                const TensorShapeVec &inp_shape) = 0;

        virtual SymbolVarArray _grad(
                size_t wrt_idx,
                const SymbolVarArray &inputs,
                const SymbolVarArray &outputs,
                const SymbolVarArray &out_grad) = 0;

        virtual size_t _get_nr_dev_comp_order_deps() = 0;

        SymbolVarArray _get_all_io_vars();

        virtual bool _init_output_dtype(
                PyObject *input_dtypes, PyObject *result) = 0;

        virtual CompGraph _get_comp_graph() = 0;

        virtual void _copy() const = 0;
        void _set_copy_result(CraniotomeDesc *result);

        virtual void _setup_serialize_params(PyObject *output) const = 0;

        virtual void _on_graph_compile_or_func_del(
                const std::vector<size_t>& used_outputs) = 0;

        %extend {
            CompNode _get_comp_node() {
                mgb_assert($self->owner_opr);
                return $self->owner_opr->comp_node();
            }

            size_t _get_opr_id() {
                mgb_assert($self->owner_opr);
                return $self->owner_opr->id();
            }
        }
};

%inline {
    static SymbolVarArray make_opr_from_craniotome_desc(
            CraniotomeDesc *desc,
            const SymbolVarArray inputs,
            const OperatorNodeConfig &config) {

        return mgb::opr::Craniotome::make(
            std::unique_ptr<CraniotomeDesc>(desc), inputs, config);
    }
}

// vim: ft=swig
