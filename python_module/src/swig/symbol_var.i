/*
 * $File: symbol_var.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%{
using mgb::cg::SymbolVar;
%}

%feature("autodoc",
"representing a symbolic variable in a computing graph") SymbolVar;

class SymbolVar {
public:
    SymbolVar flatten();
    SymbolVar rename(const std::string &name);
    bool allow_shape_change();

    %extend {

        SymbolVar fill_retain_dtype(PyObject *val) {
            return fill_retain_dtype(*$self, val);
        }

        CompGraph _get_owner_graph() {
            mgb_assert($self->node());
            auto cg = $self->node()->owner_graph()->shared_from_this();
            return CompGraph::make_from_shared_ptr(cg);
        }

        Operator _get_owner_opr() {
            mgb_assert($self->node());
            return Operator{$self->node()->owner_opr()};
        }

        CompNode _get_comp_node() {
            mgb_assert($self->node());
            return $self->node()->comp_node();
        }

        const std::string& _get_name() const {
            mgb_assert($self->node());
            return $self->node()->name();
        }

        size_t _get_id() const {
            mgb_assert($self->node());
            return $self->node()->id();
        }

        std::vector<size_t> _get_imm_shape() {
            mgb_assert($self->node());
            return npy::shape2vec($self->node()->shape());
        }

        PyObject* _get_inferred_value() {
            return get_symvar_inferred_value(*$self);
        }

        bool _is_valid() const {
            return $self->node();
        }

        PyObject* _get_dtype() const {
            return npy::dtype_mgb2np($self->dtype());
        }

        CompGraphCallbackValueProxy _eager_eval_get_value() const {
            CompGraphCallbackValueProxy ret;
            ret.setup($self->eager_eval_get_value(), false);
            return ret;
        }

        void _reeval_if_eager_eval() {
            auto &&var = $self->node();
            mgb_assert(var);
            auto &&cg = var->owner_graph();
            if (cg->options().eager_evaluation) {
                mgb_assert(var->owner_opr()->inserted_in_graph());
                cg->insert_opr(std::unique_ptr<mgb::cg::OperatorNodeBase>(
                    var->owner_opr()));
            }
        }

        bool _is_shared_device_tensor() {
            if ($self->node()
                        ->owner_opr()
                        ->same_type<mgb::opr::SharedDeviceTensor>())
                return true;
            return false;
        }

        %include "symbol_var_SymbolVar.py"

    }

};

typedef std::vector<SymbolVar> SymbolVarArray;
%template(_VectorSymbolVar) std::vector<SymbolVar>;

// SymbolVarArray compatibility; see symbol_var_array.i for more details
%typemap(out) SymbolVarArray {
    $result = swig::from(static_cast<const std::vector<SymbolVar>&>($1));
}
%typemap(directorin) const SymbolVarArray& {
    $input = swig::from(static_cast<const std::vector<SymbolVar>&>($1));
}

// vim: ft=swig
