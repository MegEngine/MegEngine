/*
 * $File: operator.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%{
#include "opr_helper.h"
#include "opr_defs.h"

using ::mgb::cg::OperatorNodeConfig;

using _AxisIndexer = AxisIndexer;

static inline PyObject* _to_mgb_supported_dtype(PyObject* dtype) {
    return ::npy::to_mgb_supported_dtype(dtype);
}

%}

%feature("autodoc", "Extra configuration for an operator") OperatorNodeConfig;
class OperatorNodeConfig {
    public:
        OperatorNodeConfig();
        void name(const std::string &name);
        void comp_node(const CompNode &node);

        %extend {
            void comp_node_arr(const std::vector<CompNode> &arr) {
                OperatorNodeConfig::CompNodeArray tarr(arr.begin(), arr.end());
                $self->comp_node_arr(tarr);
            }

            CompNode require_comp_node() {
                mgb_assert($self->comp_node().size() == 1,
                        "comp_node is required for the config");
                return $self->comp_node()[0];
            }

            void output_dtype(PyObject* dtype) {
                $self->output_dtype(npy::dtype_np2mgb(dtype));
            }
        }
};

%feature("autodoc",
          "representing a operator node in a computing graph") Operator;
class Operator {
public:
    %extend {
        size_t _get_id() const {
            return $self->id();
        }

        const std::string& _get_name() const {
            return $self->name();
        }

        SymbolVarArray _get_inputs() {
            return $self->inputs();
        }

        SymbolVarArray _get_outputs() {
            return $self->outputs();
        }

        CompGraph _get_owner_graph() {
            const auto& cg = $self->get_owner_graph();
            return CompGraph::make_from_shared_ptr(cg);
        }

        %include "operator.py"
    }
};

%template(_VectorOperator) std::vector<Operator>;

class _AxisIndexer {
public:
    static _AxisIndexer make_interval(int axis, SymbolVar begin, SymbolVar end,
                                      SymbolVar step);

    static _AxisIndexer make_index(int axis, SymbolVar idx);
};
%template(_VectorAxisIndexer) std::vector<_AxisIndexer>;

%inline {
    // all defined in opr_helper.cpp
    SymbolVarArray _create_opr(
        const char *name, const SymbolVarArray &inputs, PyObject *params,
        const OperatorNodeConfig &config);

    SymbolVar _create_subtensor_like_opr(
            const std::string &name,
            const SymbolVarArray& inputs,
            const std::vector<_AxisIndexer> &idx,
            const OperatorNodeConfig &config);

    SymbolVar _make_immutable(
            CompGraph &comp_graph, PyObject *npyarr, PyObject *dtype,
            const OperatorNodeConfig &config);
}

PyObject* _to_mgb_supported_dtype(PyObject *dtype);

%include "../cpp/opr_defs.h"

%pythoncode {

def make_opr_config(name=None, comp_node=None, output_dtype=None):
    """make :class:`.OperatorNodeConfig` from given name or comp_node

    :type name: None or str
    :param name: name for the operator
    :type comp_node: None or comp_node-compatible or iterable of
        comp_node-compatible
    :param comp_node: a single comp_node, or iterable of comp_nodes
    :type dtype: None or numpy-dtype compatible
    :param dtype: the specified dtype the operator.
    """
    rst = OperatorNodeConfig()
    if comp_node is not None:
        if isinstance(comp_node, str):
            rst.comp_node(as_comp_node(comp_node))
        elif isinstance(comp_node, collections.Iterable):
            vec = _VectorCompNode()
            for i in comp_node:
                vec.push_back(as_comp_node(i))
            rst.comp_node_arr(vec)
        else:
            rst.comp_node(as_comp_node(comp_node))
    if name is not None:
        assert isinstance(name, str)
        rst.name(name)
    if output_dtype is not None:
        rst.output_dtype(output_dtype)

    return rst

} // %pythoncode

// vim: ft=swig foldmethod=marker foldmarker=f{{{,f}}}
