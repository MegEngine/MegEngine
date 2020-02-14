/*
 * $File: comp_graph.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */


%pythoncode{
from .mgb_helper import copy_output, FuncOutputSaver
import json
} // pythoncode

%feature("autodoc", """a callable object compiled from :class:`CompGraph`.

.. note::

    Only the most recently compiled AsyncExec object can be used.
""") AsyncExec;
%feature("autodoc", """explicitly release the underlying staticially allocated
device memory""") AsyncExec::clear_device_memory;
class AsyncExec {
    public:
        AsyncExec() = delete;

        void _execute();
        void _wait();
        double _get_prev_exec_time();
        std::string _to_json_str();
        SymbolVarArray _find_mutable_input();
        std::vector<std::pair<CompNode, size_t>>
        _update_static_alloc_plan_and_get_size();

        void clear_device_memory();

        %include "comp_graph_impl_AsyncExec.py"
};

%template(_VectorAsyncExec) std::vector<AsyncExec>;

%feature("autodoc", """use device memory manager in another computing graph to
manage memory of this graph, so their memories can be shared. This is safe only
when :class:`AsyncExec` compiled from these graphs do not run concurrently.""")
CompGraph::share_device_memory_with;
%feature("valuewrapper") CompGraph;
class CompGraph {
    public:
        CompGraph();

        AsyncExec _do_compile(bool copy, bool optimize_for_inference);
        std::vector<AsyncExec> _do_compile_multi_part();
        void _add_output_spec(SymbolVar &var,  _CompGraphCallback *callback);
        void _add_multi_part_endpoint();
        void _clear_output_spec();
        size_t _release();

        CompGraph& share_device_memory_with(CompGraph &other);

        PyObject* _user_data();
        void clear_device_memory();

        %extend {
            size_t _id() const {
                return $self->get().id();
            }

            size_t _get_ptr_addr() const {
                return reinterpret_cast<size_t>(&$self->get());
            }

            std::string __repr__() const {
                auto &&graph = $self->get();
                return mgb::ssprintf("<CompGraph #%zu at %p>", graph.id(), &graph);
            }
        }

        %include "comp_graph_impl_CompGraph.py"
};

%include "comp_graph_tools.i"

// vim: ft=swig
