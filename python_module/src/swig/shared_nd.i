/*
 * $File: shared_nd.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%pythoncode {
from .mgb_helper import SharedNDLazyInitializer
} // pythoncode

%feature("autodoc", """a value stored on computing device and can be modified
by special operators in the graph""") SharedND;
class SharedND {
    public:
        SharedND(CompNode comp_node, PyObject *dtype);

        void _set_init_shape(const std::vector<size_t> &shape);
        void _resize(const std::vector<size_t> &shape);
        void _reset_zero();

        PyObject* _get_npyarr();
        PyObject* _get_dtype();
        std::vector<size_t> _get_shape();

        void _copy_from_npyarr(PyObject *npyarr);
        void _copy_from_value_proxy(CompGraphCallbackValueProxy &value);
        void _share_from_value_proxy(CompGraphCallbackValueProxy &value);
        static SharedND _from_symvar(SymbolVar symvar);

        void _set_copy_sync(bool flag);
        uintptr_t _pubapi_dev_tensor_ptr(int version);

        void copy_to_sub_from_shared(
                int axis, ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step,
                const SharedND &rhs);

        void copy_from_shared_sub(const SharedND &rhs,
                int axis, ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step);

        CompNode _get_comp_node();

        SymbolVar _as_sym_var(CompGraph &graph, const std::string &name,
                bool volatile_);

        void _share_memory_from(const SharedND &rhs, size_t begin);

        void _reset_dev_tensor(const SharedND& rhs);

        %include "shared_nd_SharedND.py"
};
%template(_VectorSharedND) std::vector<SharedND>;

class _HostSharedND {
    public:
        _HostSharedND(CompNode node, PyObject *dtype);
        static _HostSharedND make_proxy(SymbolVar var);

        SymbolVar _as_sym_var(CompGraph &cg, bool enable_static_infer,
                              const std::string &name);
        PyObject* _get_dtype();
        void _resize(const std::vector<size_t> &shape);
        void _copy_from_npyarr(PyObject *npyarr, bool borrow);
        void _enable_borrow_on_cpu(bool flag);
        std::string __repr__() const;

        %include "shared_nd_HostSharedND.py"
};


%feature("autodoc",
"""a scalar value that can be modified after it has been created;
compared to :class:`SharedND`, it has the advantage that no comp node needs to
be specified.""") SharedScalar;
class SharedScalar {
    public:
        SharedScalar(PyObject *val);
        void _set(PyObject *val);
        PyObject* _get();
        bool _dtype_locked();
        void _lock_dtype();
        SymbolVar _as_sym_var(CompGraph &cg, CompNode &cn);

        %pythoncode {

            def lock_dtype(self):
                """lock dtype so further set() calls must pass the same dtyped
                value"""
                self._lock_dtype()

            @property
            def dtype_locked(self):
                """whether dtype is locked"""
                return self._dtype_locked()

            def set(self, val):
                self._set(val)

            def get(self):
                """get the value stored in this SharedScalar"""
                return self._get()[0]

            def __getstate__(self):
                state = self.__dict__.copy()
                del state['this']
                state['__shared_scalar_value'] = self.get()
                state['__shared_scalar_dtype_locked'] = self.dtype_locked
                return state

            def __setstate__(self, state):
                val = SharedScalar(state.pop('__shared_scalar_value'))
                if state.pop('__shared_scalar_dtype_locked', True):
                    val._lock_dtype()
                self.this = val.this
                for k, v in state.items():
                    self.__dict__[k] = v

            def __repr__(self):
                return 'SharedScalar({})'.format(self.get())
        }
};


// vim: ft=swig
