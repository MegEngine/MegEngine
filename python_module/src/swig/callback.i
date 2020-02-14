/*
 * $File: callback.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%feature("autodoc",
"""It is used to be passed as arguments to callbacks (used in
:meth:`.CompGraph.compile`, :func:`.callback_injector`, and
:meth:`.CraniotomeBase.execute`). Object of this type could also be directly
passed to :meth:`.SharedND.set_value`, to bypass some host and device
communication. Note that the underlying buffer may be reused after the callback
returns, so reference to this object should not be passed outside of the
callback, and :meth:`get_value` should be called immediately if the numerical
value is needed.""")
CompGraphCallbackValueProxy;

class CompGraphCallbackValueProxy {
    public:

        PyObject* _get_npyarr();
        PyObject* _get_dtype();
        std::vector<size_t> _get_shape();

        uintptr_t _pubapi_dev_tensor_ptr(int version);
        CompNode _get_comp_node();

        %pythoncode{

            @property
            def shape(self):
                """get shape of the var

                :type: tuple of int
                """
                return tuple(map(int, self._get_shape()))

            @property
            def comp_node(self):
                """get comp node of the var

                :type: :class:`.CompNode`
                """
                return self._get_comp_node()

            @property
            def dtype(self):
                """get data type of the var

                :type: :class:`.numpy.dtype`
                """
                return self._get_dtype()

            def get_value(self, *, borrow_mem=False):
                """get value as numpy array

                :param borrow_mem: whether to forward internal buffer with
                    zero-copy; if True, the content in returned buffer would be
                    modified directly by asynchronous graph execution.
                """
                ret = self._get_npyarr()
                if not borrow_mem:
                    ret = ret.copy()
                return ret

            @property
            def dev_ptr(self):
                """this method is DEPRECATED; use :meth:`pubapi_dev_tensor_ptr`
                instead"""
                return self._pubapi_dev_tensor_ptr(0)

            @property
            def pubapi_dev_tensor_ptr(self):
                """get a pointer to the corresponding mgb::pubapi::DeviceTensor object

                :rtype: int
                :return: the address as an integer
                """
                return self._pubapi_dev_tensor_ptr(1)
        }
};
%template(_VectorCompGraphCallbackValueProxy)
    std::vector<CompGraphCallbackValueProxy>;

%feature("director") _CompGraphCallback;
class _CompGraphCallback {
    public:
        _CompGraphCallback();

        void set_eager_copy(bool flag);

        virtual ~_CompGraphCallback();
        virtual void call(std::vector<CompGraphCallbackValueProxy> &value) = 0;
};

%feature("director") _SplitPartCallback;
class _SplitPartCallback {
    public:
        _SplitPartCallback();
        virtual ~_SplitPartCallback();

        virtual std::vector<size_t> call(size_t tot_size) = 0;
};

%feature("director") _SetGradCallback;
class _SetGradCallback {
    public:
        _SetGradCallback();
        virtual ~_SetGradCallback();

        virtual SymbolVar call(CompGraph &graph) = 0;
        virtual bool empty() = 0;
};

%feature("director") _TimeoutCallback;
class _TimeoutCallback {
    public:
        _TimeoutCallback();
        virtual ~_TimeoutCallback();

        virtual bool call() = 0;
};

%pythoncode{
import collections
import inspect
from .mgb_helper import callback_lazycopy

class _CompGraphCallbackPyWrapper(_CompGraphCallback):
    """wraps around a callable to be used as comp graph callback"""

    def __init__(self, f):
        super().__init__()
        if isinstance(f, callback_lazycopy):
            f = f.func
            self.set_eager_copy(False)
        else:
            self.set_eager_copy(True)
            assert isinstance(f, collections.Callable)
        self._func = f
        self.__disown__()

    def call(self, value):
        if value.size() == 1:
            self._func(value[0])
        else:
            self._func(value)


_CompGraphCallbackPyWrapperNoEager = lambda f: (
    _CompGraphCallbackPyWrapper(callback_lazycopy(f)))

class _SplitPartCallbackPyWrapper(_SplitPartCallback):
    def __init__(self, f):
        super().__init__()
        assert isinstance(f, collections.Callable)
        self._func = f
        self.__disown__()

    def call(self, size):
        return tuple(map(int, self._func(size)))


class _SetGradCallbackPyWrapper(_SetGradCallback):
    def __init__(self, f):
        super().__init__()
        if f is None:
            self._func = None
        else:
            assert isinstance(f, collections.Callable)
            nr_arg = len(list(filter(
                lambda x: (
                    x.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                    x.default == inspect.Parameter.empty),
                inspect.signature(f).parameters.values())))
            if not nr_arg:
                f = lambda graph, f0=f: f0()
            else:
                assert nr_arg == 1, 'bad callback for SetGrad: {}'.format(f)
            self._func = f

        self.__disown__()

    def call(self, graph):
        if self._func is None:
            return SymbolVar()

        ret = self._func(graph)
        if ret is None:
            ret = SymbolVar()
        else:
            assert isinstance(ret, SymbolVar), (
                'bad return value for var maker: {!r}'.format(ret))
        return ret

    def empty(self):
        return self._func is None


class _TimeoutCallbackPyWrapper(_TimeoutCallback):
    def __init__(self, f):
        super().__init__()
        assert isinstance(f, collections.Callable)
        self._func = f
        self.__disown__()
    
    def call(self):
        return bool(self._func())


} // %pythoncode

// vim: ft=swig
