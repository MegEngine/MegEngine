%pythoncode{

@property
def id(self):
    """an integer increasing ID"""
    return self._id()

def __eq__(self, rhs):
    return isinstance(rhs, CompGraph) and self.id == rhs.id

def __hash__(self):
    return self.id

@property
def user_data(self):
    """get a dict that is associated with this computing graph to store
    arbitrary user data"""
    return self._user_data()

def _process_output_spec(self, inputs, outspec):
    """process user-provided output spec and add to the output staging list of
    this graph

    :return: a callable ``f(func)` to update compiled :class:`.AsyncExec` status
    """
    assert outspec

    if isinstance(outspec, copy_output):
        outspec = [outspec]
        expand_single_output = True
    else:
        expand_single_output = False

    var2output_saver = {}
    output_vars = []

    for spec in outspec:
        if isinstance(spec, copy_output):
            var = spec.symvar
            output_vars.append(var)
            if var in var2output_saver:
                continue

            callback = FuncOutputSaver(spec.borrow_mem)
            var2output_saver[var] = callback
        elif isinstance(spec, SymbolVar):
            var = spec
            callback = None
        else:
            var, callback = spec
            assert isinstance(var, SymbolVar)
        if callback is not None:
            callback = _CompGraphCallbackPyWrapper(callback)
        self._add_output_spec(var, callback)

    def update(func):
        assert isinstance(func, AsyncExec)
        func.inputs = inputs
        if output_vars:
            func._var2output_saver = var2output_saver
            func._expand_single_output = expand_single_output
            func.outputs = output_vars

    return update

def compile(self, inputs, outspec, *, copy=False, optimize_for_inference=False):
    """Compile the graph to get a callable function for numerical evaluation

    .. warning::

        If ``compile()`` is called multiple times, only the most recent result
        function can be used.

    :type inputs: iterable of :class:`.SymbolVar` or None
    :param inputs: specifying the positional parameters to be passed to the
        generated function, or use None for keyword params only
    :type outspec: iterable of *single_outspec*
    :param outspec: specifying how the compiled function should
        return outputs. Each *single_outspec* may be one of the
        following forms:

            * a pair of (var, callback), the callback would be called during
              function execution with a :class:`.CompGraphCallbackValueProxy`
              argument corresponding to the given symbolvar. Additionally,
              *callback* may be wrapped by :class:`.callback_lazycopy`; see the
              its document for details.
            * a single :class:`.SymbolVar`, to ensure this var is computed (so
              the non-pure operators on its dependency path could take effect)
            * a :class:`.copy_output` object, so the var's value would be
              copied to the return value of compiled function. If there is one
              such spec, the function would be synchronous.
    :param copy: whether to copy the graph
    :param optimize_for_inference: whether to run
        :func:`.optimize_for_inference` on the output vars before compiling
    :rtype: :class:`.AsyncExec`
    """

    self._clear_output_spec()
    ret_update = self._process_output_spec(inputs, outspec)
    ret = self._do_compile(copy, optimize_for_inference)
    ret_update(ret)
    return ret

def compile_outonly(self, outputs, *, inputs=None):
    """compile for only output; (almost) equavalent to
    ``self.compile(inputs, [copy_output(i) for i in outputs])``

    :type outputs: :class:`.SymbolVar` or list of
        :class:`.SymbolVar`
    :param outputs: the output symbol vars
    """
    if isinstance(outputs, SymbolVar):
        outputs = copy_output(outputs)
    else:
        assert isinstance(outputs, collections.Iterable), (
            '{} not iterable'.format(outputs))
        outputs = [copy_output(i) for i in outputs]

    return self.compile(inputs, outputs)

def compile_multi_part(self, io_specs):
    """Compile multiple functions for partial execution. Each function would
    only execute the oprs necessary to compute current outspec, and intermediate
    results from previous functions are reused. The functions would share
    underlying device storage with this graph.

    .. warning::

        Each individual partial function would have a newly created computing
        graph. Therefore plugins attached on this graph would not be effective
        on the partial functions.

    :param io_specs: input/output specifications as a list of
        ``(inputs, outspec)`` pairs. Each pair is defined as the params of
        :meth:`compile`.
    :return: a list of :class:`.AsyncExec` objects as the functions
        corresponding to each part
    """
    self._clear_output_spec()
    updaters = []
    for inputs, outspec in io_specs:
        updaters.append(self._process_output_spec(inputs, outspec))
        self._add_multi_part_endpoint()
    funcs = self._do_compile_multi_part()
    for i, j in zip(funcs, updaters):
        j(i)
    return funcs

def make_shared(self, comp_node, *, dtype=None,
                shape=None, value=None, name=None, volatile=False):
    """make a shared value belonging to this comp graph; see
        :func:`.make_shared`"""
    from . import make_shared
    return make_shared(comp_node, dtype=dtype, shape=shape, value=value,
                       comp_graph=self, name=name, volatile=volatile)

def make_immutable(self, comp_node, value, *, dtype=None, name=None):
    """make an immutable value belonging to this comp graph; see
        :func:`.make_immutable`"""
    from . import make_immutable
    return make_immutable(comp_node, self, value, dtype=dtype, name=name)

def make_arg(self, comp_node, *, dtype=np.float32, shape=None, name=None,
             value=None):
    """make a runtime argument belonging to this comp graph; see
        :func:`.make_arg`"""
    from . import make_arg
    return make_arg(comp_node, self, dtype=dtype, shape=shape, name=name,
                    value=value)

def set_option(self, name, val):
    """set comp graph option; see :func:`.set_comp_graph_option`"""
    from .config import set_comp_graph_option
    return set_comp_graph_option(self, name, val)

def is_eager(self):
    """return True if comp_graph is in eager mode"""
    from .config import comp_graph_is_eager
    return comp_graph_is_eager(self)

def release(self):
    """explicitly release the underlying computing graph storage; this is
    mostly useful in eager evaluation mode, since doing so would release the
    underlying device storage

    :return: original reference count before release
    :rtype: int
    """
    return int(self._release())

}
