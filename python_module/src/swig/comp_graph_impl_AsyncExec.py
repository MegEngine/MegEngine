%pythoncode {

_var2output_saver = None
"""map from var id to corresponding output saver; setup by
:meth:`.CompGraph.compile`"""

_expand_single_output = False
"""whether output contains only a single element and it should not be wrapped
by a list; setup by :meth:`.CompGraph.compile`"""

__output_savers = None
"""list of (symvar, output saver)"""

__inputs = None
"""list of (symvar, _HostSharedND.make_proxy(symvar)) pairs"""

__allow_args_input = None

__warned_unused_keys = None

__auto_wait_enabled = True

callbefore_func = None

callback_func = None

def __normalize_var_list(self, vlist):
    if len(vlist) == 1:
        vlist, = vlist
    if isinstance(vlist, SymbolVar):
        return [vlist]
    ret = []
    for i in vlist:
        assert isinstance(i, SymbolVar)
        ret.append(i)
    return ret

def _setup_args(self, args, kwargs):
    if kwargs:
        assert not args, 'should not provide both args and kwargs'
        for symvar, hsv in self.__inputs:
            val = kwargs.pop(symvar.name, None)
            assert val is not None, (
                'missing input at runtime: {}'.format(symvar))
            hsv.set_value(val, borrow=self.__auto_wait_enabled)

        if kwargs:
            keys = set(kwargs.keys())
            if keys != self.__warned_unused_keys:
                from .logconf import get_logger
                logger = get_logger()
                logger.warning(
                    'extra kwargs provided for megbrain AsyncExec: {}'.format(
                        keys))
                self.__warned_unused_keys = keys
        return

    assert not args or self.__allow_args_input, (
        'pass non-keyword args to function compiled without'
        ' inputs spec')
    assert len(args) == len(self.__inputs), (
        'inputs do not match: args={} needed={}'.format(
            args, [i[0] for i in self.__inputs]))
    for (symvar, hsv), val in zip(self.__inputs, args):
        hsv.set_value(val, borrow=self.__auto_wait_enabled)

def enable_borrow_on_cpu(self, flag=True):
    """whether to allow borrow input tensor memory on CPU; if set to True, then
    the user should ensure that memory buffers of input tensors are unchanged.

    This is set to False by default.
    """
    for _, i in self.__inputs:
        i.enable_borrow_on_cpu(flag)

def __call__(self, *args, **kwargs):
    """Execute the function; either one of positional arguments or keyword
    arguments must be given. Set :attr:`inputs` to change the order of
    positional arguments. The keys in keyword arguments are the names of input
    symvars

    :return: if auto wait is disabled, the return value would be
        :class:`FuncOutputSaver` objects corresponding to the vars marked by
        :class:`copy_output`; if auto wait is enabled, the numerical values as
        :class:`numpy.ndarray` would be returned.
    """
    if self.callbefore_func:
        if not callable(self.callbefore_func):
            raise TypeError(
                    "callbefore func must be callable: {}".format(self.callbefore_func))
        self.callbefore_func()
    self._setup_args(args, kwargs)
    self._execute()
    if self.callback_func:
        if not callable(self.callback_func):
            raise TypeError(
                    "callback func must be callable: {}".format(self.callback_func))
        self.callback_func()
    if self.__auto_wait_enabled:
        self.wait()

    if not self.__output_savers:
        return
    ret = []
    if self.__auto_wait_enabled:
        for _, i in self.__output_savers:
            ret.append(i.get())
    else:
        for _, i in self.__output_savers:
            ret.append(i)
    if self._expand_single_output:
        ret, = ret
    return ret

def wait(self):
    """wait for previous async exec to finish; wait is needed (i.e. the
    function runs in async mode) only when there is no output callback (i.e.
    all outputs are given by dest symvar only), or :meth:`disable_auto_wait` is
    called explicitly.

    :return: self"""
    self._wait()
    return self

@property
def prev_exec_time(self):
    """previous execution time in seconds"""
    return self._get_prev_exec_time()

@property
def inputs(self):
    """get input vars needed at runtime, in the order as the values that should
    be passed to :meth:`__call__`

    :setter: Set the order of input vars, which must be created by
        :func:`.make_arg`. None could also given, and in such case only keyword
        arguments would be allowed for :meth:`__call__`

    :type: tuple of :class:`.SymbolVar`
    """
    return tuple(i[0] for i in self.__inputs)

@inputs.setter
def inputs(self, *inputs):
    if self.__inputs is None:
        needed = tuple(self._find_mutable_input())
        used_names = set()
        for i in needed:
            assert i.name not in used_names, (
                'duplicated input name: {}'.format(i.name))
            used_names.add(i.name)
        self.__inputs = [(i, _HostSharedND.make_proxy(i)) for i in needed]

    if len(inputs) == 1 and inputs[0] is None:
        self.__allow_args_input = False
        return

    inputs = self.__normalize_var_list(inputs)
    inpvar2proxy = dict(self.__inputs)
    self.__allow_args_input = True
    reordered = []
    for i in inputs:
        proxy = inpvar2proxy.pop(i, None)
        if proxy is None:
            raise TypeError('extra input var provided: {}; needed: {}'.format(
                i, self.inputs))
        reordered.append((i, proxy))

    assert not inpvar2proxy, 'inputs not provided: {}'.format(
        list(inpvar2proxy.keys()))

    self.__inputs = reordered

@property
def available_outputs(self):
    """get output vars that could be used to set :attr:`outputs`. The order may
    be unstable

    :type: tuple of :class:`.SymbolVar`"""
    return tuple(self._var2output_saver.keys())

@property
def outputs(self):
    """get output vars whose corresponding values would be returned by
    :meth:`__call__`

    :setter: set the order of output vars to be returned. Duplicated vars could
        be included, but all the vars must have been provided to
        :meth`.CompGraph.compile`.

    :type: tuple of :class:`.SymbolVar`"""
    if not self.__output_savers:
        return

    if self._expand_single_output:
        (var, saver), = self.__output_savers
        return var
    return tuple(var for var, saver in self.__output_savers)

@outputs.setter
def outputs(self, *outputs):
    olist = []
    for var in self.__normalize_var_list(outputs):
        saver = self._var2output_saver.get(var)
        assert saver is not None, 'var {} is not set to be output var'.format(
            var)
        olist.append((var, saver))
    self.__output_savers = olist

def dump(self):
    """dump internal graph and execution sequence as
        json-serializable object"""
    return json.loads(self._to_json_str())

def disable_auto_wait(self):
    """if there is output callback function, then by default when
    :meth:`__call__` is invoked, it would not return until all computation is
    finished. This behavior can be changed by disabling auto wait, so the
    function returns as early as possible."""
    self.__auto_wait_enabled = False

def update_static_alloc_plan_and_get_size(self):
    """update static memory allocation plan without actual allocation

    :return: a dict that maps from comp node to size of allocation in bytes
    """
    return {k: v for k, v in self._update_static_alloc_plan_and_get_size()}

}
