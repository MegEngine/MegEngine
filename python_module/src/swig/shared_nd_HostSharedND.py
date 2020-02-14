%pythoncode{

__dtype = None

def as_sym_var(self, cg, enable_static_infer, name=None):
    """get symvar to represent value of this HostSharedND in a
    computing graph

    :type cg: :class:`.CompGraph`
    :param cg: computing graph
    :type enable_static_infer: :class:`bool`
    :param enable_static_infer: whether to enable static value
        inference for this symvar; if set to True, the value must
        be set up before calling :meth:`as_sym_var`.
    """
    if name is None:
        name = ''
    return self._as_sym_var(cg, enable_static_infer, name)

def symvar(self, comp_graph, name=None, *, enable_static_infer=None):
    return self.as_sym_var(comp_graph, enable_static_infer, name)

def enable_borrow_on_cpu(self, flag):
    """whether to allow borrow memory in :meth:`set_value` if
    the underlying comp ndoe is on CPU"""
    self._enable_borrow_on_cpu(flag)

def _set_value_print_warn(
        self, reason, *,
        disabled=os.getenv('MGB_DISABLE_SET_VALUE_WARN') is not None):
    if disabled:
        return
    from .logconf import get_logger
    logger = get_logger()
    logger.warning('set {} from incompatible object is slow: {}'.format(
        self, reason))

def set_value(self, w, *, borrow=False):
    """set value to given numpy array

    :param borrow: if set to True, the memory of *w* may be
        borrowed, and *w* must remain unmodified during usage of
        this object
    :type borrow: bool
    :return: self
    """
    if self.__dtype is None:
        self.__dtype = self._get_dtype()

    if not isinstance(w, np.ndarray):
        wtype = type(w)
        w = np.ascontiguousarray(w, self.__dtype)
        if w.size >= 1024:
            self._set_value_print_warn(
                'not an ndarray object: {}'.format(wtype))
    elif w.size >= 1024:
        if w.dtype != self.__dtype:
            self._set_value_print_warn(
                'dtype mismatch: expect {}, get {}'.format(
                    self.__dtype, w.dtype))
        elif not w.flags['C_CONTIGUOUS']:
            self._set_value_print_warn('non-contiguous ndarray')

    self._copy_from_npyarr(w, borrow)
    return self

}
