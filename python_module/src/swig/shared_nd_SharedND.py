%pythoncode{

__lazy_initializer = None

def __apply_lazy_initializer(self):
    if self.__lazy_initializer is not None:
        self.set_value(self.__lazy_initializer.get_value())
        # __lazy_initializer released by self.set_value()

@property
def shape(self):
    """get shape of unerlying data"""
    if self.__lazy_initializer is not None:
        val = self.__lazy_initializer.get_shape()
    else:
        val = self._get_shape()
    return tuple(map(int, val))

@property
def comp_node(self):
    return self._get_comp_node()

@property
def dtype(self):
    return self._get_dtype()

@property
def lazy_initializer(self):
    """object to specify how to initialize this SharedND, or None
    if not set

    Please not that the initializer could be called at any time.

    :type: :class:`.SharedNDLazyInitializer`
    """
    return self.__lazy_initializer

@lazy_initializer.setter
def lazy_initializer(self, init):
    assert not len(self._get_shape()), (
        'can not set initializer for initialized SharedND')
    assert isinstance(init, SharedNDLazyInitializer)
    self.__lazy_initializer = init

def set_value(self, w, *, sync=True, inplace=False, share=False):
    """set value from a numpy array or from outputs in callback

    .. warning::

        If sync is false, a reference to input is kept and the caller is
        responsible to ensure that the input would not be modified after
        this function returns.

    :param w: value to be set
    :type w: :class:`numpy.ndarray`-compatible, :class:`SharedND` or
        :class:`.CompGraphCallbackValueProxy`
    :param sync: whether to sync device before returns
    :type sync: bool
    :param inplace: whether to copy in-place from another :class:`.SharedND`,
        guaranteed no memory allocating; if True, this SharedND must have the
        same shape as *w*, and buffer for this :class:`SharedND` would not be
        re-allocated.
    :param share: directly share the buffer in a
        :class:`.CompGraphCallbackValueProxy` with zero copy
    :return: self
    """

    if self is w:
        return self

    if share:
        assert isinstance(w, CompGraphCallbackValueProxy)
        self._share_from_value_proxy(w)
        return self

    self._set_copy_sync(sync)
    if isinstance(w, CompGraphCallbackValueProxy):
        self._copy_from_value_proxy(w)
        return self

    if isinstance(w, SharedND):
        w.__apply_lazy_initializer()
        ax_type = -2
        if inplace:
            ax_type = -3
        self.copy_from_shared_sub(w, ax_type, -1, -1, -1)
        return self
    assert not inplace, 'inplace only implemented for copying from SharedND'

    if self.__lazy_initializer is not None:
        del self.__lazy_initializer
    self._copy_from_npyarr(w)
    return self

def get_value(self):
    """get value as numpy array
    :return: numpy array, or None if value is empty"""
    self.__apply_lazy_initializer()
    return self._get_npyarr()

def resize(self, *shape):
    """resize the SharedND to given shape and allocate memory, without
    initializing data; usually :meth:`pubapi_dev_tensor_ptr` is then called to
    get the buffer address and pass it to some other library

    :return: self
    """
    if len(shape) == 1 and isinstance(shape[0], collections.Iterable):
        shape = shape[0]
    self._resize(shape)
    return self

def reset_zero(self):
    """reset dev_tensor to zeros"""
    self._reset_zero()

def copy_to(self, dest):
    """copy value to numpy array

    :type dest: :class:`np.ndarray`
    :param dest: destination array to write value of this var to,
        which must match shape, have float32 dtype and be
        contiguous
    """
    check_cont_ndarray(dest)
    wflat = dest.reshape(-1)
    assert wflat.ctypes.data == dest.ctypes.data
    self._copy_to_flatten(wflat)
    return dest

@property
def dev_ptr(self):
    """this method is DEPRECATED; use :meth:`pubapi_dev_tensor_ptr` instead"""
    return self._pubapi_dev_tensor_ptr(0)

@property
def pubapi_dev_tensor_ptr(self):
    """get a pointer to the corresponding mgb::pubapi::DeviceTensor object

    :rtype: int
    :return: the address as an integer
    """
    return self._pubapi_dev_tensor_ptr(1)

def symvar(self, comp_graph, name=None, *, volatile=False):
    """convert to SymbolVar to be put into a computing graph

    :param volatile: whether shape/ptr is allowed to change
    """
    self.__apply_lazy_initializer()
    assert self.shape, "initial shape must be available"
    if name is None:
        name = ''
    return self._as_sym_var(comp_graph, name, volatile)

def __getstate__(self):
    state = self.__dict__.copy()
    del state['this']
    state['value'] = self.get_value()
    state['comp_node'] = self.comp_node
    state['dtype'] = self.dtype
    return state

def __setstate__(self, state):
    val = state.pop('value')
    dtype = state.pop('dtype', 'float32')
    snd = SharedND(state.pop('comp_node'), dtype)
    if val is not None:
        assert val.dtype == dtype
        snd.set_value(val)
    self.this = snd.this
    for k, v in state.items():
        self.__dict__[k] = v

def share_memory_from(self, rhs, offset):
    """
    share memory from another SharedND, self and rhs must be initialized
    :param rhs: another sharedND used to share memory
    :type rhs: :class:`SharedND`

    :param offset: offset in rhs sharedND
    :type offset: int
    """
    assert self != rhs
    self._share_memory_from(rhs, offset)

def reset_dev_tensor(self, rhs):
    """
    reset devive tensor to another SharedND, self and rhs must be initialized.
    :param rhs: another sharedND whose device tensor to be reset to.  
    :type rhs: :class:`SharedND`
    """
    assert self != rhs
    self._reset_dev_tensor(rhs)

}
