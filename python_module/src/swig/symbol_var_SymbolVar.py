%pythoncode {

__owner_graph = None
__owner_opr = None

@property
def owner_graph(self):
    """get the owner graph; note that a reference would be kept in this var"""
    if self.__owner_graph is None:
        self.__owner_graph = self._get_owner_graph()
    return self.__owner_graph

@property
def owner_opr(self):
    """get the owner opr; get owner graph explicitly so it can keep a reference
    to its owner graph"""
    if self.__owner_opr is None:
        self.__owner_opr = self._get_owner_opr()

    self.__owner_opr.owner_graph
    return self.__owner_opr

@property
def comp_node(self):
    return self._get_comp_node()

@property
def name(self):
    return self._get_name()

@property
def id(self):
    """an integer identifier for this var that is unique in the computing
    graph"""
    return int(self._get_id())

@property
def imm_shape(self):
    """shape as immediate number

    :type: tuple of int
    """
    return tuple(map(int, self._get_imm_shape()))

@property
def inferred_value(self):
    """get statically inferred value of this var, or None if
    inference failed

    :type: :class:`numpy.ndarray` or None"""
    return self._get_inferred_value()

@property
def valid(self):
    """whether this symvar is valid (i.e. has corresponding var node in
    graph)"""
    return self._is_valid()

@property
def volatile(self):
    """whether the shape is volatile"""
    return not self._is_shared_device_tensor()

@property
def dtype(self):
    """get underling data type
    :rtype: :class:`numpy.dtype`"""
    return self._get_dtype()

def __hash__(self):
    return hash((self.owner_graph, self.id))

def __eq__(self, rhs):
    return (isinstance(rhs, SymbolVar) and
            self.owner_graph == rhs.owner_graph and
            self.id == rhs.id)

def _binary_opr(self, mode, rhs):
    from .opr import elemwise
    return elemwise([self, rhs], mode=mode)

def _binary_opr_lhs(self, mode, lhs):
    from .opr import elemwise
    return elemwise([lhs, self], mode=mode)

def __add__(self, rhs):
    return self._binary_opr('ADD', rhs)
def __radd__(self, lhs):
    return self._binary_opr_lhs('ADD', lhs)

def __sub__(self, rhs):
    return self._binary_opr('SUB', rhs)
def __rsub__(self, lhs):
    return self._binary_opr_lhs('SUB', lhs)

def __mul__(self, rhs):
    return self._binary_opr('MUL', rhs)
def __rmul__(self, lhs):
    return self._binary_opr_lhs('MUL', lhs)

def __matmul__(self, rhs):
    from .opr import matrix_mul
    return matrix_mul(self, rhs)
def __rmatmul__(self, rhs):
    from .opr import matrix_mul
    return matrix_mul(rhs, self)

def __lshift__(self, rhs):
    return self._binary_opr('SHL', rhs)
def __rshift__(self, rhs):
    return self._binary_opr('SHR', rhs)

def __truediv__(self, rhs):
    return self._binary_opr('TRUE_DIV', rhs)
def __rtruediv__(self, lhs):
    return self._binary_opr_lhs('TRUE_DIV', lhs)

def __floordiv__(self, rhs):
    return self._binary_opr('FLOOR_DIV', rhs)
def __rfloordiv__(self, rhs):
    return self._binary_opr_lhs('FLOOR_DIV', rhs)

def __mod__(self, rhs):
    return self._binary_opr('MOD', rhs)
def __rmod__(self, rhs):
    return self._binary_opr_lhs('MOD', rhs)

def __pow__(self, rhs):
    return self._binary_opr('POW', rhs)
def __rpow__(self, lhs):
    return self._binary_opr_lhs('POW', lhs)

def __lt__(self, rhs):
    return self._binary_opr('LT', rhs)
def __gt__(self, lhs):
    return self._binary_opr_lhs('LT', lhs)

def __le__(self, rhs):
    return self._binary_opr('LEQ', rhs)
def __ge__(self, lhs):
    return self._binary_opr_lhs('LEQ', lhs)

def __neg__(self):
    from .opr import elemwise
    return elemwise([self], mode='NEGATE')

def __getitem__(self, idx):
    from .helper import cvt_getitem_to_idx_desc
    inpvar, desc = cvt_getitem_to_idx_desc(self, idx)
    if desc is None:
        return inpvar
    return _create_subtensor_like_opr('subtensor', [inpvar], desc, make_opr_config())
        
def reshape(self, *shp):
    from .opr import reshape
    return reshape(self, shp)

def broadcast(self, *shp):
    from .opr import broadcast
    return broadcast(self, shp)

def sum(self, axis=None, keepdims=False):
    from .opr import reduce_
    return reduce_(self, 'SUM', axis, keepdims)

def max(self, axis=None, keepdims=False):
    from .opr import reduce_
    return reduce_(self, 'MAX', axis, keepdims)

def min(self, axis=None, keepdims=False):
    from .opr import reduce_
    return reduce_(self, 'MIN', axis, keepdims)

def prod(self, axis=None, keepdims=False):
    from .opr import reduce_
    return reduce_(self, 'PRODUCT', axis, keepdims)

def mean(self, axis=None, keepdims=False):
    from .opr import mean
    return mean(self, axis, keepdims)

def dimshuffle(self, *pattern, **kwargs):
    from .opr import dimshuffle
    ndim = kwargs.pop('ndim', 0)
    assert not kwargs
    return dimshuffle(self, pattern=pattern, ndim=ndim)

def astype(self, target_dtype):
    """see :func:`typecvt`"""
    from .opr import typecvt
    return typecvt(self, target_dtype)

@property
def shape(self):
    from .opr import get_var_shape
    return get_var_shape(self)

def axis_shape(self, axis):
    assert axis >= 0
    from .opr import get_var_shape
    return get_var_shape(self, axis=axis)

@property
def eager_val(self):
    """get value in eager evaluation mode"""
    return self._eager_eval_get_value() if self.owner_graph.is_eager() else None


def __iter__(self):
    """add __iter__ to avoid implicit iteration by calling
    __getitem__"""
    raise NotImplementedError('SymbolVar var could not be itered')

def __repr__(self):
    return 'SymbolVar(id={},name={})'.format(self.id, self.name)

}
