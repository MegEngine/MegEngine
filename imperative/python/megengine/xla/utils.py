import itertools as it
from functools import cached_property
from typing import cast


# [[a, b], [c], [d, e, f]] -> [a, b, c, d, e, f]
def flatten_list(xs):
    return list(it.chain.from_iterable(xs))


# [a, b, c, d, e, f], [2, 1, 3] -> [[a, b], [c], [d, e, f]]
def unflatten_list(xs, ns):
    xs_iter = iter(xs)
    unflattened = [[next(xs_iter) for _ in range(n)] for n in ns]
    _unflatten_done = object()
    assert next(xs_iter, _unflatten_done) is _unflatten_done
    return unflattened


# [a, b, c, d, e, f], [2, 1, 3] -> [a, b], [c], [d, e, f]
def split_list(args, ns):
    args = list(args)
    lists = []
    for n in ns:
        lists.append(args[:n])
        args = args[n:]
    lists.append(args)
    return lists


# zip with args length check
def safe_zip(*args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(zip(*args))


# ((x, a), (y, b), (z, c)) -> (x, y, z), (a, b, c)
def unzip2(xys):
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


def unzip3(xyzs):
    xs = []
    ys = []
    zs = []
    for x, y, z in xyzs:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return tuple(xs), tuple(ys), tuple(zs)


def _unwrap_func(f):
    if isinstance(f, property):
        return cast(property, f).fget
    elif isinstance(f, cached_property):
        return f.func
    return f


def use_cpp_class(cpp_cls):
    def wrapper(cls):
        exclude_methods = {"__module__", "__dict__", "__doc__"}

        for attr_name, attr in cls.__dict__.items():
            if attr_name not in exclude_methods and not hasattr(
                _unwrap_func(attr), "_use_cpp"
            ):
                setattr(cpp_cls, attr_name, attr)

        cpp_cls.__doc__ = cls.__doc__

        return cpp_cls

    return wrapper


def use_cpp_method(f):
    original_func = _unwrap_func(f)
    original_func._use_cpp = True
    return f


# (a, b, c), 1, x -> (a, x, b, c)
def tuple_insert(t, idx, val):
    assert 0 <= idx <= len(t), (idx, len(t))
    return t[:idx] + (val,) + t[idx:]
