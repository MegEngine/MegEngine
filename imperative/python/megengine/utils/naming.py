from ..core._imperative_rt.core2 import pop_scope, push_scope


class AutoNaming:
    r"""Name all executed operators automaticlly during tracing and record all tensors
    renamed by the user.
    """

    scopes = []
    c_ops = []
    name2ops = {}
    handle2names = {}
    __cls_attributes__ = {"scopes", "c_ops", "name2ops", "handle2names"}

    @classmethod
    def clear(cls):
        for attr in cls.__cls_attributes__:
            getattr(cls, attr).clear()

    @classmethod
    def push_scope(cls, scope):
        if scope is not None:
            push_scope(scope)
        cls.scopes.append(scope)

    @classmethod
    def pop_scope(cls):
        scope = cls.scopes.pop()
        if scope is not None:
            pop_scope(scope)

    @classmethod
    def get_scope(cls):
        return ".".join(s for s in cls.scopes if s is not None)

    @classmethod
    def gen_name(cls, x) -> str:
        scope = cls.get_scope()
        name = x.c_name if x.c_name else x._name
        return scope + "." + name if len(scope) else name

    @classmethod
    def record_var_name(cls, handle, name):
        cls.handle2names[handle] = name

    @classmethod
    def get_var_name(cls, handle):
        return cls.handle2names.pop(handle, None)

    @classmethod
    def record_opnode(cls, op):
        ops = cls.name2ops.get(op.name, [])
        if op not in ops:
            ops.append(op)
        cls.name2ops[op.name] = ops

    @classmethod
    def remove_duplicate_names(cls):
        for key, ops in cls.name2ops.items():
            if len(ops) == 1:
                continue
            for i, op in enumerate(ops):
                op.name = key + "[%s]" % str(i)
                if len(op.outputs) == 1:
                    continue
                for var in op.outputs:
                    var.name = var.name.replace(key, op.name)
        cls.name2ops.clear()
