# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..core._imperative_rt.core2 import pop_scope, push_scope


class AutoNaming:
    r"""
    Name all executed operators automaticlly during tracing and record all tensors
    renamed by the user.
    """

    def __init__(self):
        self.scopes = []
        self.c_ops = []
        self.name2ops = {}
        self.handle2names = {}

    def clear(self):
        for var in vars(self).values():
            var.clear()

    def push_scope(self, scope):
        push_scope(scope)
        self.scopes.append(scope)

    def pop_scope(self):
        scope = self.scopes.pop()
        pop_scope(scope)

    def get_scope(self):
        return ".".join(self.scopes)

    def record_var_name(self, handle, name):
        self.handle2names[handle] = name

    def get_var_name(self, handle):
        return self.handle2names.pop(handle, None)

    def record_opnode(self, op):
        ops = self.name2ops.get(op.name, [])
        ops.append(op)
        self.name2ops[op.name] = ops

    def remove_duplicate_names(self):
        for key, ops in self.name2ops.items():
            if len(ops) == 1:
                continue
            for i, op in enumerate(ops):
                op.name = key + "[%s]" % str(i)
                if len(op.outputs) == 1:
                    continue
                for var in op.outputs:
                    var.name = var.name.replace(key, op.name)
        self.name2ops.clear()


auto_naming = AutoNaming()
