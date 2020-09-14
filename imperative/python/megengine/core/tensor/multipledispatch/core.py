# Copyright (c) 2014 Matthew Rocklin
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of multipledispatch nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#
# --------------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
#  This file has been modified by Megvii ("Megvii Modifications").
#  All Megvii Modifications are Copyright (C) 2014-2020 Megvii Inc. All rights reserved.
# --------------------------------------------------------------------------------------

import inspect
import sys

from .dispatcher import Dispatcher, MethodDispatcher, ambiguity_warn

global_namespace = dict()


def dispatch(*types, **kwargs):
    """ Dispatch function on the types of the inputs

    Supports dispatch on all non-keyword arguments.

    Collects implementations based on the function name.  Ignores namespaces.

    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Examples
    --------

    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x):
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0

    Specify an isolated namespace with the namespace keyword argument

    >>> my_namespace = dict()
    >>> @dispatch(int, namespace=my_namespace)
    ... def foo(x):
    ...     return x + 1

    Dispatch on instance methods within classes

    >>> class MyClass(object):
    ...     @dispatch(list)
    ...     def __init__(self, data):
    ...         self.data = data
    ...     @dispatch(int)
    ...     def __init__(self, datum):
    ...         self.data = [datum]
    """
    namespace = kwargs.get("namespace", global_namespace)

    types = tuple(types)

    def _df(func):
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(
                name, MethodDispatcher(name),
            )
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types, func)
        return dispatcher

    return _df


def ismethod(func):
    """ Is func a method?

    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    if hasattr(inspect, "signature"):
        signature = inspect.signature(func)
        return signature.parameters.get("self", None) is not None
    else:
        if sys.version_info.major < 3:
            spec = inspect.getargspec(func)
        else:
            spec = inspect.getfullargspec(func)
        return spec and spec.args and spec.args[0] == "self"
