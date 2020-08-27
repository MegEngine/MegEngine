# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""This python module contains functions to apply the operators defined by
megbrain.

.. note::
    Most of the functions are automatically generated, and their signature have
    the form contain a ``param`` argument (or more than one arguments such as
    :func:`convolution` that has ``param`` and ``execution_polity``) and also
    accept keyword arguments. In such case, it can be called by either
    providing a param object of appropriate type, or by passing the arguments
    needed by the constructor of param object to the keyword arguments.
    Furthermore, for a param that needs an enumeration member, the enum name
    can be used to refer to the enum object.

    For example, the following statements are equivalent::

        elemwise([a, b], mode='max')
        elemwise([a, b], mode=opr_param_defs.Elemwise.Mode.MAX)
        elemwise([a, b], param=opr_param_defs.Elemwise('max'))
"""

__git_commit__ = "{%git_commit%}"

import collections

from . import helper
from .helper import PodOpVisitor
from . import param_defs
from ..._imperative_rt import OperatorNodeConfig as Config

__all__ = {%all%}

{%body%}
