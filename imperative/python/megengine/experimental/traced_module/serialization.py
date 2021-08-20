# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Dict

from ...core._imperative_rt import OpDef
from ...core.ops import builtin
from ...version import __version__

OPDEF_PARAM_LOADER = {}


def get_opdef_state(obj: OpDef) -> Dict:
    state = obj.__getstate__()
    state["type"] = type(obj)
    state["version"] = __version__
    return state


def load_opdef_from_state(state: Dict) -> OpDef:
    assert "type" in state and issubclass(state["type"], OpDef)
    assert "version" in state
    opdef_type = state.pop("type")
    if opdef_type in OPDEF_PARAM_LOADER:
        loader = OPDEF_PARAM_LOADER[opdef_type]
        state = loader(state)
    state.pop("version")
    opdef_obj = opdef_type()
    opdef_obj.__setstate__(state)
    return opdef_obj
