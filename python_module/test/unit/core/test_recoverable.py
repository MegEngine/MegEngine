# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine as mge
import megengine._internal as mgb
from megengine.core import tensor
from megengine.test import assertTensorClose


def test_recoverable():
    a = tensor()
    b = tensor()
    a_np = np.random.random((4, 3)).astype("float32")
    b_np = np.random.random((3, 7)).astype("float32")
    a.set_value(a_np)
    b.set_value(b_np)

    # Do some normal computation.
    a2 = a * 2
    ab = a @ b

    # Raise a computation error.
    with pytest.raises(mgb.MegBrainError):
        _ = a * b

    # Variable a2 and ab should be still usable after error happened.
    assertTensorClose(a2.numpy(), a_np * 2)
    assertTensorClose(ab.numpy(), a_np @ b_np)

    # Should allow computation as well.
    ab2 = ab ** 2
    assertTensorClose(ab2.numpy(), (a_np @ b_np) ** 2)
