# -*- coding: utf-8 -*-
import numpy as np

import megengine.functional as F
from megengine import Tensor


def test_einsum():
    cases = [
        # ("ii->i", [(2, 2)]), # TODO: implement batch diag
        ("ij->ji", [(2, 3)]),
        ("ij->", [(2, 3)]),
        ("kj, ji", [(4, 5), (5, 3)]),
        ("...ij->...ji", [(2, 3, 4, 5)]),
        ("ij->j", [(2, 3)]),
        ("ik, k->i", [(2, 3), (3,)]),
        ("ik, kj->ij", [(2, 3), (3, 4)]),
        ("i, i->", [(2,), (2,)]),
        ("ij, ij->", [(2, 3), (2, 3)]),
        ("i, j->ij", [(2,), (3,)]),
        ("ijk, ikl->ijl", [(2, 3, 4), (2, 4, 5)]),
        ("pqrs, tuqvr->pstuv", [(2, 3, 4, 5), (6, 7, 3, 8, 4)]),
        ("ik, jkl, il->ij", [(2, 3), (4, 3, 5), (2, 5)]),
        ("...ij, ...jk->...ik", [(2, 3, 4, 5), (2, 3, 5, 6)]),
        ("...i->...", [(2, 3, 4, 5)]),
        ("...ik, ...kj", [(2, 3, 4), (2, 4, 5)]),
        ("...ik, ...kj", [(2, 3, 4, 5), (2, 3, 5, 6)]),
        ("i...k, k...j", [(2, 3, 4, 5), (5, 3, 4, 6)]),
        ("ik..., kj...", [(2, 3, 4, 5), (3, 6, 4, 5)]),
    ]
    for equation, shapes in cases:
        inputs = [*map(lambda x: np.random.randint(100, size=x).astype(np.float32), shapes)]
        np_out = np.einsum(equation, *inputs)
        mge_out = F.einsum(equation, *map(lambda x: Tensor(x), inputs)).numpy()
        np.testing.assert_equal(np_out, mge_out)
