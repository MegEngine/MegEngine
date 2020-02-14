# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from megengine import Tensor, tensor
from megengine.jit import trace
from megengine.test import assertTensorClose


def check_equal(np_tensor, mge_tensor):
    assertTensorClose(np_tensor, mge_tensor.numpy())


def test_index():
    a_shape = (10, 10, 10, 10, 10)
    a = np.random.random(a_shape).astype(dtype=np.float32)
    b = tensor(a)
    test_set = {}
    test_set["a"] = np.random.random(a_shape).astype(dtype=np.float32)
    test_set["b"] = tensor(test_set["a"])
    test_set["c"] = tensor(test_set["a"])

    def check_id_2(np_idx, mge_idx):
        # print('start :', mge_idx)
        def get_b(symbolic, *args):
            # print('get_b:', args)
            def get_func(inp):
                for i in mge_idx:
                    if isinstance(i, (list, Tensor)):
                        return inp.ai[mge_idx]
                return inp[mge_idx]

            func = trace(get_func, symbolic=symbolic)
            return func(*args)

        def set_b(symbolic, *args):
            # print('set_b:', args)
            def set_func(inp, val):
                for i in mge_idx:
                    if isinstance(i, (list, Tensor)):
                        return inp.set_ai(val)[mge_idx]
                return inp.set_subtensor(val)[mge_idx]

            func = trace(set_func, symbolic=symbolic)
            return func(*args)

        sub_a = a[np_idx]
        for symbolic in [True, False]:
            sub_b = get_b(symbolic, b)
            check_equal(sub_a, sub_b)
            # do not support set
            # print(mge_idx)
            if not mge_idx:
                continue
            go_flag = False
            for i in mge_idx:
                if i is np.newaxis:
                    go_flag = True
                    break
            if go_flag:
                continue
            if not symbolic:
                test_set["b"] = set_b(symbolic, test_set["b"], sub_b)
                check_equal(test_set["a"], test_set["b"])
            else:
                test_set["a"][np_idx] = sub_a
                test_set["c"] = set_b(symbolic, test_set["c"], sub_b)
                check_equal(test_set["a"], test_set["c"])

    def check_idx(*idx):
        check_id_2(idx, idx)

    def tensor_wrap(*idx):
        check_idx(*idx)
        tensor_idx = []
        numpy_idx = []
        for i in idx:
            numpy_idx.append(np.asarray(i).astype(np.int32))
            tensor_idx.append(tensor(numpy_idx[-1]))
        a_idx = tuple(numpy_idx)
        b_idx = tuple(tensor_idx)
        check_id_2(a_idx, b_idx)

    def test_one_dim():
        check_idx(-7)
        check_idx(-7, -7)
        check_idx(-7, -7, -7)
        check_idx(-7, -7, -7, -7)
        check_idx(-7, -7, -7, -7, -7)
        check_idx(7, 7, 7, 7, 7)
        check_idx(7, 7, 7, 7)
        check_idx(7, 7, 7)
        check_idx(7, 7)
        check_idx(7)
        check_idx()

    def test_slice():

        check_idx(slice(1, 7))
        check_idx(slice(1, 7))
        check_idx(slice(7, None))
        check_idx(slice(1, 7, 2))
        check_idx(slice(-7, 5))
        check_idx(slice(7, None, 7))
        check_idx(slice(None, 7, 7))

    def test_new_axis():
        check_idx(7, np.newaxis, 7)
        check_idx(7, np.newaxis, slice(3, 7))

    def test_ellipsis():
        check_idx(..., 7)

        check_idx(7, ...)

        check_idx(7, ..., 7, 7)

        check_idx(7, ..., 7, -7)

        check_idx(7, ..., slice(1, 7), -7)

    def test_integer_array():

        index = [[6, 7, 8], [9, 7, 4], [1, 1, 1], [3, 5, 6], [7, 8, 1]]

        tensor_wrap(index[0])
        tensor_wrap(index[0], index[1])
        tensor_wrap(index[0], index[1], index[2])
        tensor_wrap(index[0], index[1], index[2], index[3])
        tensor_wrap(index[0], index[1], index[2], index[3], index[4])

        # multi dimension
        index = [
            [6, 7, 8, 8, 9, 7],
            [9, 7, 4, 1, 8, 2],
            [1, 1, 1, 0, 3, 3],
            [3, 5, 6, 1, 6, 3],
            [7, 8, 1, 1, 8, 2],
        ]

        tensor_wrap(index[0])
        tensor_wrap(index[0], index[1])
        tensor_wrap(index[0], index[1], index[2])
        tensor_wrap(index[0], index[1], index[2], index[3])
        tensor_wrap(index[0], index[1], index[2], index[3], index[4])

        # braodcast
        # index = [
        #     [6, 7, 8, 8, 9, 7],  # 2 * 3
        #     [2],  # 1
        #     [1, 1, 1],  # 1 * 3
        #     [6, 2],  # 2 * 1
        #     [7, 8, 1, 1, 8, 2],  # 2 * 3
        # ]

        # tensor_wrap(index[0])
        # tensor_wrap(index[0], index[1])
        # tensor_wrap(index[0], index[1], index[2])
        # tensor_wrap(index[0], index[1], index[2], index[3])
        # tensor_wrap(index[0], index[1], index[2], index[3], index[4])

    def test_multi_dim():
        check_equal(a[7][7, 7, 7], b[7][7, 7, 7])
        check_equal(a[7, 7, 7][7], b[7, 7, 7][7])

        check_equal(a[7][7][7, 7], b[7][7][7, 7])
        check_equal(a[7][7, 7][7], b[7][7, 7][7])
        check_equal(a[7, 7][7][7], b[7, 7][7][7])

        check_equal(a[7, 7, 7][7, 7], b[7, 7, 7][7, 7])
        check_equal(a[7, 7][7, 7, 7], b[7, 7][7, 7, 7])

        check_equal(a[7][1:7:2], b[7][1:7:2])

        check_equal(a[7][7:], b[7][7:])

        check_equal(a[7][-7:-1], b[7][-7:-1])

        check_equal(a[7][-1:-7:-1], b[7][-1:-7:-1])

        check_equal(a[7:8][:], b[7:8][:])
        check_equal(a[7][:], b[7][:])

        check_equal(a[7][:][7], b[7][:][7])
        check_equal(a[:][7][7], b[:][7][7])

        check_equal(a[7][7], b[7][7])
        check_equal(a[7][7][7], b[7][7][7])
        check_equal(a[7][7][7][7], b[7][7][7][7])
        check_equal(a[7][7][7][7][7], b[7][7][7][7][7])

    def test_hard():
        check_idx(slice(None, None), [6, 7, 8], slice(0, 7), [6, 7, 8])
        check_idx(slice(None, None), slice(0, 7), [6, 7, 8], [6, 7, 8])
        # check_idx(slice(None, None), slice(0, 7), [[6], [7], [8]], [[6, 7, 8]])
        # check_idx(slice(None, None), [[6, 7, 8]], slice(1, 3), [[6], [7], [8]])
        # check_idx(slice(None, None), [[6, 7, 8]], [[6], [7], [8]], slice(1, 3))
        check_idx([6, 7, 8], [6, 7, 8], 7, slice(2, 7))
        # check_idx(Ellipsis, 1, [[[6]]], [[[0]]], slice(1, 4, 2))
        # check_idx(slice(2, 4, 2), 1, Ellipsis, 0, [[[7]], [[3]]])
        # check_idx(slice(7, 10, 2), 3, Ellipsis, [[[5, 0]]], [[[3, 1]]])
        # check_idx(slice(7, 9, 1), [[[4]]], 8, Ellipsis, [[[6]], [[9]]])

    def test_super_random():
        from random import randint
        from random import random as rand

        def true_or_false(ture_prob):
            return rand() < ture_prob

        def random_list(limit, size, one_base=False):
            if one_base:
                return [randint(1, limit) for _ in range(0, size)]
            else:
                # 0 <= x < limit
                return [randint(0, limit - 1) for _ in range(0, size)]

        def generate_random_int_matrix(limit, shape):
            if len(shape) == 0:
                return []
            if len(shape) == 1:
                return random_list(limit, shape[0])
            return [
                generate_random_int_matrix(limit, shape[1:]) for _ in range(0, shape[0])
            ]

        def generate_boardcast_shape(limit_shape):
            # new_len = randint(1, len(limit_shape))
            new_len = len(limit_shape)
            return [(1 if true_or_false(0.3) else i) for i in limit_shape[:new_len]]

        def g_slice(size):
            start = randint(0, size)
            if start == size:
                start = None
            end = randint(1 if start is None else start + 1, size + 1)
            if end == size + 1:
                end = None
            return slice(start, end, 1 if true_or_false(0.3) else 2)

        def g_int(size):
            return randint(0, size - 1)

        def g_inedx(limit_shape):
            new_len = randint(len(limit_shape) // 2, len(limit_shape))
            output = []
            # [5] -> (0 ~ 4)

            cur_dim, cur_new = len(limit_shape), 0
            use_int_array = False
            i = 0
            while len(output) < new_len:
                flag = rand()
                single_idx = None
                old_dim, old_new, old_use_int_array = cur_dim, cur_new, use_int_array
                if flag < 0.3:
                    single_idx = g_int(limit_shape[i])
                    cur_dim -= 1
                elif flag < 0.5:
                    single_idx = g_slice(limit_shape[i])
                elif flag < 0.9:
                    if not use_int_array:
                        board_cast_dim = random_list(10, 1, one_base=True)
                        cur_dim += len(board_cast_dim)
                        use_int_array = True
                    cur_dim -= 1
                    integer_array_shape = generate_boardcast_shape(board_cast_dim)
                    single_idx = generate_random_int_matrix(
                        limit_shape[i], integer_array_shape
                    )
                else:
                    cur_dim += 1
                    cur_new += 1
                    single_idx = np.newaxis
                # MAX_DIM  < 7
                if cur_dim > 7 or cur_new + len(limit_shape) > 7:
                    cur_dim, cur_new, use_int_array = (
                        old_dim,
                        old_new,
                        old_use_int_array,
                    )
                    continue
                if not single_idx is np.newaxis:
                    i += 1
                output.append(single_idx)
                # print('[cur_dim]: ', cur_dim, output)

            if cur_dim < 7 and rand() < 0.3 and new_len < len(limit_shape):
                output.insert(randint(0, len(output)), Ellipsis)

            return tuple(output)

        for i in range(0, 17):
            idx = g_inedx(a_shape)
            # print('[task {}] {}'.format(i, idx))
            check_idx(*idx)

    test_one_dim()
    test_multi_dim()
    test_slice()
    test_new_axis()
    test_ellipsis()
    test_integer_array()
    test_hard()
    test_super_random()
