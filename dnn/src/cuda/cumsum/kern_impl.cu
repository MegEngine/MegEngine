/**
 * \file dnn/src/cuda/cumsum/kern_impl.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "./kern_helper.cuh"
#include "./kern_impl.cuinl"

using namespace megdnn::cuda;
using namespace cumsum::detail::cubwrap;

namespace {

template <typename T>
struct FakeOp {
    __device__ T visit(int) { return 0; }
    __device__ static T apply(T, T) { return 0; }
};

template <bool reverse, typename T>
uint32_t get_workspace_elems_for_cub_1d_with_dtype_reverse(uint32_t nr_item) {
    typedef FakeOp<T> Op;
    Op op;
    InputIterator<T, Op, reverse> inp_iter(op, nr_item);
    OutputIterator<T, reverse> out_iter(NULL, nr_item);
    ScanOp<T, Op> scan_op;

    size_t wk_size0 = 0, wk_size1 = 0;
    cuda_check(cub::DeviceScan::ExclusiveScan(NULL, wk_size0, inp_iter,
                                              out_iter, scan_op, 0, nr_item));
    cuda_check(cub::DeviceScan::InclusiveScan(NULL, wk_size1, inp_iter,
                                              out_iter, scan_op, nr_item));
    return std::max(wk_size0, wk_size1);
}

template <typename T>
uint32_t get_workspace_elems_for_cub_1d_with_dtype(uint32_t nr_item) {
    return std::max(get_workspace_elems_for_cub_1d_with_dtype_reverse<false, T>(
                            nr_item),
                    get_workspace_elems_for_cub_1d_with_dtype_reverse<true, T>(
                            nr_item));
}

}  // namespace

uint32_t cumsum::get_workspace_bytes_for_cub_1d(uint32_t nr_item,
                                                uint32_t item_size) {
    switch (item_size) {
#define CASE(size, type) \
    case size:           \
        return get_workspace_elems_for_cub_1d_with_dtype<type>(nr_item)
        CASE(1, uint8_t);
        CASE(2, uint16_t);
        CASE(4, uint32_t);
        CASE(8, uint64_t);
#undef CASE
        default:
            report_error(megdnn_mangle("unsupported item size in cumsum"));
    }
}

uint32_t cumsum::get_workspace_in_bytes(uint32_t A, uint32_t B, uint32_t C,
                                        uint32_t item_size) {
    if (A == 1 && C == 1) {
        return get_workspace_bytes_for_cub_1d(B, item_size);
    }
    uint32_t BX, BY;
    get_BX_BY(A, B, C, BX, BY);
    uint32_t BY2 = BY * 2;
    uint32_t res = 0;
    while (B > BY2) {
        B = (B + BY2 - 1) / BY2;
        res += A * B * C;
    }
    return res * item_size;
}

void cumsum::get_BX_BY(uint32_t /* A */, uint32_t /* B */, uint32_t C,
                       uint32_t& BX, uint32_t& BY) {
    BX = 1;
    while (BX < C && BX * 2 <= 32)
        BX *= 2;
    BY = 512 / BX;
}

// vim: syntax=cpp.doxygen
