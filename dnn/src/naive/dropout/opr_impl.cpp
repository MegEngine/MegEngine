/**
 * \file dnn/src/naive/dropout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/dropout/opr_impl.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;
using namespace std;
namespace {

using Param = megdnn::Dropout::Param;

dt_float32 get_random_number(uint64_t x) {
    union {
        uint32_t i;
        dt_float32 f;
    } u;
    u.i = (0x7F << 23) | (x >> 41);
    return 2 - u.f;
}

template <typename T>
void forward(
        T* inp, T* oup, void* raw_reserved, size_t len, Xoroshiro128plus& rng,
        float drop_prob) {
    uint8_t* reserved = reinterpret_cast<uint8_t*>(raw_reserved);
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < len; ++i) {
        float rn = get_random_number(rng());
        reserved[i] = rn < drop_prob ? 0 : 1;
        oup[i] = static_cast<T>(reserved[i] ? static_cast<float>(inp[i]) * scale : 0.f);
    }
}

template <typename T>
void backward(T* doup, T* dinp, void* raw_reserved, size_t len, float drop_prob) {
    uint8_t* reserved = reinterpret_cast<uint8_t*>(raw_reserved);
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < len; ++i) {
        dinp[i] =
                static_cast<T>(reserved[i] ? static_cast<float>(doup[i]) * scale : 0.f);
    }
}

}  // namespace

namespace megdnn {
namespace naive {

size_t DropoutForwardImpl::get_mask_size_in_bytes(const TensorLayout& inp) {
    return inp.total_nr_elems();
}

void DropoutForwardImpl::exec(
        _megdnn_tensor_in inp, _megdnn_tensor_out oup, _megdnn_tensor_out mask,
        _megdnn_workspace workspace) {
    check_exec(inp.layout, oup.layout, mask.layout, workspace.size);
    size_t length = inp.layout.total_nr_elems();
    uint64_t seed = param().seed;

    m_rng.ensure_seed(seed);

#define cb(DType)                                                          \
    if (inp.layout.dtype == DType()) {                                     \
        using T = typename DTypeTrait<DType>::ctype;                       \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward<T>(                           \
                inp.ptr<T>(), oup.ptr<T>(), mask.raw_ptr(), length, m_rng, \
                param().drop_prob));                                       \
        return;                                                            \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

void DropoutBackwardImpl::exec(
        _megdnn_tensor_in doup, _megdnn_tensor_in mask, _megdnn_tensor_out dinp,
        _megdnn_workspace workspace) {
    check_exec(doup.layout, mask.layout, dinp.layout, workspace.size);
    size_t length = doup.layout.total_nr_elems();

#define cb(DType)                                                     \
    if (doup.layout.dtype == DType()) {                               \
        using T = typename DTypeTrait<DType>::ctype;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward<T>(                     \
                doup.ptr<T>(), dinp.ptr<T>(), mask.raw_ptr(), length, \
                param().drop_prob));                                  \
        return;                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
