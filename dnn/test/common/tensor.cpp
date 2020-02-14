/**
 * \file dnn/test/common/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/tensor.h"
#include "test/common/random_state.h"

#include <random>

using namespace megdnn;

void test::init_gaussian(SyncedTensor<dt_float32> &tensor,
        dt_float32 mean, dt_float32 stddev) {
    auto ptr = tensor.ptr_mutable_host();
    auto n = tensor.layout().span().dist_elem();
    auto &&gen = RandomState::generator();
    std::normal_distribution<dt_float32> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = dist(gen);
    }
}

std::shared_ptr<TensorND> test::make_tensor_h2d(
        Handle *handle, const TensorND &htensor) {
    auto span = htensor.layout.span();
    TensorND ret{nullptr, htensor.layout};
    uint8_t* mptr = static_cast<uint8_t*>(
            megdnn_malloc(handle, span.dist_byte()));
    megdnn_memcpy_H2D(handle,
            mptr, static_cast<uint8_t*>(htensor.raw_ptr) + span.low_byte,
            span.dist_byte());
    ret.raw_ptr = mptr + span.low_byte;
    auto deleter = [handle, mptr](TensorND *p) {
        megdnn_free(handle, mptr);
        delete p;
    };
    return {new TensorND(ret), deleter};
}

std::shared_ptr<TensorND> test::make_tensor_d2h(
        Handle *handle, const TensorND &dtensor) {
    auto span = dtensor.layout.span();
    TensorND ret{nullptr, dtensor.layout};
    auto mptr = new uint8_t[span.dist_byte()];
    ret.raw_ptr = mptr + span.low_byte;
    megdnn_memcpy_D2H(handle,
            mptr, static_cast<uint8_t*>(dtensor.raw_ptr) + span.low_byte,
            span.dist_byte());
    auto deleter = [mptr](TensorND *p) {
        delete []mptr;
        delete p;
    };
    return {new TensorND(ret), deleter};
}

std::vector<std::shared_ptr<TensorND>> test::load_tensors(const char *fpath) {
    FILE *fin = fopen(fpath, "rb");
    megdnn_assert(fin);
    std::vector<std::shared_ptr<TensorND>> ret;
    for (; ; ) {
        char dtype[128];
        size_t ndim;
        if (fscanf(fin, "%s %zu", dtype, &ndim) != 2)
            break;
        TensorLayout layout;
        do {
#define cb(_dt) if (!strcmp(DTypeTrait<dtype::_dt>::name, dtype)) { \
            layout.dtype = dtype::_dt(); \
            break; \
}
            MEGDNN_FOREACH_DTYPE_NAME(cb)
#undef cb
            char msg[256];
            sprintf(msg, "bad dtype on #%zu input: %s", ret.size(), dtype);
            ErrorHandler::on_megdnn_error(msg);
        } while(0);
        layout.ndim = ndim;
        for (size_t i = 0; i < ndim; ++ i) {
            auto nr = fscanf(fin, "%zu", &layout.shape[i]);
            megdnn_assert(nr == 1);
        }
        auto ch = fgetc(fin);
        megdnn_assert(ch == '\n');
        layout.init_contiguous_stride();

        auto size = layout.span().dist_byte();
        auto mptr = new uint8_t[size];
        auto nr = fread(mptr, 1, size, fin);
        auto deleter = [mptr](TensorND *p) {
            delete []mptr;
            delete p;
        };
        ret.emplace_back(new TensorND{mptr, layout}, deleter);
        megdnn_assert(nr == size);
    }
    fclose(fin);
    return ret;
}

// vim: syntax=cpp.doxygen
