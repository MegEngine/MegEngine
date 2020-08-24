/**
 * \file dnn/test/cuda/argsort.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"

#include "../src/cuda/argsort/opr_impl.h"

using namespace megdnn;
using namespace test;

namespace {
class ArgsortRNG final : public RNG {
    bool m_rev_order = false;
    DType m_dtype;

    template <typename T>
    void fill(T* ptr, int n) {
        if (m_rev_order) {
            for (int i = 0; i < n; ++i)
                ptr[i] = static_cast<T>(n / 2 - i);
        } else {
            for (int i = 0; i < n; ++i)
                ptr[i] = static_cast<T>(i - n / 2);
            COMPAT_RANDOM(ptr, ptr + n);
        }
    }

    void gen(const TensorND& tensor) override {
        auto n = tensor.layout.total_nr_elems();
        if (m_dtype == dtype::Float32{}) {
            fill(tensor.ptr<dt_float32>(), n);
        } else {
            megdnn_assert(m_dtype == dtype::Int32{});
            fill(tensor.ptr<dt_int32>(), n);
        }
    }

public:
    ArgsortRNG(DType dt) : m_dtype{dt} {}

    void set_rev_order(bool flag) { m_rev_order = flag; }
};
void run_forward_test(Handle* handle, DType dtype) {
    Checker<ArgsortForward> checker(handle);
    using Param = Argsort::Param;
    using Order = Param::Order;
    ArgsortRNG rng{dtype};
    checker.set_dtype(2, dtype::Int32());
    checker.set_dtype(0, dtype).set_rng(0, &rng);
    for (size_t i = 3; i < 10240; i *= 2) {
        Param param;

        param.order = Order::ASCENDING;
        checker.set_param(param).execs({{3, i + 1}, {}, {}});
        param.order = Order::DESCENDING;
        checker.set_param(param).execs({{3, i - 1}, {}, {}});
        checker.set_param(param).execs({{13, i + 3}, {}, {}});
    }
    {
        // reverse sort large array
        constexpr size_t N = 200003;
        rng.set_rev_order(true);
        Param param;
        param.order = Order::ASCENDING;
        checker.set_param(param).execs({{1, N}, {}, {}});
    }
}

void run_backward_test(Handle* handle, DType dtype) {
    class IdxRng final : public RNG {
        void gen(const TensorND& tensor) override {
            auto ptr = tensor.ptr<dt_int32>();
            auto m = tensor.layout[0], n = tensor.layout[1];
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    ptr[j] = j;
                }
                COMPAT_RANDOM(ptr, ptr + n);
                ptr += n;
            }
        }
    } rng;
    Checker<ArgsortBackward> checker(handle);
    checker.set_dtype(1, dtype::Int32()).set_rng(1, &rng);
    checker.set_dtype(0, dtype);
    checker.set_dtype(2, dtype);
    for (size_t i = 16; i < 4096; i *= 2) {
        checker.execs({{3, i}, {3, i}, {3, i}});
        checker.execs({{3, i + 3}, {3, i + 3}, {3, i + 3}});
        checker.execs({{3, i + 3}, {3, i + 3}, {3, i + 7}});
    }
}

}  // anonymous namespace

TEST_F(CUDA, ARGSORT_FORWARD_F32) {
    run_forward_test(handle_cuda(), dtype::Float32{});
}

TEST_F(CUDA, ARGSORT_FORWARD_I32) {
    run_forward_test(handle_cuda(), dtype::Int32{});
}

TEST_F(CUDA, ARGSORT_BACKWARD_F32) {
    run_backward_test(handle_cuda(), dtype::Float32{});
}

TEST_F(CUDA, ARGSORT_BACKWARD_I32) {
    run_backward_test(handle_cuda(), dtype::Int32{});
}

// vim: syntax=cpp.doxygen

