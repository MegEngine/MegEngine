/**
 * \file dnn/test/common/topk.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/topk.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs/general.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

namespace {
class EqualValueRng final : public RNG {
    std::mt19937_64 m_rng{23};

public:
    void gen(const TensorND& tensor) override {
        memset(tensor.raw_ptr, 0, tensor.layout.span().dist_byte());
        ASSERT_EQ(2u, tensor.layout.ndim);
        size_t m = tensor.layout[0], n = tensor.layout[1];
        for (size_t i = 0; i < m; ++i) {
            int pos0 = m_rng() % n, pos1;
            do {
                pos1 = m_rng() % n;
            } while (pos0 == pos1);

            pos0 += i * n;
            pos1 += i * n;

#define CASE(ev, dt)                             \
    case DTypeEnum::ev: {                        \
        auto p = tensor.ptr<dt>();               \
        p[pos0] = p[pos1] = static_cast<dt>(-1); \
        break;                                   \
    }

            switch (tensor.layout.dtype.enumv()) {
                CASE(Float32, float);
                CASE(Int32, int);
                MEGDNN_INC_FLOAT16(CASE(Float16, half_float::half));
                default:
                    megdnn_throw("bad dtype");
            }
        }
#undef CASE
    }
};
}  // namespace

template <typename Dtype>
void test::run_topk_test(Handle* handle) {
    Checker<TopK> checker{handle};
    using Mode = TopK::Param::Mode;

    bool tie_breaking_mode = false;
    Mode cur_mode;
    auto output_canonizer = [&](const CheckerHelper::TensorValueArray& arr) {
        if (cur_mode == Mode::KTH_ONLY) {
            return;
        }
        auto pinp = arr[0].ptr<typename DTypeTrait<Dtype>::ctype>();
        auto pval = arr[1].ptr<typename DTypeTrait<Dtype>::ctype>();
        auto pidx = arr.at(2).ptr<int>();
        size_t m = arr[1].layout[0], n = arr[1].layout[1];
        using idx_val = std::pair<int, typename DTypeTrait<Dtype>::ctype>;
        std::vector<idx_val> data(n);
        auto compare = [](const idx_val& it1, const idx_val& it2) {
            return (it1.second > it2.second);
        };
        for (size_t i = 0; i < m; ++i) {
            if (cur_mode == Mode::VALUE_IDX_NOSORT) {
                // sort output pairs to canonize
                for (size_t j = 0; j < n; ++j) {
                    data[j].first = pidx[i * n + j];
                    data[j].second = pval[i * n + j];
                }
                std::sort(data.begin(), data.end(), compare);
                for (size_t j = 0; j < n; ++j) {
                    pidx[i * n + j] = data[j].first;
                    pval[i * n + j] = data[j].second;
                }
            }
            if (tie_breaking_mode) {
                // check if indices are correct and mark all indices to be zero
                for (size_t j = 0; j < n; ++j) {
                    auto idx = pidx[i * n + j];
                    auto val = pval[i * n + j];
                    // + 0 can change the type, such as changing half to float
                    ASSERT_EQ(pinp[i * arr[0].layout[1] + idx] + 0, val + 0);
                    pidx[i * n + j] = 0;
                }
            }
        }
    };

    auto run = [&](int k, size_t m, size_t n, Mode mode, int lda = 0) {
        if (::testing::Test::HasFailure()) {
            return;
        }
        cur_mode = mode;
        checker.set_proxy(k);
        checker.set_param(mode);
        TensorLayout layout{{m, n}, Dtype{}};
        if (lda) {
            layout.stride[0] = lda;
        }

        checker.set_output_canonizer(output_canonizer);

        if (mode == Mode::KTH_ONLY) {
            checker.execl({layout, {}});
        } else {
            checker.execl({layout, {}, {}});
        }
        if (!checker.prev_succ()) {
            fprintf(stderr,
                    "topk failed for (%zu,%zu):%d mode=%d cont=%d tie=%d\n", m,
                    n, k, static_cast<int>(mode), !lda, tie_breaking_mode);
            return;
        }
    };

    std::unique_ptr<IIDRNG> rng0;
    std::unique_ptr<RNG> rngf16;
    std::unique_ptr<NoReplacementRNG> rng1;
    switch (DTypeTrait<Dtype>::enumv) {
        case DTypeEnum::Float32: {
            rng0 = std::make_unique<UniformFloatRNG>(-100.f, 100.f);
            rng1 = std::make_unique<NoReplacementRNG>(rng0.get());
            checker.set_rng(0, rng1.get());
            break;
        }
        case DTypeEnum::Int32: {
            rng0 = std::make_unique<UniformIntRNG>(INT_MIN, INT_MAX);
            rng1 = std::make_unique<NoReplacementRNG>(rng0.get());
            checker.set_rng(0, rng1.get());
            break;
        }
        case DTypeEnum::Float16: {
            rngf16 = std::make_unique<Float16PeriodicalRNG>();
            checker.set_rng(0, rngf16.get());
            break;
        }
        default: {
            megdnn_throw(
                    ssprintf("only float32,int32 and float16 supported for "
                             "cuda and opencl topk"));
        }
    }

    for (auto mode :
         {Mode::KTH_ONLY, Mode::VALUE_IDX_NOSORT, Mode::VALUE_IDX_SORTED}) {
        run(1, 1, 1, mode);
        run(-1, 1, 1, mode);
        run(1, 23, 1, mode);
        run(1, 23, 100, mode);
        run(-1, 23, 100, mode);
        run(5, 23, 100, mode);
        run(-7, 23, 100, mode);
        run(23, 3, 50001, mode);
        run(5, 123, 3, mode);         // equiv to sort
        run(-5, 123, 3, mode);        // equiv to rev sort
        run(5, 3, 1231, mode, 2000);  // non contig

//! opencl on armv7's CI does not support large batch.
//! but P30 and MI9 are ok. fix it in the future.
#if !defined(MEGDNN_ARMV7) && defined(MGB_CUDA)
        run(3, 70000, 5, mode, 10);  // non contig
#endif
    }

    // special case to check if tie-break is correct
    auto tie_rng = std::make_unique<EqualValueRng>();
    tie_breaking_mode = true;
    checker.set_rng(0, tie_rng.get());
    for (auto mode : {Mode::VALUE_IDX_NOSORT, Mode::VALUE_IDX_SORTED}) {
        run(3, 1, 5, mode);
        run(3, 25, 4567, mode);
        run(8, 132, 10, mode);
    }
}
namespace megdnn {
namespace test {
#define INST(t) template void run_topk_test<t>(Handle*)

INST(dtype::Float32);
INST(dtype::Int32);
MEGDNN_INC_FLOAT16(INST(dtype::Float16));
#undef INST
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen
