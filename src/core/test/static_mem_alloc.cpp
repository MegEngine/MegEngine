/**
 * \file src/core/test/static_mem_alloc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/utils/arith_helper.h"
#include "megbrain/utils/timer.h"
#include "../impl/graph/var_node_mem_mgr/static_mem_alloc.h"

#include <random>

using namespace mgb;
using namespace cg;

#ifdef WIN32
#pragma message "static_mem_alloc disabled because it causes the program to crash at startup"
#else

#define ITER_ALGO(cb) \
    cb(INTERVAL_MOVE) \
    cb(BEST_FIT) \
    cb(PUSHDOWN)

namespace {

struct TestParam {
    using Algo = StaticMemAlloc::AllocatorAlgo;

    Algo algo;
    size_t align, padding, nr_rand_opr, rng_seed;

    static decltype(auto) make_values(
            const std::vector<size_t> &aligns,
            const std::vector<size_t> &paddings,
            const std::vector<size_t> &nr_rand_opr) {
        std::vector<TestParam> data;
        std::mt19937_64 rng(next_rand_seed());
        //std::mt19937_64 rng(0);

        for (auto nr: nr_rand_opr) {
            size_t seed = rng();
            for (auto align: aligns) {
                for (auto padding: paddings) {
#define itcb(algo) data.push_back({Algo::algo, align, padding, nr, seed});
                    ITER_ALGO(itcb)
#undef itcb
                }
            }
        }
        return ::testing::ValuesIn(data);
    }
};

std::ostream& operator << (std::ostream &ostr, const TestParam &p) {
    std::string algo;
#define itcb(a) \
    do { \
        if (p.algo == StaticMemAlloc::AllocatorAlgo::a)  \
            algo = #a; \
    } while(0);
    ITER_ALGO(itcb);
#undef itcb

    ostr << "algo=" << algo << " align=" << p.align << " padding=" << p.padding;
    if (p.nr_rand_opr != 1)
        ostr << " nr_rand_opr=" << p.nr_rand_opr << " rng_seed=" << p.rng_seed;
    return ostr;
}

class BasicCorrectness: public ::testing::TestWithParam<TestParam> {
    protected:
        std::unique_ptr<cg::StaticMemAlloc> m_allocator;

        size_t padding() const {
            return GetParam().padding;
        }

        size_t align(size_t addr) const {
            return get_aligned_power2(addr, GetParam().align);
        }

    public:

        void SetUp() override {
            m_allocator = StaticMemAlloc::make(GetParam().algo);
            m_allocator->alignment(GetParam().align);
            m_allocator->padding(GetParam().padding);
        }
};

class RandomOpr: public BasicCorrectness {
};

decltype(auto) makeuk(int v) {
    return reinterpret_cast<cg::StaticMemAlloc::UserKeyType>(v);
}

} // anonymous namespace

TEST_P(BasicCorrectness, Alloc) {
    cg::StaticMemAlloc *allocator = this->m_allocator.get();
    allocator->add(0, 1, 1, makeuk(0));
    allocator->add(0, 1, 1, makeuk(1));
    allocator->add(1, 2, 2, makeuk(2));
    allocator->solve();
    ASSERT_EQ(std::max(align(2 + padding()), 2 * align(1 + padding())),
            allocator->tot_alloc());
    ASSERT_EQ(std::max(align(2 + padding()), 2 * align(1 + padding())),
            allocator->tot_alloc_lower_bound());
}

TEST_P(BasicCorrectness, Overwrite) {
    cg::StaticMemAlloc *allocator = this->m_allocator.get();
    auto id0 = allocator->add(0, 2, 3, makeuk(0));
    auto id1 = allocator->add(1, 3, 1, makeuk(1));
    auto id2 = allocator->add(2, 4, 1, makeuk(2));
    allocator->add_overwrite_spec(id1, id0, 1);
    allocator->add_overwrite_spec(id2, id1, 0);
    allocator->solve();

    ASSERT_EQ(align(3 + padding()), allocator->tot_alloc());
    ASSERT_EQ(align(3 + padding()), allocator->tot_alloc_lower_bound());
}

TEST_P(BasicCorrectness, OverwriteSameEnd) {
    cg::StaticMemAlloc *allocator = this->m_allocator.get();
    auto id1 = allocator->add(1, 2, 1, makeuk(1));
    auto id0 = allocator->add(0, 2, 1, makeuk(0));
    allocator->add_overwrite_spec(id1, id0, 0);
    allocator->solve();

    ASSERT_EQ(align(1 + padding()), allocator->tot_alloc());
    ASSERT_EQ(align(1 + padding()), allocator->tot_alloc_lower_bound());
}

INSTANTIATE_TEST_CASE_P(TestStaticMemAllocAlgo,
        BasicCorrectness, TestParam::make_values({1, 2}, {1, 2}, {1}));


#ifdef  __OPTIMIZE__
constexpr size_t INTERVAL_MOVE_MAX_SIZE = 600;
#else
constexpr size_t INTERVAL_MOVE_MAX_SIZE = 400;
#endif

TEST_P(RandomOpr, Main) {
    cg::StaticMemAlloc *allocator = this->m_allocator.get();
    auto &&param = this->GetParam();
    std::mt19937_64 rng(param.rng_seed);

    if (param.algo == TestParam::Algo::INTERVAL_MOVE &&
            param.nr_rand_opr > INTERVAL_MOVE_MAX_SIZE)
        return;

    constexpr size_t MAX_SIZE = 4096;

    // [0, 1)
    auto uniform = [&]() {
        return rng() / (std::mt19937_64::max() + 1.0);
    };

    // int [lo, hi)
    auto uniform_i = [&](size_t lo, size_t hi = 0) -> size_t {
        if (!hi) {
            hi = lo;
            lo = 0;
        }
        mgb_assert(lo <= hi);
        return (hi - lo) * uniform() + lo;
    };

    // begin, end, size, id
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> reqs;

    // indices in reqs that overwrite others
    std::vector<size_t> overwrite_src_idx;

    for (size_t i = 0; i < param.nr_rand_opr; ++ i) {

        bool overwrite = false;
        size_t begin, ov_dest, ov_offset, size;
        if (!reqs.empty() && uniform() <= 0.2) {
            size_t idx;
            if (!overwrite_src_idx.empty() && uniform() <= 0.5)
                idx = overwrite_src_idx[uniform_i(overwrite_src_idx.size())];
            else
                idx = uniform_i(0, reqs.size());
            begin = std::get<1>(reqs[idx]);
            if (begin) {
                -- begin;
                auto tot_sz = std::get<2>(reqs[idx]);
                if (tot_sz >= 2) {
                    ov_dest = std::get<3>(reqs[idx]);
                    ov_offset = uniform_i(tot_sz);
                    size = uniform_i(1, tot_sz - ov_offset);
                    overwrite = true;
                }
            }
        }
        if (!overwrite) {
            begin = uniform_i(param.nr_rand_opr);
            size = uniform_i(1, MAX_SIZE);
        }
        auto end = begin + uniform_i(1, param.nr_rand_opr),
             id = allocator->add(begin, end, size, makeuk(i));
        reqs.emplace_back(begin, end, size, id);
        if (overwrite) {
            allocator->add_overwrite_spec(id, ov_dest, ov_offset);
            overwrite_src_idx.push_back(reqs.size() - 1);
        }
    }

    RealTimer timer;
    allocator->solve();
    std::ostringstream ostr;
    ostr << param;
    auto sz_tot = allocator->tot_alloc(),
         sz_lower = allocator->tot_alloc_lower_bound();
    mgb_log("%s: time=%.3f size=%zu/%zu cost=%.3f", ostr.str().c_str(),
            timer.get_secs(), sz_tot, sz_lower, double(sz_tot) / sz_lower - 1);

}

INSTANTIATE_TEST_CASE_P(TestStaticMemAllocAlgo,
        RandomOpr, TestParam::make_values({1, 256}, {1, 32}, {
            10, INTERVAL_MOVE_MAX_SIZE, 1000, 10000}));

TEST(TestStaticMemAllocAlgo, PushdownChain) {
    auto allocator = StaticMemAlloc::make(
            StaticMemAlloc::AllocatorAlgo::PUSHDOWN);
    constexpr size_t NR = 5;
    for (size_t i = 0; i < NR; ++ i)
        allocator->add(i, i + 2, i + 1, makeuk(i));
    allocator->solve();
    ASSERT_EQ(NR + NR - 1, allocator->tot_alloc_lower_bound());
    ASSERT_EQ(NR + NR - 1, allocator->tot_alloc());
}

#endif // WIN32

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

