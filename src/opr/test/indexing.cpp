/**
 * \file src/opr/test/indexing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/indexing.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"

#include "megdnn/tensor_iter.h"

#include <random>

using namespace mgb;

#ifdef WIN32
namespace {
template <class ForwardIterator, class T>
void iota(ForwardIterator first, ForwardIterator last, T value) {
    while (first != last) {
        *first++ = value;
        ++value;
    }
}
}  // namespace
#else
using std::iota;
#endif

namespace {

void gen_index_onehot(int* max_value, HostTensorND& dest) {
    mgb_assert(*max_value > 0);
    RNGxorshf rng{next_rand_seed()};
    std::uniform_int_distribution<int> dist{0, *max_value - 1};

    auto ptr = dest.ptr<float>();
    for (size_t i = 0, it = dest.layout().total_nr_elems(); i < it; ++i) {
        ptr[i] = dist(rng);
    }
}

void test_one_hot_get(int32_t axis, const TensorShapeArray& test_cases) {
    using Checker = AutoOprChecker<2, 1>;

    auto cvt_opr = megdnn_naive_handle()->create_operator<megdnn::TypeCvt>();
    auto nopr =
            megdnn_naive_handle()->create_operator<megdnn::IndexingOneHot>();
    nopr->param() = {axis};

    HostTensorND index_i;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::IndexingOneHot::make(inputs[0], inputs[1], {axis})};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&src = *inp[0], &&index = *inp[1];
        index_i.comp_node(index.comp_node())
                .dtype(dtype::Int32())
                .resize(index.shape());
        cvt_opr->exec(index.as_megdnn(), index_i.as_megdnn());
        TensorShape oshp(src.shape());
        oshp.shape[axis] = 1;
        dest[0].resize(oshp);
        nopr->exec(src.as_megdnn(), index_i.as_megdnn(), dest[0].as_megdnn(),
                   {});
    };

    Checker checker{make_graph, fwd};
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    int cur_max_index_val = -1;
    {
        using namespace std::placeholders;
        checker.set_input_generator(
                1, std::bind(gen_index_onehot, &cur_max_index_val, _1));
    }
    checker.set_input_allow_grad(1, false);
    for (auto&& i : test_cases) {
        TensorLayout index_layout{i, dtype::Float32()};
        index_layout.remove_axis_inplace(axis);
        cur_max_index_val = i.shape[axis];
        checker.run({i, index_layout}, opt);
    }
}

void test_one_hot_set(int32_t axis, const TensorShapeArray& test_cases) {
    using Checker = AutoOprChecker<3, 1>;

    auto cvt_opr = megdnn_naive_handle()->create_operator<megdnn::TypeCvt>();
    auto nopr =
            megdnn_naive_handle()->create_operator<megdnn::IndexingSetOneHot>();
    nopr->param() = {axis};

    HostTensorND index_i;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::IndexingSetOneHot::make(inputs[0], inputs[1], inputs[2],
                                             {axis})};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&index = *inp[1], &&sub = *inp[2];
        index_i.comp_node(index.comp_node())
                .dtype(dtype::Int32())
                .resize(index.shape());
        cvt_opr->exec(index.as_megdnn(), index_i.as_megdnn());
        dest[0].copy_from(data);
        nopr->exec(dest[0].as_megdnn(), index_i.as_megdnn(), sub.as_megdnn(),
                   {});
    };

    Checker checker{make_graph, fwd};
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    int cur_max_index_val = -1;
    {
        using namespace std::placeholders;
        checker.set_input_generator(
                1, std::bind(gen_index_onehot, &cur_max_index_val, _1));
    }
    checker.set_input_allow_grad(1, false);
    for (auto&& i : test_cases) {
        TensorLayout sub_layout{i, dtype::Float32()};
        sub_layout[axis] = 1;
        auto index_layout = sub_layout.remove_axis(axis);
        cur_max_index_val = i.shape[axis];
        checker.run({i, index_layout, sub_layout}, opt);
    }
    return;
}

void test_one_hot(int32_t axis, const TensorShapeArray& test_cases) {
    test_one_hot_get(axis, test_cases);
    test_one_hot_set(axis, test_cases);
}

}  // anonymous namespace

TEST(TestOprIndexing, OneHot2D) {
    TensorShapeArray cases = {{1, 1}, {2, 2}, {10, 8}, {8, 10}};
    test_one_hot(0, cases);
    test_one_hot(1, cases);
}

TEST(TestOprIndexing, OneHot3D) {
    TensorShapeArray cases = {{1, 1, 1}, {2, 2, 2}, {3, 2, 3}};
    for (size_t i = 0; i < 3; i++)
        test_one_hot(i, cases);
}

TEST(TestOprIndexing, OneHot4D) {
    TensorShapeArray cases = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 2, 3, 4}};
    for (size_t i = 0; i < 4; i++)
        test_one_hot(i, cases);
}

TEST(TestOprIndexing, OneHot5D) {
    TensorShapeArray cases = {
            {1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {3, 2, 3, 4, 5}};
    for (size_t i = 0; i < 5; i++)
        test_one_hot(i, cases);
}

TEST(TestOprIndexing, Remap) {
    using Checker = AutoOprChecker<2, 1>;

    TensorShape cur_inp_shp;
    std::mt19937 rng{static_cast<std::mt19937::result_type>(next_rand_seed())};
    auto gen_index = [&](HostTensorND& dest) {
        auto ptr = dest.ptr<float>();
        auto dshp = dest.shape();
        mgb_assert(dshp[dshp.ndim - 1] == cur_inp_shp.ndim);
        for (size_t i = 0, it = dshp.total_nr_elems() / cur_inp_shp.ndim;
             i < it; ++i) {
            for (size_t j = 0; j < cur_inp_shp.ndim; ++j)
                *(ptr++) = rng() % cur_inp_shp[j];
        }
    };
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::IndexingRemap::make(
                inputs[0], opr::TypeCvt::make(inputs[1], dtype::Int32()), {})};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto map_shp = inp[1]->shape();
        auto inp_ly = inp[0]->layout();
        auto out_shp = map_shp;
        --out_shp.ndim;
        dest[0].resize(out_shp);
        auto optr = dest[0].ptr<float>(), iptr = inp[0]->ptr<float>(),
             mptr = inp[1]->ptr<float>();
        for (size_t i = 0, it = out_shp.total_nr_elems(); i < it; ++i) {
            size_t offset = 0;
            for (size_t j = 0; j < inp_ly.ndim; ++j)
                offset += inp_ly.stride[j] * mptr[j];
            mptr += inp_ly.ndim;
            *(optr++) = iptr[offset];
        }
    };

    Checker checker{make_graph, fwd};
    checker.set_input_generator(1, gen_index);
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    checker.set_input_allow_grad(1, false);
    TensorShape ishp[3] = {{2, 3}, {8, 4, 2}, {1}},
                mshp[3] = {{5, 8, 2}, {3, 1, 3}, {8, 1}};
    for (int i = 0; i < 3; ++i) {
        cur_inp_shp = ishp[i];
        checker.run({cur_inp_shp, mshp[i]}, opt);
    }
}

TEST(TestOprIndexing, MultiAxisVecFwdOnly) {
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 8, 8});
    auto host_idx = std::make_shared<HostTensorND>(
            host_x->comp_node(), TensorShape{2}, dtype::Int32());
    auto graph = ComputingGraph::make();
    using AIdx = opr::indexing::AxisIndexer;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::Host2DeviceCopy::make(*graph, host_idx),
         y = opr::IndexingMultiAxisVec::make(
                 x, {AIdx::make_index(1, idx),
                     AIdx::make_interval(-1, x.make_scalar(2), x.make_scalar(5),
                                         None)});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    host_idx->ptr<int>()[0] = 3;
    host_idx->ptr<int>()[1] = -2;
    func->execute();
    ASSERT_EQ(TensorShape({5, 2, 3}), host_y.shape());

    auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                ASSERT_EQ(px[i * 64 + (j + 1) * 3 * 8 + k + 2], *(py++));
            }
        }
    }
}

namespace {
void mavi_gen_index(int* p_axis_size, HostTensorND& dest) {
    auto axis_size = *p_axis_size;
    mgb_assert(axis_size > 0);
    RNGxorshf rng{next_rand_seed()};
    std::uniform_int_distribution<int> dist{-axis_size, axis_size - 1};

    auto ptr = dest.ptr<float>();
    for (size_t i = 0, it = dest.layout().total_nr_elems(); i < it; ++i) {
        ptr[i] = dist(rng);
    }
}

void mavi_iter_data_value(
        HostTensorND& data, HostTensorND& value, HostTensorND& idx,
        const thin_function<void(float& data, float& value)>& callback) {
    auto pidx = idx.ptr<float>();
    auto value_iter = megdnn::tensor_iter<float>(value.as_megdnn()).begin();

    int data_idx[TensorLayout::MAX_NDIM];
    for (size_t i = 0, it = value.shape().total_nr_elems(); i < it; ++i) {
        std::copy(value_iter.idx(), value_iter.idx() + value.shape().ndim,
                  data_idx);
        data_idx[0] = data_idx[0] * 2 + 1;
        int& idx_last = data_idx[value.shape().ndim - 1];
        idx_last = pidx[idx_last];
        if (idx_last < 0)
            idx_last += data.shape(data.shape().ndim - 1);
        callback(*data.ptr<float>(data_idx, data_idx + value.shape().ndim),
                 *value_iter);
        ++value_iter;
    }
}

}  // anonymous namespace

TEST(TestOprIndexing, MultiAxisVec) {
    using Checker = AutoOprChecker<2, 1>;
    using AIdx = opr::indexing::AxisIndexer;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0],
                  idx = opr::TypeCvt::make(inputs[1], dtype::Int32());
        return {opr::IndexingMultiAxisVec::make(
                x, {AIdx::make_index(-1, idx),
                    AIdx::make_interval(0, x.make_scalar(1), x.make_scalar(-1),
                                        x.make_scalar(2))})};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx = *inp[1];
        auto dshp = data.shape();
        dshp[0] = (dshp[0] - 3) / 2 + 1;
        dshp[dshp.ndim - 1] = idx.shape(0);
        auto cb = [](float& data, float& value) { value = data; };
        mavi_iter_data_value(data, dest[0].resize(dshp), idx, cb);
    };

    Checker checker{make_graph, fwd};
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    int cur_axis_size = -1;
    {
        using namespace std::placeholders;
        checker.set_input_generator(
                1, std::bind(mavi_gen_index, &cur_axis_size, _1));
    }
    checker.set_input_allow_grad(1, false);

    cur_axis_size = 3;
    checker.run({TensorShape{3, 3}, {10}}, opt);
    cur_axis_size = 8;
    checker.run({TensorShape{7, 2, 8}, {15}}, opt);
    cur_axis_size = 9;
    checker.run({TensorShape{12, 1, 2, 9}, {23}}, opt);
}

TEST(TestOprIndexing, MultiAxisVecWithEmptyIndexDesc) {
    auto graph = ComputingGraph::make();
    auto host_x = HostTensorGenerator<>{}({2, 3});
    auto run_check = [&](SymbolVar y) {
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({2, 3}), host_y.shape());
        func->execute();
        MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
    };

    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    run_check(opr::IndexingMultiAxisVec::make(x, {}));
}

TEST(TestOprIndexing, IncrMultiAxisVec) {
    using Checker = AutoOprChecker<3, 1>;
    using AIdx = opr::indexing::AxisIndexer;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0],
                  idx = opr::TypeCvt::make(inputs[1], dtype::Int32()),
                  val = inputs[2];
        return {opr::IndexingIncrMultiAxisVec::make(
                x, val,
                {AIdx::make_index(-1, idx),
                 AIdx::make_interval(0, x.make_scalar(1), x.make_scalar(-1),
                                     x.make_scalar(2))})};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx = *inp[1];
        auto cb = [](float& data, float& value) { data += value; };
        mavi_iter_data_value(dest[0].copy_from(data), *inp[2], idx, cb);
    };

    Checker checker{make_graph, fwd};
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    int cur_axis_size = -1;
    {
        using namespace std::placeholders;
        checker.set_input_generator(
                1, std::bind(mavi_gen_index, &cur_axis_size, _1));
    }
    checker.set_input_allow_grad(1, false);

    cur_axis_size = 3;
    checker.run({TensorShape{3, 3}, {10}, {1, 10}}, opt);
    cur_axis_size = 8;
    checker.run({TensorShape{7, 2, 8}, {15}, {3, 2, 15}}, opt);
    cur_axis_size = 9;
    checker.run({TensorShape{12, 1, 2, 9}, {23}, {5, 1, 2, 23}}, opt);
}

TEST(TestOprIndexing, SetMultiAxisVec) {
    using Checker = AutoOprChecker<3, 1>;
    using AIdx = opr::indexing::AxisIndexer;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0],
                  idx = opr::TypeCvt::make(inputs[1], dtype::Int32()),
                  val = inputs[2];
        return {opr::IndexingSetMultiAxisVec::make(
                x, val,
                {AIdx::make_index(-1, idx),
                 AIdx::make_interval(0, x.make_scalar(1), x.make_scalar(-1),
                                     x.make_scalar(2))})};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx = *inp[1];
        auto cb = [](float& data, float& value) { data = value; };
        mavi_iter_data_value(dest[0].copy_from(data), *inp[2], idx, cb);
    };

    Checker checker{make_graph, fwd};
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    int cur_axis_size = -1;
    RNGxorshf rng{next_rand_seed()};
    auto gen_idx = [&cur_axis_size, &rng](HostTensorND& dest) {
        std::vector<size_t> cand(cur_axis_size);
        iota(cand.begin(), cand.end(), 0);
        std::shuffle(cand.begin(), cand.end(), rng);
        auto nr = dest.shape().total_nr_elems();
        mgb_assert(nr <= cand.size());
        auto ptr = dest.ptr<float>();
        for (size_t i = 0; i < nr; ++i) {
            ptr[i] = cand[i];
            if (rand() % 2) {
                ptr[i] -= cur_axis_size;
            }
        }
    };
    checker.set_input_allow_grad(1, false);
    checker.set_input_generator(1, gen_idx);

    cur_axis_size = 3;
    checker.run({TensorShape{3, 3}, {3}, {1, 3}}, opt);
    cur_axis_size = 23;
    checker.run({TensorShape{7, 2, 23}, {15}, {3, 2, 15}}, opt);
    cur_axis_size = 18;
    checker.run({TensorShape{12, 1, 2, 18}, {1}, {5, 1, 2, 1}}, opt);
}

TEST(TestOprIndexing, SetMultiAxisVecWithEmptyIndexDesc) {
    auto graph = ComputingGraph::make();
    auto host_x = HostTensorGenerator<>{}({2, 3}),
        host_y = HostTensorGenerator<>{}({2, 3});
    auto run_check = [&](SymbolVar z) {
        HostTensorND host_z;
        auto func = graph->compile({make_callback_copy(z, host_z)});
        // warning should be printed on the first execution
        func->execute();
        ASSERT_EQ(TensorShape({2, 3}), host_z.shape());
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_z, *host_y);
    };

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);

    run_check(opr::IndexingSetMultiAxisVec::make(x, y, {}));
}

TEST(TestOprIndexing, MultiAxisVecDegenerate) {
    auto graph = ComputingGraph::make();
    auto host_x = HostTensorGenerator<>{}({2, 3}),
         host_idx = HostTensorGenerator<dtype::Int32>{-2, 3}({1});
    auto run_check = [&](SymbolVar y) {
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        // warning should be printed on the first execution
        func->execute();
        ASSERT_EQ(TensorShape({2, 1}), host_y.shape());
        func->execute();
        auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
        for (int i = 0; i < 2; ++i) {
            ASSERT_EQ(px[i * 3 + 1], py[i]);
        }
    };

    host_idx->ptr<int>()[0] = -2;
    using MAV = opr::IndexingMultiAxisVec;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::Host2DeviceCopy::make(*graph, host_idx);

    run_check(MAV::make(x, {MAV::AxisIndexer::make_index(1, idx)}));
    run_check(MAV::make(
            x, {MAV::AxisIndexer::make_interval(1, idx, idx + 1, None)}));
}

TEST(TestOprIndexing, MultiAxisVecModifyDegenerate) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> genf32;
    auto host_x = genf32({2, 3}),
         host_idx = HostTensorGenerator<dtype::Int32>{-2, 3}({1}),
         host_mod = genf32({2, 1});
    auto run_check = [&](SymbolVar y, bool is_incr) {
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        // warning should be printed on the first execution
        func->execute();
        ASSERT_EQ(TensorShape({2, 3}), host_y.shape());
        func->execute();
        auto px = host_x->ptr<float>(), py = host_y.ptr<float>(),
             pmod = host_mod->ptr<float>();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                float expect;
                if (j == 1) {
                    expect = pmod[i];
                    if (is_incr)
                        expect += px[i * 3 + j];
                } else {
                    expect = px[i * 3 + j];
                }
                MGB_ASSERT_FLOAT_EQ(expect, py[i * 3 + j]);
            }
        }
    };

    host_idx->ptr<int>()[0] = -2;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::Host2DeviceCopy::make(*graph, host_idx),
         mod = opr::Host2DeviceCopy::make(*graph, host_mod);

    {
        using MAV = opr::IndexingSetMultiAxisVec;
        run_check(MAV::make(x, mod, {MAV::AxisIndexer::make_index(1, idx)}),
                  false);
        run_check(MAV::make(x, mod,
                            {MAV::AxisIndexer::make_interval(1, idx, idx + 1,
                                                             None)}),
                  false);
    }

    {
        using MAV = opr::IndexingIncrMultiAxisVec;
        run_check(MAV::make(x, mod, {MAV::AxisIndexer::make_index(1, idx)}),
                  true);
        run_check(MAV::make(x, mod,
                            {MAV::AxisIndexer::make_interval(1, idx, idx + 1,
                                                             None)}),
                  true);
    }
}

TEST(TestOprIndexing, ZeroSize) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2}), host_mask = gen({2});
    host_mask->ptr<float>()[0] = 2;
    host_mask->ptr<float>()[1] = -2;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         mask = opr::Host2DeviceCopy::make(*graph, host_mask),
         idx = opr::CondTake::make(x, mask, {opr::CondTake::Param::Mode::LT, 0})
                       .at(1),
         y = opr::IndexingMultiAxisVec::make(
                 x,
                 {opr::IndexingMultiAxisVec::AxisIndexer::make_index(0, idx)});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_EQ(host_y.shape(), TensorShape({1}));
    ASSERT_EQ(host_y.ptr<float>()[0], host_x->ptr<float>()[1]);
    host_mask->ptr<float>()[1] = 2;

    func->execute();
    ASSERT_EQ(host_y.shape(), TensorShape({0}));
}

#if MGB_ENABLE_EXCEPTION
namespace {

void check_async_error(cg::AsyncExecutable* func, int nr_error) {
    try {
        func->execute().wait();
    } catch (MegBrainError& exc) {
        auto info = static_cast<const cg::OperatorNodeExcExtraInfo*>(
                exc.extra_info());
        auto msg_expect = ssprintf("%d async err", nr_error);
        ASSERT_TRUE(!strncmp(exc.what(), msg_expect.c_str(), msg_expect.size()))
                << "bad exception message: " << exc.what()
                << "\nnr_error=" << nr_error;
        mgb_log("caught exception: %s opr=%s{%s}", exc.what(),
                info->opr()->cname(), info->opr()->dyn_typeinfo()->name);
        return;
    }
    ASSERT_TRUE(0) << "exception not thrown";
}

}  // anonymous namespace

TEST(TestOprIndexing, IndexingOneHotError) {
    REQUIRE_GPU(1);
    auto graph = ComputingGraph::make();
    auto cn = CompNode::load("gpux");
    auto host_x = HostTensorGenerator<>{}({5, 7}, cn),
         host_idx = HostTensorGenerator<dtype::Int32>{0, 6}({5}, cn);

    using Opr = opr::IndexingOneHot;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::Host2DeviceCopy::make(*graph, host_idx),
         y = Opr::make(x, idx, {1});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    // no error
    func->execute();

    // one error
    host_idx->ptr<int>()[1] = 8;
    check_async_error(func.get(), 1);

    // three errors
    host_idx->ptr<int>()[3] = -1;
    host_idx->ptr<int>()[4] = -10;
    check_async_error(func.get(), 3);
}

TEST(TestOprIndexing, MultiAxisVecError) {
    REQUIRE_GPU(1);
    auto graph = ComputingGraph::make();
    auto cn = CompNode::load("gpux");
    auto host_x = HostTensorGenerator<>{}({2, 3}, cn),
         host_idx = HostTensorGenerator<dtype::Int32>{-1, 1}({6}, cn);

    using MAV = opr::IndexingMultiAxisVec;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::Host2DeviceCopy::make(*graph, host_idx),
         y = MAV::make(x, {MAV::AxisIndexer::make_index(1, idx)});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    // no error
    func->execute();

    // one error
    host_idx->ptr<int>()[2] = 8;
    check_async_error(func.get(), 1);

    // two errors
    host_idx->ptr<int>()[5] = -10;
    check_async_error(func.get(), 2);
}

namespace {
void mesh_indexing_impl(HostTensorND& src, HostTensorND& dst,
                        HostTensorND& idx0, HostTensorND& idx1,
                        const std::vector<int>& axis,
                        std::function<void(float&, float&)> cb, bool batched) {
    auto pidx0 = idx0.ptr<int>();
    auto pidx1 = idx1.ptr<int>();
    auto dst_iter = megdnn::tensor_iter<float>(dst.as_megdnn()).begin();
    int data_idx[TensorLayout::MAX_NDIM];
    for (size_t i = 0; i < dst.shape().total_nr_elems(); ++i) {
        std::copy(dst_iter.idx(), dst_iter.idx() + dst.shape().ndim, data_idx);
        int ndim = dst.shape().ndim;
        data_idx[(ndim + axis[2]) % ndim] =
                data_idx[(ndim + axis[2]) % ndim] * 2 + 1;
        int& idx0 = data_idx[(ndim + axis[0]) % ndim];
        int& idx1 = data_idx[(ndim + axis[1]) % ndim];
        if (!batched) {
            idx0 = pidx0[idx0];
            idx1 = pidx1[idx1];
        } else {
            size_t n = dst_iter.idx()[0];
            idx0 = pidx0[n * dst.shape()[(ndim + axis[0]) % ndim] + idx0];
            idx1 = pidx1[n * dst.shape()[(ndim + axis[1]) % ndim] + idx1];
        }
        if (idx0 < 0) {
            idx0 += src.shape((ndim + axis[0]) % ndim);
        }
        if (idx1 < 0) {
            idx1 += src.shape((ndim + axis[1]) % ndim);
        }
        cb(*src.ptr<float>(data_idx, data_idx + dst.shape().ndim), *dst_iter);
        ++dst_iter;
    }
}

void mesh_gen_index(int* p_axis_size, HostTensorND& dest) {
    auto axis_size = *p_axis_size;
    mgb_assert(axis_size > 0);
    RNGxorshf rng{next_rand_seed()};
    std::uniform_int_distribution<int> dist{-axis_size, axis_size - 1};

    auto ptr = dest.ptr<int>();
    for (size_t i = 0, it = dest.layout().total_nr_elems(); i < it; ++i) {
        ptr[i] = dist(rng);
    }
}

void mesh_gen_non_replacement_indx(int *p_axis_size, HostTensorND& dest) {
    auto axis_size = *p_axis_size;
    mgb_assert(axis_size > 0);
    RNGxorshf rng{next_rand_seed()};
    size_t size, stride;
    if (dest.layout().ndim == 1) {
        size = 1;
        stride = dest.layout()[0];
    } else {
        mgb_assert(dest.layout().ndim == 2);
        size = dest.layout()[0];
        stride = dest.layout().stride[0];
    }
    for (size_t n = 0; n < size; ++n) {
        std::uniform_int_distribution<int> dist{-axis_size, axis_size - 1};
        std::set<int> used;

        auto ptr = dest.ptr<int>() + n * stride;
        for (size_t i = 0; i < stride; ++i) {
            while (true) {
                int val = dist(rng);
                int true_val = (val + axis_size) % axis_size;
                mgb_assert(true_val >= 0);
                if (used.find(true_val) == used.end()) {
                    ptr[i] = val;
                    used.insert(true_val);
                    break;
                }
            }
        }
    }
}
}  // namespace

TEST(TestOprIndexing, MeshIndexing) {
    set_rand_seed(19260817);
    {
        HostTensorGenerator<> gen;
        auto host_x = gen({5, 8, 8});
        auto host_idx_0 = std::make_shared<HostTensorND>(
                host_x->comp_node(), TensorShape{3}, dtype::Int32());
        auto host_idx_1 = std::make_shared<HostTensorND>(
                host_x->comp_node(), TensorShape{2}, dtype::Int32());
        auto graph = ComputingGraph::make();
        using AIdx = opr::indexing::AxisIndexer;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             idx_0 = opr::Host2DeviceCopy::make(*graph, host_idx_0),
             idx_1 = opr::Host2DeviceCopy::make(*graph, host_idx_1),
             y = opr::MeshIndexing::make(
                     x, {AIdx::make_index(0, idx_0), AIdx::make_index(1, idx_1),
                         AIdx::make_interval(-1, x.make_scalar(2),
                                             x.make_scalar(5), None)});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        host_idx_0->ptr<int>()[0] = 1;
        host_idx_0->ptr<int>()[1] = 2;
        host_idx_0->ptr<int>()[2] = 3;
        host_idx_1->ptr<int>()[0] = 3;
        host_idx_1->ptr<int>()[1] = -2;
        func->execute();
        ASSERT_EQ(TensorShape({3, 2, 3}), host_y.shape());

        auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 3; ++k) {
                    ASSERT_EQ(px[(i + 1) * 64 + (j + 1) * 3 * 8 + k + 2],
                              *(py++));
                }
            }
        }
    }

    using Checker = AutoOprChecker<3, 1>;
    using AIdx = opr::indexing::AxisIndexer;
    std::vector<int> axis;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0], idx0 = inputs[1], idx1 = inputs[2];
        return {opr::MeshIndexing::make(
                x, {AIdx::make_index(axis[0], idx0),
                    AIdx::make_index(axis[1], idx1),
                    AIdx::make_interval(axis[2], x.make_scalar(1),
                                        x.make_scalar(-1), x.make_scalar(2))})};
    };
    auto set_value = [](float& a, float& b) { b = a; };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx0 = *inp[1], &&idx1 = *inp[2];
        auto dshp = data.shape();
        int ndim = dshp.ndim;
        dshp[(ndim + axis[2]) % ndim] =
                (dshp[(ndim + axis[2]) % ndim] - 3) / 2 + 1;
        dshp[(ndim + axis[0]) % ndim] = idx0.shape(0);
        dshp[(ndim + axis[1]) % ndim] = idx1.shape(0);
        mesh_indexing_impl(data, dest[0].resize(dshp), idx0, idx1, axis,
                           set_value, false);
    };
    int axis0_size = -1;
    int axis1_size = -1;
    auto setup_checker = [&](Checker& checker, bool enable_grad) {
        using namespace std::placeholders;
        checker.set_input_generator(1,
                                    std::bind(mesh_gen_index, &axis0_size, _1));
        checker.set_input_dtype(1, dtype::Int32());

        checker.set_input_generator(2,
                                    std::bind(mesh_gen_index, &axis1_size, _1));
        checker.set_input_dtype(2, dtype::Int32());
        checker.set_input_allow_grad(1, false);
        checker.set_input_allow_grad(2, false);
        checker.set_output_allow_grad(0, enable_grad);
    };
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, true);
        axis = {0, 1, -1};
        axis0_size = 3;
        axis1_size = 3;
        checker.run({TensorShape{3, 3, 5}, {10}, {12}}, opt);
        axis0_size = 5;
        axis1_size = 7;
        checker.run({TensorShape{5, 7, 10, 10}, {3}, {5}}, opt);
        axis0_size = 5;
        axis1_size = 7;
        checker.run({TensorShape{5, 7, 10, 20}, {9}, {3}}, opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {-2, 1, -1};
        axis0_size = 50;
        axis1_size = 30;
        checker.run({TensorShape{10, 30, 50, 24}, {101}, {7}}, opt);
        axis0_size = 10;
        axis1_size = 20;
        checker.run({TensorShape{7, 20, 30, 10, 24}, {99}, {7}}, opt);
        checker.run({TensorShape{9, 20, 30, 10, 25}, {66}, {7}}, opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {2, 1, 0};
        axis0_size = 10;
        axis1_size = 20;
        checker.run({TensorShape{10, 20, 10, 3, 7}, {99}, {1}}, opt);
        checker.run({TensorShape{9, 30, 20, 3, 7}, {99}, {5}}, opt);
        checker.run({TensorShape{8, 20, 10, 7, 7}, {1}, {99}}, opt);
    }
}

TEST(TestOprIndexing, BatchedMeshIndexing) {
    using Checker = AutoOprChecker<3, 1>;
    using AIdx = opr::indexing::AxisIndexer;
    std::vector<int> axis;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0], idx0 = inputs[1], idx1 = inputs[2];
        return {opr::BatchedMeshIndexing::make(
                x, {AIdx::make_index(axis[0], idx0),
                    AIdx::make_index(axis[1], idx1),
                    AIdx::make_interval(axis[2], x.make_scalar(1),
                                        x.make_scalar(-1), x.make_scalar(2))})};
    };
    auto set_value = [](float& a, float& b) { b = a; };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx0 = *inp[1], &&idx1 = *inp[2];
        auto dshp = data.shape();
        int ndim = dshp.ndim;
        dshp[(ndim + axis[2]) % ndim] =
                (dshp[(ndim + axis[2]) % ndim] - 3) / 2 + 1;
        dshp[(ndim + axis[0]) % ndim] = idx0.shape(1);
        dshp[(ndim + axis[1]) % ndim] = idx1.shape(1);
        mesh_indexing_impl(data, dest[0].resize(dshp), idx0, idx1, axis,
                           set_value, true);
    };
    int axis0_size = -1;
    int axis1_size = -1;
    auto setup_checker = [&](Checker& checker, bool enable_grad) {
        using namespace std::placeholders;
        checker.set_input_generator(1,
                                    std::bind(mesh_gen_index, &axis0_size, _1));
        checker.set_input_dtype(1, dtype::Int32());

        checker.set_input_generator(2,
                                    std::bind(mesh_gen_index, &axis1_size, _1));
        checker.set_input_dtype(2, dtype::Int32());
        checker.set_input_allow_grad(1, false);
        checker.set_input_allow_grad(2, false);
        checker.set_output_allow_grad(0, enable_grad);
    };
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, true);
        axis = {1, 2, -1};
        axis0_size = 3;
        axis1_size = 3;
        checker.run({TensorShape{3, 3, 3, 5}, {3, 10}, {3, 12}}, opt);
        axis0_size = 7;
        axis1_size = 10;
        checker.run({TensorShape{5, 7, 10, 10}, {5, 3}, {5, 5}}, opt);
        axis0_size = 7;
        axis1_size = 10;
        checker.run({TensorShape{5, 7, 10, 20}, {5, 9}, {5, 3}}, opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {-2, 1, -1};
        axis0_size = 50;
        axis1_size = 30;
        checker.run({TensorShape{10, 30, 50, 24}, {10, 101}, {10, 7}}, opt);
        axis0_size = 10;
        axis1_size = 20;
        checker.run({TensorShape{7, 20, 30, 10, 24}, {7, 99}, {7, 7}}, opt);
        checker.run({TensorShape{9, 20, 30, 10, 25}, {9, 66}, {9, 7}}, opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {2, 1, -2};
        axis0_size = 10;
        axis1_size = 20;
        checker.run({TensorShape{10, 20, 10, 3, 7}, {10, 99}, {10, 1}}, opt);
        checker.run({TensorShape{9, 30, 20, 3, 7}, {9, 99}, {9, 5}}, opt);
        checker.run({TensorShape{8, 20, 10, 7, 7}, {8, 1}, {8, 99}}, opt);
    }
}

TEST(TestOprIndexing, IncrMeshIndexing) {
    using Checker = AutoOprChecker<4, 1>;
    using AIdx = opr::indexing::AxisIndexer;
    std::vector<int> axis;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0], idx0 = inputs[1], idx1 = inputs[2],
                  val = inputs[3];
        return {opr::IncrMeshIndexing::make(
                x, val,
                {AIdx::make_index(axis[0], idx0),
                 AIdx::make_index(axis[1], idx1),
                 AIdx::make_interval(axis[2], x.make_scalar(1),
                                     x.make_scalar(-1), x.make_scalar(2))})};
    };
    auto value_addition = [](float& data, float& value) { data += value; };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx0 = *inp[1], &&idx1 = *inp[2];
        dest[0].copy_from(data);
        mesh_indexing_impl(dest[0], *inp[3], idx0, idx1, axis, value_addition,
                           false);
    };
    int axis0_size = -1;
    int axis1_size = -1;
    auto setup_checker = [&](Checker& checker, bool enable_grad) {
        using namespace std::placeholders;
        checker.set_input_generator(1,
                                    std::bind(mesh_gen_index, &axis0_size, _1));
        checker.set_input_dtype(1, dtype::Int32());

        checker.set_input_generator(2,
                                    std::bind(mesh_gen_index, &axis1_size, _1));
        checker.set_input_dtype(2, dtype::Int32());
        checker.set_input_allow_grad(1, false);
        checker.set_input_allow_grad(2, false);
        checker.set_output_allow_grad(0, enable_grad);
    };
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, true);
        opt.numdiff_max_err = 1e-2;
        axis = {0, 1, -1};
        axis0_size = 3;
        axis1_size = 3;
        checker.run({TensorShape{3, 3, 5}, {10}, {12}, {10, 12, 2}}, opt);
        axis0_size = 5;
        axis1_size = 7;
        checker.run({TensorShape{5, 7, 10, 10}, {3}, {5}, {3, 5, 10, 4}}, opt);
        axis0_size = 5;
        axis1_size = 7;
        checker.run({TensorShape{5, 7, 10, 20}, {9}, {3}, {9, 3, 10, 9}}, opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {-2, 1, -1};
        axis0_size = 50;
        axis1_size = 30;
        checker.run({TensorShape{10, 30, 50, 24}, {101}, {7}, {10, 7, 101, 11}},
                    opt);
        axis0_size = 10;
        axis1_size = 20;
        checker.run(
                {TensorShape{7, 20, 30, 10, 24}, {99}, {7}, {7, 7, 30, 99, 11}},
                opt);
        checker.run(
                {TensorShape{9, 20, 30, 10, 25}, {66}, {7}, {9, 7, 30, 66, 12}},
                opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {2, 1, 0};
        axis0_size = 10;
        axis1_size = 20;
        checker.run(
                {TensorShape{10, 20, 10, 3, 7}, {99}, {1}, {4, 1, 99, 3, 7}},
                opt);
        checker.run({TensorShape{9, 30, 20, 3, 7}, {99}, {5}, {4, 5, 99, 3, 7}},
                    opt);
        checker.run({TensorShape{8, 20, 10, 7, 7}, {1}, {99}, {3, 99, 1, 7, 7}},
                    opt);
    }
}

TEST(TestOprIndexing, BatchedIncrMeshIndexing) {
    using Checker = AutoOprChecker<4, 1>;
    using AIdx = opr::indexing::AxisIndexer;
    std::vector<int> axis;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0], idx0 = inputs[1], idx1 = inputs[2],
                  val = inputs[3];
        return {opr::BatchedIncrMeshIndexing::make(
                x, val,
                {AIdx::make_index(axis[0], idx0),
                 AIdx::make_index(axis[1], idx1),
                 AIdx::make_interval(axis[2], x.make_scalar(1),
                                     x.make_scalar(-1), x.make_scalar(2))})};
    };
    auto value_addition = [](float& data, float& value) { data += value; };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx0 = *inp[1], &&idx1 = *inp[2];
        dest[0].copy_from(data);
        mesh_indexing_impl(dest[0], *inp[3], idx0, idx1, axis, value_addition,
                           true);
    };
    int axis0_size = -1;
    int axis1_size = -1;
    auto setup_checker = [&](Checker& checker, bool enable_grad) {
        using namespace std::placeholders;
        checker.set_input_generator(1,
                                    std::bind(mesh_gen_index, &axis0_size, _1));
        checker.set_input_dtype(1, dtype::Int32());

        checker.set_input_generator(2,
                                    std::bind(mesh_gen_index, &axis1_size, _1));
        checker.set_input_dtype(2, dtype::Int32());
        checker.set_input_allow_grad(1, false);
        checker.set_input_allow_grad(2, false);
        checker.set_output_allow_grad(0, enable_grad);
    };
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, true);
        axis = {2, 1, -1};
        axis0_size = 3;
        axis1_size = 3;
        checker.run({TensorShape{3, 3, 3, 5}, {3, 10}, {3, 12}, {3, 12, 10, 2}},
                    opt);
        axis0_size = 5;
        axis1_size = 7;
        checker.run({TensorShape{5, 7, 5, 10}, {5, 3}, {5, 5}, {5, 5, 3, 4}},
                    opt);
        axis0_size = 5;
        axis1_size = 7;
        opt.numdiff_max_err = 1e-2;
        checker.run({TensorShape{5, 7, 5, 20}, {5, 9}, {5, 3}, {5, 3, 9, 9}},
                    opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {-2, 1, -1};
        axis0_size = 50;
        axis1_size = 30;
        checker.run({TensorShape{10, 30, 50, 24},
                     {10, 101},
                     {10, 7},
                     {10, 7, 101, 11}},
                    opt);
        axis0_size = 10;
        axis1_size = 20;
        checker.run({TensorShape{7, 20, 30, 10, 24},
                     {7, 99},
                     {7, 7},
                     {7, 7, 30, 99, 11}},
                    opt);
        checker.run({TensorShape{9, 20, 30, 10, 25},
                     {9, 66},
                     {9, 7},
                     {9, 7, 30, 66, 12}},
                    opt);
    }
}

TEST(TestOprIndexing, SetMeshIndexing) {
    using Checker = AutoOprChecker<4, 1>;
    using AIdx = opr::indexing::AxisIndexer;
    std::vector<int> axis;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        SymbolVar x = inputs[0], idx0 = inputs[1], idx1 = inputs[2],
                  val = inputs[3];
        return {opr::SetMeshIndexing::make(
                x, val,
                {AIdx::make_index(axis[0], idx0),
                 AIdx::make_index(axis[1], idx1),
                 AIdx::make_interval(axis[2], x.make_scalar(1),
                                     x.make_scalar(-1), x.make_scalar(2))})};
    };
    auto set_value = [](float& data, float& value) { data = value; };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&idx0 = *inp[1], &&idx1 = *inp[2];
        dest[0].copy_from(data);
        mesh_indexing_impl(dest[0], *inp[3], idx0, idx1, axis, set_value,
                           false);
    };
    int axis0_size = -1;
    int axis1_size = -1;
    auto setup_checker = [&](Checker& checker, bool enable_grad) {
        using namespace std::placeholders;
        checker.set_input_generator(
                1, std::bind(mesh_gen_non_replacement_indx, &axis0_size, _1));
        checker.set_input_dtype(1, dtype::Int32());

        checker.set_input_generator(
                2, std::bind(mesh_gen_non_replacement_indx, &axis1_size, _1));
        checker.set_input_dtype(2, dtype::Int32());
        checker.set_input_allow_grad(1, false);
        checker.set_input_allow_grad(2, false);
        checker.set_output_allow_grad(0, enable_grad);
    };
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, true);

        axis = {0, 1, -1};
        axis0_size = 1;
        axis1_size = 1;
        checker.run({TensorShape{1, 1, 5}, {1}, {1}, {1, 1, 2}}, opt);

        axis0_size = 19;
        axis1_size = 20;
        checker.run({TensorShape{19, 20, 5}, {10}, {12}, {10, 12, 2}}, opt);
        axis0_size = 5;
        axis1_size = 7;
        checker.run({TensorShape{5, 7, 10, 10}, {3}, {5}, {3, 5, 10, 4}}, opt);
        axis0_size = 5;
        axis1_size = 7;

        opt.numdiff_max_err = 1e-2;
        checker.run({TensorShape{5, 7, 10, 20}, {5}, {3}, {5, 3, 10, 9}}, opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {-2, 1, -1};
        axis0_size = 50;
        axis1_size = 30;
        checker.run({TensorShape{10, 30, 50, 24}, {27}, {7}, {10, 7, 27, 11}},
                    opt);
        axis0_size = 10;
        axis1_size = 20;
        checker.run(
                {TensorShape{7, 20, 30, 10, 24}, {9}, {7}, {7, 7, 30, 9, 11}},
                opt);
        checker.run(
                {TensorShape{9, 20, 30, 10, 25}, {6}, {7}, {9, 7, 30, 6, 12}},
                opt);
    }
    {
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        setup_checker(checker, false);

        axis = {2, 1, 0};
        axis0_size = 10;
        axis1_size = 20;
        checker.run(
                {TensorShape{10, 20, 10, 3, 7}, {9}, {1}, {4, 1, 9, 3, 7}},
                opt);
        checker.run({TensorShape{9, 20, 10, 3, 7}, {9}, {5}, {4, 5, 9, 3, 7}},
                    opt);
        checker.run({TensorShape{8, 20, 10, 7, 7}, {1}, {9}, {3, 9, 1, 7, 7}},
                    opt);
    }
    { // only interval AxisIndexer given
        using Checker = AutoOprChecker<2, 1>;
        auto make_graph =
                [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
            SymbolVar x = inputs[0], val = inputs[1];
            return {opr::SetMeshIndexing::make(
                    x, val,
                    {AIdx::make_interval(0, x.make_scalar(1),
                        None, x.make_scalar(2))})};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            dest[0].copy_from(*inp[0]);
            auto value = *inp[1];
            auto value_iter = megdnn::tensor_iter<float>(value.as_megdnn()).begin();
            size_t n = dest[0].layout().stride[0];
            float* raw_ptr = dest[0].ptr<float>();
            for (size_t i = 0; i < value.shape().total_nr_elems(); ++i) {
                ptrdiff_t offset = (i / n * 2 + 1) * n + i % n;
                *(raw_ptr + offset) = *value_iter;
                ++ value_iter;
            }
        };
        Checker checker{make_graph, fwd};
        checker.run({TensorShape{11}, {5}});
        checker.run({TensorShape{6, 7}, {3, 7}});
        checker.run({TensorShape{4, 7, 1}, {2, 7, 1}});
        checker.run({TensorShape{7, 1, 1, 2}, {3, 1, 1, 2}});
    }
}

namespace {

template<class Opr>
void run_multi_axis_vec_empty_shape(
        const TensorShape& ishp, const TensorShape& idx0,
        const TensorShape& idx1, const TensorShape& tshp) {
    mgb_assert(ishp.ndim >= 4);
    mgb_assert(idx0.is_empty() || idx1.is_empty());
    using AI = opr::indexing::AxisIndexer;
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_idx;
    auto host_x = gen(ishp);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
        idx_dynamic_shape = opr::MarkDynamicVar::make(
            opr::ImmutableTensor::make(*graph, *gen_idx(idx0))),
        idx_static_shape =
            opr::ImmutableTensor::make(*graph, *gen_idx(idx1)),
        y = Opr::make(x, {
                    AI::make_interval(-1, None, None, x.make_scalar(2)),
                    AI::make_index(1, idx_dynamic_shape),
                    AI::make_index(2, idx_static_shape)});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_TRUE(host_y.shape().is_empty());
    MGB_ASSERT_SHAPE_EQ(host_y.shape(), tshp);
}

template<class Opr>
void run_modify_multi_axis_vec_empty_shape(
        const TensorShape& ishp, const TensorShape& vshp,
        const TensorShape& idx0, const TensorShape& idx1) {
    mgb_assert(ishp.ndim >= 4);
    mgb_assert(vshp.is_empty() && (idx0.is_empty() || idx1.is_empty()));
    using AI = opr::indexing::AxisIndexer;
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_idx;
    auto host_x = gen(ishp), host_v = gen(vshp);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
        v = opr::Host2DeviceCopy::make(*graph, host_v),
        idx_dynamic_shape = opr::MarkDynamicVar::make(
            opr::ImmutableTensor::make(*graph, *gen_idx(idx0))),
        idx_static_shape =
            opr::ImmutableTensor::make(*graph, *gen_idx(idx1)),
        y = Opr::make(x, v, {
                    AI::make_interval(-1, None, None, x.make_scalar(2)),
                    AI::make_index(1, idx_dynamic_shape),
                    AI::make_index(2, idx_static_shape)});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
}

}

TEST(TestOprIndexing, MultiAxisVecEmptyShape) {
    TensorShape ishp{8, 2, 3, 4};
    size_t n = ishp[0], last_ndim = ishp[ishp.ndim - 1] / 2;
    run_multi_axis_vec_empty_shape<opr::IndexingMultiAxisVec>(
            ishp, {0}, {0}, {n, 0, last_ndim});
    run_multi_axis_vec_empty_shape<opr::MeshIndexing>(
            ishp, {0}, {2}, {n, 0, 2, last_ndim});
    run_multi_axis_vec_empty_shape<opr::MeshIndexing>(
            ishp, {3}, {0}, {n, 3, 0, last_ndim});
    run_multi_axis_vec_empty_shape<opr::BatchedMeshIndexing>(
            ishp, {n, 0}, {n, 2}, {n, 0, 2, last_ndim});
    run_multi_axis_vec_empty_shape<opr::BatchedMeshIndexing>(
            ishp, {n, 4}, {n, 0}, {n, 4, 0, last_ndim});

    run_modify_multi_axis_vec_empty_shape<opr::IndexingIncrMultiAxisVec>(
            ishp, {n, 0, last_ndim}, {0}, {0});
    run_modify_multi_axis_vec_empty_shape<opr::IndexingSetMultiAxisVec>(
            ishp, {n, 0, last_ndim}, {0}, {0});
    run_modify_multi_axis_vec_empty_shape<opr::IncrMeshIndexing>(
            ishp, {n, 0, 2, last_ndim}, {0}, {2});
    run_modify_multi_axis_vec_empty_shape<opr::SetMeshIndexing>(
            ishp, {n, 3, 0, last_ndim}, {3}, {0});
    run_modify_multi_axis_vec_empty_shape<opr::BatchedIncrMeshIndexing>(
            ishp, {n, 4, 0, last_ndim}, {n, 4}, {n, 0});
    run_modify_multi_axis_vec_empty_shape<opr::BatchedSetMeshIndexing>(
            ishp, {n, 0, 5, last_ndim}, {n, 0}, {n, 5});
}

#endif  // MGB_ENABLE_EXCEPTION

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
