/**
 * \file src/opr/test/misc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/misc.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"

#include <numeric>
#include <random>

using namespace mgb;

namespace {
    void shape_abc(const TensorShape &shape, size_t axis,
            size_t &A, size_t &B, size_t &C) {
        auto acc_mul = [](const size_t *first, const size_t *last) {
            return std::accumulate(
                    first, last, 1u, std::multiplies<size_t>());
        };
        A = acc_mul(shape.shape, shape.shape+axis);
        B = shape.shape[axis];
        C = acc_mul(shape.shape+axis+1, shape.shape+shape.ndim);
    }

    void argsort_data_gen(HostTensorND& dest) {
        mgb_assert(dest.layout().ndim == 2 && dest.layout().is_contiguous());
        size_t m = dest.layout()[0], n = dest.layout()[1];
        auto ptr = dest.ptr<float>();
        RNGxorshf rng{next_rand_seed()};
        std::uniform_real_distribution<float> dist_base{-10.f, 10.f},
                dist_delta{0.1f, 1.2f};
        for (size_t i = 0; i < m; ++i) {
            auto v = dist_base(rng);
            for (size_t j = 0; j < n; ++j) {
                ptr[j] = v;
                v += dist_delta(rng);
            }
            std::shuffle(ptr, ptr + n, rng);
            ptr += n;
        }
    }
}

TEST(TestOprMisc, Argmxx) {
    auto run = [](bool is_max, int32_t axis, TensorShape sshape) {
        auto dshape = sshape;
        dshape.shape[axis] = 1;
        using Checker = AutoOprChecker<1, 1>;
        auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
            if (is_max)
                return {opr::Argmax::make(inputs[0], {axis})};
            else
                return {opr::Argmin::make(inputs[0], {axis})};
        };
        auto better_than = [&](float curr, float best)
        {
            if (is_max)
                return curr > best;
            else
                return curr < best;
        };
        auto fwd = [&](Checker::NumOutArray &out, Checker::NumInpArray inp) {
            out[0].dtype(dtype::Int32()).resize(dshape);
            size_t A, B, C;
            shape_abc(sshape, axis, A, B, C);
            for (size_t a = 0; a < A; ++a) for (size_t c = 0; c < C; ++c) {
                float best_val;
                size_t best_arg = -1;
                if (is_max)
                    best_val = std::numeric_limits<float>::lowest();
                else
                    best_val = std::numeric_limits<float>::max();
                for (size_t b = 0; b < B; ++b) {
                    float curr_val = inp[0]->ptr<float>()[(a*B+b)*C+c];
                    if (better_than(curr_val, best_val)) {
                        best_val = curr_val;
                        best_arg = b;
                    }
                }
                out[0].ptr<int>()[a*C+c] = best_arg;
            }
        };
        Checker{make_graph, fwd}.
            set_input_allow_grad(0, false).
            set_output_allow_grad(0, false).
            run({sshape}).
            run({sshape}).
            run({sshape});
    };
    run(true, 0, {5});
    run(true, 1, {2, 3, 4, 5});
    run(true, 2, {2, 3, 4, 5});
    run(true, 3, {2, 3, 4, 5});
    run(false, 0, {3, 4, 5});
    run(false, 1, {2, 3, 4, 5});
    run(false, 2, {2, 3, 4, 5});
    run(false, 3, {2, 3, 4, 5});
}

TEST(TestOprMisc, Argsort) {
    using Order = opr::Argsort::Param::Order;
    auto run = [](Order order) {
        using Checker = AutoOprChecker<1, 2>;
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            return opr::Argsort::make(inputs[0], order);
        };
        auto fwd = [&](Checker::NumOutArray& out, Checker::NumInpArray inp) {
            size_t m = inp[0]->shape()[0], n = inp[0]->shape()[1];
            auto pi = inp[0]->ptr<float>();
            auto poval = out[0].resize({m, n}).ptr<float>();
            auto poidx = out[1].resize({m, n}).ptr<int>();

            using KV = std::pair<float, int>;
            std::vector<KV> row(n);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    row[j].first = pi[i * n + j];
                    row[j].second = j;
                }
                if (order == Order::ASCENDING) {
                    std::sort(row.begin(), row.end());
                } else {
                    std::sort(row.begin(), row.end(), std::greater<KV>{});
                }

                for (size_t j = 0; j < n; ++j) {
                    poval[i * n + j] = row[j].first;
                    poidx[i * n + j] = row[j].second;
                }
            }
        };
        Checker::RunOptions opt;
        opt.numdiff_eps = 0.045;
        Checker{make_graph, fwd}
                .set_input_generator(0, argsort_data_gen)
                .set_output_allow_grad(1, false)
                .run({TensorShape{1, 1}}, opt)
                .run({TensorShape{5, 3}}, opt)
                .run({TensorShape{10, 24}}, opt);
    };
    run(Order::ASCENDING);
    run(Order::DESCENDING);
}

TEST(TestOprMisc, Cumsum) {
    using Param = opr::Cumsum::Param;
    auto run = [](const Param &param) {
        using Checker = AutoOprChecker<1, 1>;
        auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
                return {opr::Cumsum::make(inputs[0], param)};
            };
        auto fwd = [&](Checker::NumOutArray &out, Checker::NumInpArray inp) {
            out[0].resize(inp[0]->shape());

            auto pin = inp[0]->ptr<float>(), pout = out[0].ptr<float>();
            size_t A, B, C;
            int real_axis = param.axis;
            if (real_axis < 0) real_axis += 3;
            shape_abc(inp[0]->shape(), real_axis, A, B, C);
            ptrdiff_t stride = C;
            if (param.reverse)
                stride = -stride;
            for (size_t i = 0; i < A; ++ i) {
                for (size_t k = 0; k < C; ++ k) {
                    auto pi = pin + i * B * C + k,
                         po = pout + i * B * C + k;
                    if (param.reverse) {
                        pi += (B - 1) * C;
                        po += (B - 1) * C;
                    }
                    if (param.exclusive) {
                        *po = 0;
                        po += stride;
                    }
                    float sum = 0;
                    for (size_t j = 0; j < B - 1; ++ j) {
                        sum += pi[j * stride];
                        po[j * stride] = sum;
                    }
                    if (!param.exclusive) {
                        po[(B - 1) * stride] = sum + pi[(B - 1) * stride];
                    }
                }
            }
        };
        Checker{make_graph, fwd}.
            run({TensorShape{2, 3, 4}}).
            run({TensorShape{3, 1, 2}}).
            run({TensorShape{4, 2, 3}});
    };

    // test negative axis
    for (int32_t axis = -3; axis < 3; ++axis)
        for (int mask = 0; mask < 4; ++mask)
            run({axis, bool(mask >> 1), bool(mask & 1)});
}

TEST(TestOprMisc, CondTake) {
    using Param = opr::CondTake::Param;
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        return {opr::CondTake::make(
                inputs[0], inputs[1], {Param::Mode::LT})[0]};
    };

    auto fwd = [&](Checker::NumOutArray &out, Checker::NumInpArray inp) {
        std::vector<float> values;
        auto data = inp[0]->ptr<float>(), mask = inp[1]->ptr<float>();
        auto isize = inp[0]->shape().total_nr_elems();
        for (size_t i = 0; i < isize; ++ i) {
            if (mask[i] < 0) {
                values.push_back(data[i]);
            }
        }
        out[0].resize({values.size()});
        memcpy(out[0].ptr<float>(),
                values.data(), sizeof(float) * values.size());
    };

    auto ensure_nonempty = [](Checker::NumInpArray inp) {
        auto mask = inp[1]->ptr<float>();
        auto isize = inp[1]->shape().total_nr_elems();
        for (size_t i = 0; i < isize; ++ i) {
            if (mask[i] < 0)
                return;
        }
        mask[isize - 1] = -1;
    };

    auto mki = [](const TensorShape &shp) -> Checker::ShapeInpArray {
        return {shp, shp};
    };
    Checker{make_graph, fwd}.
        set_input_allow_grad(1, false).
        set_input_coordinator(ensure_nonempty).
        run(mki({2})).
        run(mki({3, 5, 8})).
        run(mki({100}));
}

TEST(TestOprMisc, CondTakeEmptyOut) {
    using Param = opr::CondTake::Param;
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    host_x->ptr<float>()[0] = 1;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto out = opr::CondTake::make(x, x, {Param::Mode::LT});
    HostTensorND host_out0, host_out1;
    auto func = graph->compile({make_callback_copy(out[0], host_out0),
            make_callback_copy(out[1], host_out1)});
    func->execute();
    ASSERT_EQ(TensorShape{0}, host_out0.shape());
    ASSERT_EQ(TensorShape{0}, host_out1.shape());
}

TEST(TestOprMisc, TopKValueOnly) {
    auto run = [](bool dyn_k, bool non_contig) {
        using Checker = AutoOprChecker<1, 1>;
        std::shared_ptr<HostTensorND> host_k;

        SymbolVar var_x0, var_x1;

        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            auto k = opr::Host2DeviceCopy::make(
                    *inputs[0].node()->owner_graph(), host_k);
            if (dyn_k) {
                k = opr::MarkDynamicVar::make(k);
            }
            auto x = inputs[0];
            if (non_contig) {
                var_x0 = x;
                x = opr::Subtensor::make(
                        x, {opr::Subtensor::AxisIndexer::make_interval(
                                   1, None, opr::GetVarShape::make(x, 1) / 2,
                                   None)});
                var_x1 = x;
            }
            auto outs = opr::TopK::make(x, k, opr::TopK::Param::Mode::KTH_ONLY);
            return {outs[0]};
        };
        auto fwd = [&](Checker::NumOutArray& out, Checker::NumInpArray inp) {
            auto opr = megdnn_naive_handle()->create_operator<megdnn::TopK>();
            int k = host_k->ptr<int>()[0];
            HostTensorND x = *inp[0];
            if (non_contig) {
                auto layout = x.layout();
                layout.shape[1] /= 2;
                x = x.sub(SubTensorSpec::make_from_layout(layout));
            }

            TensorLayout outl0, outl1;
            opr->deduce_layout(k, x.layout(), outl0, outl1);

            size_t wk_size =
                    opr->get_workspace_in_bytes(k, x.layout(), outl0, outl1);
            std::unique_ptr<dt_byte[]> wk_store{new dt_byte[wk_size]};
            opr->exec(k, x.as_megdnn(), out[0].resize(outl0).as_megdnn(), {},
                      {wk_store.get(), wk_size});
        };
        Checker checker{make_graph, fwd};
        checker.set_input_generator(0, argsort_data_gen);

        host_k = std::make_shared<HostTensorND>(checker.comp_node(),
                                                TensorShape{1}, dtype::Int32{});
        host_k->ptr<int>()[0] = 1;
        Checker::RunOptions opt;
        opt.numdiff_eps = 0.047;
        auto invoke = [&](int k, size_t m, size_t n) {

            host_k->ptr<int>()[0] = k;
            checker.run({TensorShape{m, n}}, opt);
        };

        if (!non_contig) {
            invoke(1, 1, 1);
        }
        invoke(-2, 3, 2);
        invoke(-1, 4, 5);
        invoke(3, 10, 33);
        invoke(-8, 23, 35);

        if (non_contig) {
            ASSERT_EQ(prev_dev_ptr(var_x0), prev_dev_ptr(var_x1));
        }
    };

    for (auto i : {false, true}) {
        for (auto j : {false, true}) {
            run(i, j);
        }
    }
}

TEST(TestOprMisc, TopKSorted) {
    using Checker = AutoOprChecker<1, 2>;
    std::shared_ptr<HostTensorND> host_k;
    auto constexpr mode = opr::TopK::Param::Mode::VALUE_IDX_SORTED;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto k = opr::Host2DeviceCopy::make(*inputs[0].node()->owner_graph(),
                                            host_k);
        auto x = inputs[0];
        return opr::TopK::make(x, k, mode);
    };
    auto fwd = [&](Checker::NumOutArray& out, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->create_operator<megdnn::TopK>();
        opr->param().mode = mode;
        int k = host_k->ptr<int>()[0];
        TensorLayout outl0, outl1;
        opr->deduce_layout(k, inp[0]->layout(), outl0, outl1);

        size_t wk_size =
                opr->get_workspace_in_bytes(k, inp[0]->layout(), outl0, outl1);
        std::unique_ptr<dt_byte[]> wk_store{new dt_byte[wk_size]};
        opr->exec(k, inp[0]->as_megdnn(), out[0].resize(outl0).as_megdnn(),
                  out[1].resize(outl1).as_megdnn(), {wk_store.get(), wk_size});
    };
    Checker checker{make_graph, fwd};
    checker.set_input_generator(0, argsort_data_gen)
            .set_output_allow_grad(1, false);

    host_k = std::make_shared<HostTensorND>(checker.comp_node(), TensorShape{1},
                                            dtype::Int32{});
    host_k->ptr<int>()[0] = 1;
    Checker::RunOptions opt;
    opt.numdiff_eps = 0.047;
    auto invoke = [&](int k, size_t m, size_t n) {

        host_k->ptr<int>()[0] = k;
        checker.run({TensorShape{m, n}}, opt);
    };

    invoke(1, 1, 1);
    invoke(-1, 3, 5);
    invoke(5, 13, 23);
    invoke(-8, 35, 4);
}

TEST(TestOprMisc, TopKSortedIdxOnly) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    std::shared_ptr<HostTensorND> host_x = gen({2, 5});
    std::shared_ptr<HostTensorND> host_y = gen({2, 5});
    for (size_t i = 0; i < 10; ++i) {
        host_y->ptr<float>()[i] = 0.0f;
    }
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::TopK::make(x, x.make_scalar(3),
                               opr::TopK::Param::Mode::VALUE_IDX_SORTED)[1],
         y = opr::TypeCvt::make(idx, dtype::Float32{}),
         gx = cg::grad(opr::reduce_sum(y, y.make_scalar(1)), x);
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_gx, *host_y);
}

TEST(TestOprMisc, TopKGrad) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    std::shared_ptr<HostTensorND> host_x = gen({2, 5});
    std::shared_ptr<HostTensorND> host_k = gen({1});
    host_k->ptr<float>()[0] = 3;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         k = opr::Host2DeviceCopy::make(*graph, host_k),
         ki = opr::TypeCvt::make(k, dtype::Int32{}),
         val = opr::TopK::make(x, ki,
                               opr::TopK::Param::Mode::VALUE_IDX_SORTED)[0],
         gk = cg::grad(opr::reduce_sum(val, val.make_scalar(1)), ki, true, false);
    EXPECT_TRUE(gk == nullptr);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
