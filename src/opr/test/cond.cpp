/**
 * \file src/opr/test/cond.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/cond.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/utils/timer.h"

#include <bitset>

#if MGB_ENABLE_COND_EXEC

using namespace mgb;

namespace {
using MergeMode = opr::CondExecMerge::Param::Mode;

//! return y = (pred == 1 ? x : null)
SymbolVar make_one_cond(SymbolVar pred, SymbolVar x, size_t nr_branch = 1,
                        size_t this_branch = 0, bool grad_cond_out = false) {
    SymbolVar xcond;
    SymbolVarArray keys(nr_branch, pred.make_scalar_dt(0));
    keys.at(this_branch) = pred.make_scalar_dt(1);
    auto masks = opr::CondExecPred::make(pred, keys);
    EXPECT_EQ(nr_branch, masks.size());
    using Param = opr::CondExecMark::Param;
    Param p;
    if (grad_cond_out) {
        p.grad_mode = Param::GradMode::SUM_COND_OUT;
    }
    unpack_vector(opr::CondExecMark::make(masks.at(this_branch), {x}, p),
                  xcond);
    return xcond;
}

SymbolVar make_call_rec(SymbolVar x, int* cnt) {
    auto cb = [cnt](DeviceTensorND&) { ++*cnt; };
    opr::CallbackInjector::Param param{cb};
    param.invoke_for_static_infer = false;
    return opr::CallbackInjector::make(x, param);
}

SymbolVar merge_one_out(const SymbolVarArray& inputs_orig, MergeMode mode,
                        size_t nr_distractor = 0,
                        const VarNodeArrayView& out_shapes = {}) {
    SymbolVarArray inputs;
    for (size_t i = 0; i < inputs_orig.size(); ++i) {
        for (size_t j = 0; j <= nr_distractor; ++j) {
            if (j == nr_distractor / 2) {
                inputs.push_back(inputs_orig[i]);
            } else {
                inputs.push_back(inputs_orig[i] +
                                 int(i * (nr_distractor + 1) + j + 1));
            }
        }
    }
    auto out = opr::CondExecMerge::make(
            inputs, {static_cast<uint32_t>(nr_distractor + 1), mode},
            out_shapes);
    EXPECT_EQ(nr_distractor + 1, out.size());
    return out[nr_distractor / 2];
}

void test_merge_opr(MergeMode mode, bool pred_dynamic, bool final_sum) {
    if (final_sum && mode != MergeMode::SUM_COND_OUT) {
        return;
    }

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int;
    auto host_inp0 = gen({2, 3}), host_inp1 = gen({2, 3}),
         host_pred0 = gen_int({1}), host_pred1 = gen_int({1});
    host_pred0->ptr<int>()[0] = 0;
    host_pred1->ptr<int>()[0] = 1;

    SymbolVar inp0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_inp0),
              inp1 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_inp1),
              pred0 = opr::Host2DeviceCopy::make(*graph, host_pred0),
              pred1 = opr::Host2DeviceCopy::make(*graph, host_pred1);

    if (pred_dynamic) {
        pred0 = opr::MarkDynamicVar::make(pred0);
        pred1 = opr::MarkDynamicVar::make(pred1);
    }

    int call0 = 0, call1 = 0, call2 = 0, call3 = 0;
    SymbolVar inp0_cond = make_call_rec(make_one_cond(pred0, inp0, 3, 2) / 2,
                                        &call0),
              inp1_cond = make_call_rec(make_one_cond(pred1, inp1, 4, 1) * 3,
                                        &call1),
              merged = merge_one_out({inp0_cond, inp1_cond}, mode, 3), out;

    if (final_sum) {
        // check for ExecutionMask produced by CondExecMerge
        out = make_call_rec(merged, &call3);
        out = merge_one_out({out}, MergeMode::SUM, 2) - 1;
        out = make_call_rec(out, &call2);
        mode = MergeMode::SUM;
    } else {
        out = make_call_rec(merged, &call2) - 1;
    }

    auto make_expect = [&](int pred0, int pred1) {
        HostTensorND ret{host_inp0->comp_node(), host_inp0->shape()};
        auto p0 = host_inp0->ptr<float>(), p1 = host_inp1->ptr<float>(),
             pr = ret.ptr<float>();
        for (size_t i = 0, it = ret.shape().total_nr_elems(); i < it; ++i) {
            pr[i] = -1;
            if (pred0) {
                pr[i] += p0[i] / 2;
            }
            if (pred1) {
                pr[i] += p1[i] * 3;
            }
        }
        return ret;
    };

    // static inference helper
    auto updater_shp = cg::static_infer::StaticInferUpdater::make(),
         updater_val = cg::static_infer::StaticInferUpdater::make();
    using IDType = cg::static_infer::DepType;
    if (!pred_dynamic) {
        updater_shp->add_dest({out.node(), IDType::SHAPE});
        updater_val->add_dest({out.node(), IDType::VALUE});
    } else if (mode != MergeMode::EXACT_ONE) {
        updater_shp->add_dest({out.node(), IDType::SHAPE});
    }
    auto infer_shape = [&]() {
        updater_shp->update();
        return graph->static_infer_manager().infer_shape(out.node());
    };
    auto infer_value = [&]() {
        updater_val->update();
        auto val = graph->static_infer_manager().infer_value(out.node());
        HostTensorND ret;
        ret.copy_from(val);
        return ret;
    };

    HostTensorND host_out;
    auto func = graph->compile({make_callback_copy(out, host_out)});

    auto check_all = [&](int pred0, int pred1) {
        call0 = call1 = call2 = call3 = 0;

        auto expect = make_expect(pred0, pred1);
        if (mode != MergeMode::EXACT_ONE || !pred_dynamic) {
            ASSERT_EQ(expect.shape(), infer_shape());
        }
        if (!pred_dynamic) {
            MGB_ASSERT_TENSOR_NEAR(expect, infer_value(), 1e-5);
        }
        func->execute();
        MGB_ASSERT_TENSOR_NEAR(expect, host_out, 1e-5);

        ASSERT_EQ(pred0, call0);
        ASSERT_EQ(pred1, call1);
        ASSERT_EQ(1, call2);
        if (final_sum) {
            ASSERT_EQ(pred0 || pred1, call3);
        }
    };

    for (size_t casenum = 0; casenum < 4; ++casenum) {
        int pred0 = casenum >> 1, pred1 = casenum & 1;
        host_pred0->ptr<int>()[0] = pred0;
        host_pred1->ptr<int>()[0] = pred1;

        *host_inp0 = *gen({2 + casenum, 3});
        *host_inp1 = *gen({2 + casenum, 3});

        switch (mode) {
            case MergeMode::EXACT_ONE:
            case MergeMode::EXACT_ONE_SAME_SHAPE: {
                if (pred0 + pred1 == 1) {
                    check_all(pred0, pred1);
                    ASSERT_EQ(prev_dev_ptr(pred0 ? inp0_cond : inp1_cond),
                              prev_dev_ptr(merged));
                } else {
                    if (mode == MergeMode::EXACT_ONE) {
                        if (!pred_dynamic) {
                            ASSERT_THROW(infer_shape(), MegBrainError);
                        }
                    } else {
                        ASSERT_EQ(host_inp0->shape(), infer_shape());
                    }
                    if (!pred_dynamic) {
                        ASSERT_THROW(infer_value(), MegBrainError);
                    }
                    ASSERT_THROW(func->execute(), MegBrainError);
                }
                break;
            }
            case MergeMode::SUM:
            case MergeMode::SUM_COND_OUT: {
                if (pred0 || pred1 || mode == MergeMode::SUM) {
                    check_all(pred0, pred1);
                } else {
                    // no pred, and mode is SUM_COND_OUT
                    ASSERT_EQ(host_inp0->shape(), infer_shape());
                    call0 = call1 = call2 = 0;
                    func->execute();
                    ASSERT_EQ(0, call0);
                    ASSERT_EQ(0, call1);
                    ASSERT_EQ(0, call2);
                }
                break;
            }
            default:
                mgb_trap();
        }
    }
}

void test_simple_grad(bool grad_cond_out) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_y = gen({2, 3}), host_pred = gen({1});

    host_pred->ptr<float>()[0] = 0;

    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y"),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred);
    auto branches = opr::CondExecPred::make(
            pred, {pred.make_scalar(0.f), pred.make_scalar(1.f),
                   pred.make_scalar(2.f)});
    using GradMode = opr::CondExecMark::Param::GradMode;
    auto get_marked = [&branches, grad_cond_out](SymbolVar x, size_t br) {
        SymbolVar ret;
        unpack_vector(
                opr::CondExecMark::make(branches.at(br), {x},
                                        {grad_cond_out ? GradMode::SUM_COND_OUT
                                                       : GradMode::SUM}),
                ret);
        return ret;
    };
    int call_x = 0, call_y = 0;
    auto cond_x0 = get_marked(x, 0).rename("cx0"),
         cond_x1 = get_marked(x, 1).rename("cx1"),
         cond_y = get_marked(y, 2).rename("cy"),
         z = merge_one_out({cond_x0 * 2, cond_x1 * 3, cond_y * 2.3f},
                           MergeMode::EXACT_ONE_SAME_SHAPE)
                     .rename("merged"),
         loss = opr::reduce_sum_sqr(z + y, z.make_scalar(1)),
         gx = make_call_rec(cg::grad(loss, x), &call_x),
         gy = make_call_rec(cg::grad(loss, y), &call_y);

    std::array<float, 3> kx_all{2.f, 3.f, 0.f}, ky_all{1.f, 1.f, 3.3f};

    auto make_expect = [&](float kx, float ky, int wrt) {
        HostTensorND ret{host_x->comp_node(), host_x->shape()};
        auto pr = ret.ptr<float>(), px = host_x->ptr<float>(),
             py = host_y->ptr<float>();
        for (size_t i = 0, it = ret.shape().total_nr_elems(); i < it; ++i) {
            float s = px[i] * kx + py[i] * ky, ls = 2 * s;
            pr[i] = ls * (wrt ? ky : kx);
        }
        return ret;
    };

    HostTensorND host_gx, host_gy;
    auto func = graph->compile(
            {make_callback_copy(gx, host_gx), make_callback_copy(gy, host_gy)});

    for (size_t i = 0; i < 6; ++i) {
        *host_x = *gen({i + 3, 3});
        *host_y = *gen({i + 3, 3});

        int br_num = i % 3;
        host_pred->ptr<float>()[0] = br_num;
        call_x = 0;
        call_y = 0;
        func->execute();

        float kx = kx_all[br_num], ky = ky_all[br_num];

        if (grad_cond_out) {
            ASSERT_EQ(br_num <= 1, call_x);
            ASSERT_EQ(br_num == 2, call_y);
        } else {
            ASSERT_EQ(1, call_x);
            ASSERT_EQ(1, call_y);
            if (br_num < 2) {
                MGB_ASSERT_TENSOR_EQ(make_expect(kx, ky, 1), host_gy);
            } else {
                MGB_ASSERT_TENSOR_EQ(make_expect(kx, ky, 0), host_gx);
            }
        }
        if (br_num < 2) {
            MGB_ASSERT_TENSOR_EQ(make_expect(kx, ky, 0), host_gx);
        } else {
            MGB_ASSERT_TENSOR_EQ(make_expect(kx, ky, 1), host_gy);
        }
    }
}

void test_nested(bool check_grad) {
    using TwoVar = std::pair<SymbolVar, SymbolVar>;

    static auto make_bisect_pred = [](SymbolVar pred, float thresh) -> TwoVar {
        SymbolVar lt, ge;
        unpack_vector(
                opr::CondExecPred::make(pred, {pred.make_scalar_dt(thresh)},
                                        opr::CondExecPred::Mode::PIECEWISE),
                lt, ge);
        return {lt, ge};
    };
    static auto mark_two = [](SymbolVar x, TwoVar ppvs) -> TwoVar {
        SymbolVar a, b;
        unpack_vector(opr::CondExecMark::make(ppvs.first, {x}), a);
        unpack_vector(opr::CondExecMark::make(ppvs.second, {x}), b);
        return {a, b};
    };
    static auto make_bisect = [](SymbolVar x, SymbolVar pred, float thresh,
                                 int* call_lt, int* call_ge,
                                 TwoVar* pred_marked = nullptr) -> TwoVar {
        TwoVar pred_br;
        SymbolVar x_lt, x_ge;
        pred_br = make_bisect_pred(pred, thresh);
        std::tie(x_lt, x_ge) = mark_two(x, pred_br);
        if (pred_marked) {
            *pred_marked = mark_two(pred, pred_br);
        }
        return {make_call_rec(x_lt, call_lt), make_call_rec(x_ge, call_ge)};
    };

    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});

    int call_lt0, call_ge0;
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
              pred = opr::Host2DeviceCopy::make(*graph, host_pred)
                             .rename("pred"),
              x_lt_0, x_ge_0;
    TwoVar pred_th0;
    std::tie(x_lt_0, x_ge_0) =
            make_bisect(x, pred, 0, &call_lt0, &call_ge0, &pred_th0);

    x_lt_0 = x_lt_0.rename("lt0") / 2;
    x_ge_0 = x_ge_0.rename("ge0") * 2;

    int call_n0, call_n1, call_p0, call_p1;
    SymbolVar xn0, xn1, xp0, xp1;
    std::tie(xn0, xn1) = make_bisect(x_lt_0, pred_th0.first.rename("pred-neg"),
                                     -1, &call_n0, &call_n1);
    std::tie(xp0, xp1) = make_bisect(x_ge_0, pred_th0.second.rename("pred-pos"),
                                     1, &call_p0, &call_p1);

    int call_xn, call_xp;

    auto xn_merge = make_call_rec(
                 merge_one_out({xn0.rename("xn0") - 3, xn1.rename("xn1") + 3},
                               MergeMode::EXACT_ONE_SAME_SHAPE),
                 &call_xn),
         xp_merge = make_call_rec(
                 merge_one_out({xp0.rename("xp0") - 4, xp1.rename("xp1") + 4},
                               MergeMode::EXACT_ONE_SAME_SHAPE),
                 &call_xp),
         out = merge_one_out({xn_merge, xp_merge},
                             MergeMode::EXACT_ONE_SAME_SHAPE);

    // value infer would fail becase EXACT_ONE can not be satisfied (our
    // inference system has no conditional execution)
    // so we only check shape inference here
    ASSERT_EQ(host_x->shape(), out.shape());

    HostTensorND host_out, host_gx;
    ComputingGraph::OutputSpec out_spec{make_callback_copy(out, host_out)};

    if (check_grad) {
        auto loss = opr::reduce_sum_sqr(out, out.make_scalar(1)),
             gx = cg::grad(loss, x);
        out_spec.emplace_back(make_callback_copy(gx, host_gx));
    }

    auto func = graph->compile(out_spec);

    func->to_json()->writeto_fpath(output_file(
            ssprintf("TestCondExec.nested-grad%d.json", check_grad)));

    std::array<float, 4> all_biases{-3.f, 3.f, -4.f, 4.f};
    for (size_t casenum = 0; casenum < 4; ++casenum) {
        host_pred->ptr<float>()[0] = -1.5 + casenum;
        call_lt0 = call_ge0 = call_n0 = call_n1 = call_p0 = call_p1 = call_xn =
                call_xp = 0;

        *host_x = *gen({casenum + 6, 4});
        float k = casenum < 2 ? 0.5f : 2.f, b = all_biases[casenum];
        HostTensorND expect, expect_gx;

        // init expect
        {
            auto ptr = expect.copy_from(*host_x).ptr<float>();
            for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it;
                 ++i) {
                ptr[i] = ptr[i] * k + b;
            }
        }

        // init expect_gx
        if (check_grad) {
            auto ptr = expect_gx.copy_from(*host_x).ptr<float>();
            for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it;
                 ++i) {
                auto x = ptr[i];
                ptr[i] = (k * x + b) * 2 * k;
            }
        }

        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_out);
        if (check_grad) {
            MGB_ASSERT_TENSOR_EQ(expect_gx, host_gx);
        }
        ASSERT_EQ(casenum < 2, call_lt0);
        ASSERT_EQ(casenum >= 2, call_ge0);
        ASSERT_EQ(1, call_n0 + call_n1 + call_p0 + call_p1);
        ASSERT_EQ((call_n0 << 0) | (call_n1 << 1) | (call_p0 << 2) |
                          (call_p1 << 3),
                  1 << casenum);
        ASSERT_EQ(call_lt0, call_xn);
        ASSERT_EQ(call_ge0, call_xp);
    }
}

void check_waiting_spec(SymbolVar var, const VarNodeArrayView& to_wait) {
    auto&& spec = var.node()->owner_opr()->input_waiting_spec();
    if (to_wait.empty()) {
        ASSERT_TRUE(spec.empty());
        return;
    }

    ASSERT_EQ(1u, spec.size());
    ASSERT_EQ(var.node()->comp_node(), spec[0].comp_node);

    ThinHashSet<VarNode*> to_wait_set;
    for (auto i : to_wait) {
        to_wait_set.insert(i);
    }

    for (auto i : spec[0].dev_ready) {
        ASSERT_EQ(1u, to_wait_set.count(i)) << SymbolVar{i};
    }

    ASSERT_EQ(to_wait_set.size(), spec[0].dev_ready.size());
}

class DynamicMemLeakChecker final : public cg::DeviceMemoryAllocator {
    std::atomic_size_t m_nr_alive{0};

public:
    void alloc_dynamic(VarNode* var, DeviceTensorStorage& dest,
                       size_t size) override {
        ASSERT_LT(dest.size(), size);
        ++m_nr_alive;
        auto ptr = dest.comp_node().alloc_device(size);
        auto del = [ this, cn = dest.comp_node() ](void* ptr) {
            cn.free_device(ptr);
            auto nr = m_nr_alive.fetch_sub(1);
            ASSERT_GT(nr, 0u);
        };
        dest.reset(dest.comp_node(), size, {static_cast<dt_byte*>(ptr), del});
    }

    size_t nr_alive() const { return m_nr_alive; }

    ~DynamicMemLeakChecker() { EXPECT_EQ(0u, nr_alive()); }
};

}  // anonymous namespace

TEST(TestCondExec, MarkSimple) {
    int nr_call = 0;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred);
    SymbolVar xcond, ppv;
    unpack_vector(opr::CondExecPred::make(pred, {pred.make_scalar(0.f)},
                                          opr::CondExecPred::Param::Mode::CASE),
                  ppv);
    ppv = opr::CondExecPredLogical::make({ppv},
                                         opr::CondExecPredLogical::Mode::NAND);
    unpack_vector(opr::CondExecMark::make(ppv, {x}), xcond);
    {
        ASSERT_THROW(opr::CondExecMark::make(xcond, {x}), GraphError);
        // also test dedup
        auto tmp = opr::CondExecMark::mark_if_need(xcond, {x});
        ASSERT_EQ(xcond, tmp);
        ASSERT_EQ(ppv.node(), tmp.node()->owner_opr()->input().back());
    }
    auto y = make_call_rec(xcond + 2.3f, &nr_call);
    HostTensorND host_y;

    ASSERT_EQ(0u,
              y.node()->owner_opr()->node_prop().dep_map().count(ppv.node()));

    auto func = graph->compile({make_callback_copy(y, host_y)});

    // dependency added in topo sorter
    ASSERT_EQ(y.node()->owner_opr()->node_prop().dep_map().at(ppv.node()),
              cg::OperatorNodeBase::NodeProp::DepType::DEV_COMP_ORDER);

    auto make_expect = [&host_x]() {
        auto graph = ComputingGraph::make();
        HostTensorND ret;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        graph->compile({make_callback_copy(x + 2.3f, ret)})->execute();
        return ret;
    };

    auto pp = host_pred->ptr<float>();
    pp[0] = 0;
    func->execute();
    ASSERT_EQ(0, nr_call);
    ASSERT_TRUE(host_y.empty());

    pp[0] = 1;
    func->execute();
    ASSERT_EQ(1, nr_call);
    MGB_ASSERT_TENSOR_EQ(make_expect(), host_y);
    host_y = {};

    *host_x = *gen({5, 8});
    pp[0] = 0;
    func->execute();
    ASSERT_EQ(1, nr_call);
    ASSERT_TRUE(host_y.empty());

    pp[0] = 1;
    func->execute();
    ASSERT_EQ(2, nr_call);
    MGB_ASSERT_TENSOR_EQ(make_expect(), host_y);
    ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(xcond));
}

TEST(TestCondExec, MarkConst) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_pred = gen({1});
    host_pred->ptr<float>()[0] = 0;
    auto pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         y0 = make_one_cond(pred, pred.make_scalar(2.3f)),
         y1 = make_one_cond(pred + 1, pred.make_scalar(3.2f)),
         z = merge_one_out({y0, y1}, MergeMode::EXACT_ONE);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});

    func->execute();
    ASSERT_EQ(TensorShape{1}, host_z.shape());
    ASSERT_EQ(3.2f, host_z.ptr<float>()[0]);

    host_pred->ptr<float>()[0] = 1;
    func->execute();
    ASSERT_EQ(2.3f, host_z.ptr<float>()[0]);
}

TEST(TestCondExec, Merge) {
    for (int i = 0; i < 16; ++i) {
        int im = i >> 2, idyn = (i >> 1) & 1, final_sum = i & 1;
        test_merge_opr(static_cast<MergeMode>(im), idyn, final_sum);
        ASSERT_FALSE(Test::HasFailure())
                << "failed for mode=" << im << " dyn=" << idyn
                << " final_sum=" << final_sum;
    }
}

TEST(TestCondExec, SimpleGrad) {
    test_simple_grad(false);
}

TEST(TestCondExec, SimpleGradCondOut) {
    test_simple_grad(true);
}

TEST(TestCondExec, PredMode) {
    using Mode = opr::CondExecPred::Mode;

    // each case is a pair containing [pred, [branch_result]]
    using CaseDesc = std::vector<std::pair<float, std::vector<bool>>>;

    // pred opr is constructed using keys {0, 1, 2}
    auto run = [](Mode mode, const CaseDesc& cases) {
        auto graph = ComputingGraph::make();
        auto make_hv = [](float val) {
            auto ret = std::make_shared<HostTensorND>(CompNode::load("xpux"),
                                                      TensorShape{1});
            ret->ptr<float>()[0] = val;
            return ret;
        };
        auto host_pred = make_hv(0), host_x = make_hv(0);
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             pred = opr::Host2DeviceCopy::make(*graph, host_pred);
        auto branches = opr::CondExecPred::make(
                pred,
                {pred.make_scalar(0.f), pred.make_scalar(1.f),
                 pred.make_scalar(2.f)},
                mode);

        size_t nr_branch = cases[0].second.size();
        ASSERT_EQ(nr_branch, branches.size());
        SymbolVarArray branch_vars, branch_vars_dyn;
        auto x_dyn = opr::MarkDynamicVar::make(x);
        for (size_t i = 0; i < nr_branch; ++i) {
            SymbolVar ret;
            int delta = 1 << i;
            unpack_vector(opr::CondExecMark::make(branches.at(i), {x}), ret);
            branch_vars.emplace_back(ret + delta);
            unpack_vector(opr::CondExecMark::make(branches.at(i), {x_dyn}),
                          ret);
            branch_vars_dyn.emplace_back(ret + delta);
        }
        auto y = merge_one_out(branch_vars, MergeMode::SUM),
             y_dyn = merge_one_out(branch_vars_dyn, MergeMode::SUM, 0,
                                   {x.symshape()});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y_dyn, host_y)});
        auto updater = cg::static_infer::StaticInferUpdater::make();
        updater->add_dest({y.node(), cg::static_infer::DepType::VALUE});
        auto&& mgr = graph->static_infer_manager();

        for (auto&& i : cases) {
            host_pred->ptr<float>()[0] = i.first;
            updater->update();
            HostTensorND infer_val;
            infer_val.copy_from(mgr.infer_value(y.node())).sync();
            func->execute();
            ASSERT_EQ(TensorShape{1}, infer_val.shape());
            ASSERT_EQ(TensorShape{1}, host_y.shape());
            uint32_t vinfer = infer_val.ptr<float>()[0],
                     vy = host_y.ptr<float>()[0];
            ASSERT_EQ(vinfer, vy) << "input=" << i.first
                                  << " vinfer=" << std::bitset<8>{vinfer}
                                  << " vy=" << std::bitset<8>{vy};

            auto v = vy;
            for (size_t br = 0; br < nr_branch; ++br) {
                ASSERT_EQ(i.second[br], v & 1)
                        << "input=" << i.first << " branch=" << br
                        << " val=" << std::bitset<8>{vy};
                v >>= 1;
            }
        }
    };

    run(Mode::CASE, {
                            {0.f, {1, 0, 0}},
                            {2.f, {0, 0, 1}},
                            {2.1f, {0, 0, 0}},
                    });
    ASSERT_FALSE(Test::HasFailure()) << "CASE mode failed";

    run(Mode::CASE_FALLBACK,
        {{0.f, {1, 0, 0, 0}}, {2.f, {0, 0, 1, 0}}, {2.1f, {0, 0, 0, 1}}});
    ASSERT_FALSE(Test::HasFailure()) << "CASE_FALLBACK mode failed";

    run(Mode::PIECEWISE, {{-1.f, {1, 0, 0, 0}},
                          {-0.1f, {1, 0, 0, 0}},
                          {0.f, {0, 1, 0, 0}},
                          {0.1f, {0, 1, 0, 0}},
                          {0.99f, {0, 1, 0, 0}},
                          {1.f, {0, 0, 1, 0}},
                          {1.01f, {0, 0, 1, 0}},
                          {1.5f, {0, 0, 1, 0}},
                          {2.f, {0, 0, 0, 1}},
                          {2e3f, {0, 0, 0, 1}}});
    ASSERT_FALSE(Test::HasFailure()) << "PIECEWISE mode failed";

    static_assert(opr::CondExecPred::Param::MODE_NR_MEMBER == 3,
                  "not all mode tested");
}

TEST(TestCondExec, PredLogicalMode) {
    using Mode = opr::CondExecPredLogical::Mode;

    using Checker = thin_function<bool(int nr_true)>;
    auto run = [](Mode mode, const size_t nr_input, Checker checker) {
        const size_t nr_case = 1 << nr_input;
        auto host_pred = std::make_shared<HostTensorND>(CompNode::load("xpux"),
                                                        TensorShape{nr_case});
        auto host_x = std::make_shared<HostTensorND>(CompNode::load("xpux"),
                                                     TensorShape{1});
        memset(host_pred->ptr<float>(), 0, sizeof(float) * nr_case);
        host_x->ptr<float>()[0] = 0;

        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             pred = opr::Host2DeviceCopy::make(*graph, host_pred),
             pred_dyn = opr::MarkDynamicVar::make(pred);
        SymbolVarArray inputs, inputs_dyn;
        for (size_t i = 0; i < nr_input; ++i) {
            SymbolVar p, p_dyn, key = pred.make_scalar_dt(1);
            opr::Subtensor::IndexDesc idx{
                    opr::indexing::AxisIndexer::make_index(
                            0, pred.make_scalar(static_cast<int>(i)))};
            auto sub = [&idx](SymbolVar x) {
                return opr::Subtensor::make(x, idx);
            };
            unpack_vector(opr::CondExecPred::make(sub(pred), {key}), p);
            unpack_vector(opr::CondExecPred::make(sub(pred_dyn), {key}), p_dyn);
            inputs.push_back(p);
            inputs_dyn.push_back(p_dyn);
        }

        SymbolVar logic_out = opr::CondExecPredLogical::make(inputs, mode),
                  logic_out_dyn =
                          opr::CondExecPredLogical::make(inputs_dyn, mode),
                  x_mark, x_mark_dyn;
        unpack_vector(opr::CondExecMark::make(logic_out, {x}), x_mark);
        unpack_vector(opr::CondExecMark::make(logic_out_dyn, {x}), x_mark_dyn);
        auto y = merge_one_out({x_mark + 1}, MergeMode::SUM),
             y_dyn = merge_one_out({x_mark_dyn + 1}, MergeMode::SUM);
        HostTensorND host_y;

        auto func = graph->compile({make_callback_copy(y_dyn, host_y)});
        auto updater = cg::static_infer::StaticInferUpdater::make();
        updater->add_dest({y.node(), cg::static_infer::DepType::VALUE});
        auto&& mgr = graph->static_infer_manager();

        for (size_t i = 0; i < nr_case; ++i) {
            size_t nr_one = 0;
            for (size_t j = 0; j < nr_input; ++j) {
                auto cur = (i >> j) & 1;
                host_pred->ptr<float>()[j] = cur;
                nr_one += cur;
            }

            updater->update();
            int vinfer = mgr.infer_value(y.node()).ptr<float>()[0];
            func->execute();
            int vy = host_y.ptr<float>()[0];
            ASSERT_EQ(checker(nr_one), vy) << "case=" << i;
            ASSERT_EQ(vy, vinfer) << "case=" << i;
        }
    };

    for (int inp = 1; inp < 5; ++inp) {
#define DO_RUN(mode, fn)                                    \
    do {                                                    \
        run(Mode::mode, inp, fn);                           \
        ASSERT_FALSE(Test::HasFailure())                    \
                << "failed on " << #mode << " inp=" << inp; \
    } while (0)

        DO_RUN(OR, [](int n) { return n != 0; });
        DO_RUN(AND, [inp](int n) { return n == inp; });
        DO_RUN(XOR, [](int n) { return n & 1; });

        DO_RUN(NOR, [](int n) { return n == 0; });
        DO_RUN(NAND, [inp](int n) { return n != inp; });
        DO_RUN(XNOR, [](int n) { return !(n & 1); });

#undef DO_RUN
    }

    static_assert(opr::CondExecPredLogical::Param::MODE_NR_MEMBER == 6,
                  "not all mode tested");
}

TEST(TestCondExec, Nested) {
    test_nested(false);
}

TEST(TestCondExec, NestedGrad) {
    test_nested(true);
}

TEST(TestCondExec, MergeSumDyn) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         cx0 = opr::MarkDynamicVar::make(make_one_cond(pred, x)) + 1.f,
         cx1 = opr::MarkDynamicVar::make(make_one_cond(pred - 1.f, x) + 2.f);
    ASSERT_THROW(merge_one_out({cx0, cx1}, MergeMode::SUM, 0, {}), GraphError);
    auto y = merge_one_out({cx0, cx1}, MergeMode::SUM, 0, {x.symshape()});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    auto run = [&](float k, float bias) {
        host_pred->ptr<float>()[0] = bias;
        HostTensorND expect;
        expect.copy_from(*host_x);
        auto px = expect.ptr<float>();
        for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it; ++i) {
            px[i] = (px[i] + bias) * k;
        }
        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_y);
    };

    run(1.f, 1.f);
    run(0.f, -1.f);
    run(1.f, 2.f);
}

TEST(TestCondExec, AddUpdateFwd) {
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x);

    host_pred->ptr<float>()[0] = 1;

    auto x = opr::SharedDeviceTensor::make(*graph, dev_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         xmark0 = make_one_cond(pred, x, 1, 0, true),
         xmark1 = make_one_cond(pred - 1, x, 1, 0, true),
         xmerge = merge_one_out({xmark0 + 1, xmark1 + 2}, MergeMode::EXACT_ONE),
         loss = opr::reduce_sum_sqr(xmerge, x.make_scalar(1)),
         gx = cg::grad(loss, x), xud = opr::AddUpdate::make(x, gx);

    auto func = graph->compile({{xud, {}}});

    auto run = [&](float bias) {
        host_pred->ptr<float>()[0] = bias;
        dev_x->copy_from(*host_x);
        func->execute();
        HostTensorND got, expect;
        got.copy_from(*dev_x).sync();
        expect.copy_from(*host_x);
        auto px = expect.ptr<float>();
        for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it; ++i) {
            px[i] += 2 * (px[i] + bias);
        }
        MGB_ASSERT_TENSOR_EQ(expect, got);
        if (bias == 1) {
            ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(xmark0));
        } else {
            ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(xmark1));
        }
    };

    run(1);
    run(2);
}

TEST(TestCondExec, CondAddUpdate) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x);

    host_pred->ptr<float>()[0] = 1;

    auto x = opr::SharedDeviceTensor::make(*graph, dev_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         xmark = make_one_cond(pred, x),
         xud = opr::AddUpdate::make(x, xmark * 1.3f);
    auto func = graph->compile({{xud, {}}});

    auto run = [&](float pred) {
        host_pred->ptr<float>()[0] = pred;
        dev_x->copy_from(*host_x);
        func->execute();
        HostTensorND got, expect;
        got.copy_from(*dev_x).sync();
        expect.copy_from(*host_x);
        if (pred == 1.f) {
            auto px = expect.ptr<float>();
            for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it;
                 ++i) {
                px[i] *= 2.3f;
            }
        }
        MGB_ASSERT_TENSOR_EQ(expect, got);
    };

    run(3);
    run(1);
    run(2);
}

TEST(TestCondExec, MultiCnMarkWaitPred) {
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto cn0 = host_x->comp_node(), cn1 = cn0.change_stream(1);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         pred_delayed = opr::Sleep::make(pred, 0.05, {}, cn1);
    SymbolVar ppv, y;
    unpack_vector(opr::CondExecPred::make(pred_delayed,
                                          {pred_delayed.make_scalar_dt(1.f)}),
                  ppv);
    unpack_vector(opr::CondExecMark::make(ppv, {x}, {}, cn0), y);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    host_pred->ptr<float>()[0] = 0;
    func->execute();
    ASSERT_TRUE(host_y.empty());
    host_pred->ptr<float>()[0] = 1;
    func->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
}

TEST(TestCondExec, MultiCnMergeWaitPred) {
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto cn0 = host_x->comp_node(), cn1 = cn0.change_stream(1),
         cn2 = cn0.change_stream(2);
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x),
              pred = opr::Host2DeviceCopy::make(*graph, host_pred), ppv0, ppv1;
    auto make_marked = [cn1, pred](SymbolVar x, float pv, SymbolVar& ppv) {
        SymbolVar y;
        unpack_vector(opr::CondExecPred::make(opr::Sleep::make(pred, 0.05),
                                              {pred.make_scalar_dt(pv)}),
                      ppv);
        unpack_vector(opr::CondExecMark::make(ppv, {x}, {}, cn1), y);
        return y;
    };
    SymbolVar y0 = make_marked(x, 1.f, ppv0) + 1.f,  // cn1
            y1 = make_marked(x, 2.f, ppv1) + 2.f,    // cn1
            z = opr::CondExecMerge::make({y0, y1},
                                         {1, MergeMode::SUM_COND_OUT})[0];
    HostTensorND host_z;
    z.node()->comp_node(cn2);  // change z to cn2
    z.node()->owner_opr()->on_output_comp_node_stream_changed();
    auto func = graph->compile({make_callback_copy(z, host_z)});

    SymbolVar z_ppv = z.node()->owner_opr()->input().back();
    check_waiting_spec(z_ppv, {ppv0});
    check_waiting_spec(z, {y0});

    host_pred->ptr<float>()[0] = 0;
    func->execute();
    ASSERT_TRUE(host_z.empty());

    auto run = [&](float bias) {
        host_pred->ptr<float>()[0] = bias;
        HostTensorND expect;
        expect.copy_from(*host_x);
        auto px = expect.ptr<float>();
        for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it; ++i) {
            px[i] += bias;
        }
        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_z);
    };

    run(1);
    run(2);
}

TEST(TestCondExec, InputWaitingForMerge) {
    using Elemwise = opr::Elemwise;
    auto cn0 = CompNode::load("xpux"), cn1 = cn0.change_stream(1);
    HostTensorGenerator<> gen;
    auto host_pred = gen({1}, cn0), host_x = gen({2, 3}, cn0);
    host_pred->ptr<float>()[0] = 0;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().seq_opt.enable_seq_comp_node_opt = false;
    auto pred = opr::Host2DeviceCopy::make_no_value_infer(*graph, host_pred);
    auto make_delayed_pred = [pred](CompNode cn) {
        return opr::MarkDynamicVar::make(opr::Sleep::make(pred, 0.02, {}, cn));
    };

    auto make_marked = [](SymbolVar x, SymbolVar pred, float key) -> SymbolVar {
        SymbolVar ppv;
        unpack_vector(opr::CondExecPred::make(pred, {pred.make_scalar(key)}),
                      ppv);
        SymbolVar xcond;
        unpack_vector(opr::CondExecMark::make(ppv, {x}, {},
                                              {pred.node()->comp_node()}),
                      xcond);
        return xcond;
    };
    auto make_merged = [cn0](const VarNodeArrayView& arr) -> SymbolVar {
        SymbolVar ret;
        for (size_t i = 0; i < arr.size(); ++i) {
            mgb_assert((i == 0) == (arr[i]->comp_node() == cn0));
        }
        unpack_vector(opr::CondExecMerge::make(
                              arr, {1, MergeMode::SUM_COND_OUT}, {}, cn0),
                      ret);
        return ret;
    };
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         x1 = opr::Copy::make(x, cn1), pred0 = make_delayed_pred(cn0),
         pred1 = make_delayed_pred(cn1),
         y0 = make_marked(x, pred0, 1) + 1,       // on cn0
            y10 = make_marked(x1, pred1, 2) + 2,  // on cn1
            y11 = make_marked(x, pred1, 3) + 3,   // on cn1
            ymgr = make_merged({y0, y10, y11}),   // on cn0
            z = Elemwise::make({x1, opr::Sleep::make(ymgr, 0.03)},
                               Elemwise::Mode::ADD,
                               cn0);  // (cn1, cn0) -> cn0
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});

    check_waiting_spec(ymgr, {y10});
    // provable ymgr is later than x1
    check_waiting_spec(z, {});

    auto run = [&](float pv) {
        *host_x = *gen({2 + static_cast<size_t>(pv), 5});
        host_pred->ptr<float>()[0] = pv;
        host_z = {};
        func->execute();
        if (pv < 1) {
            ASSERT_TRUE(host_z.empty());
            return;
        }
        HostTensorND expect;
        auto ptr = expect.copy_from(*host_x).ptr<float>();
        for (size_t i = 0, it = host_x->shape().total_nr_elems(); i < it; ++i) {
            ptr[i] = ptr[i] * 2 + pv;
        }
        MGB_ASSERT_TENSOR_EQ(expect, host_z);
    };
    run(2);
    run(1);
    run(3);
    run(2);
    run(-1);
}

TEST(TestCondExec, GradMultiReader) {
    // multiple readers of the grad wrt var, on multiple comp nodes
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_pred = gen({1}, cns[0]), host_x = gen({2, 3}, cns[0]);
    host_pred->ptr<float>()[0] = 0;
    auto copy1 = [&cns](SymbolVar x) { return opr::Copy::make(x, cns[1]); };
    auto pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         x = opr::Host2DeviceCopy::make(*graph, host_x),
         y0 = copy1(make_one_cond(pred, x, 1, 0, true)),
         y1 = copy1(make_one_cond(pred + 1, x * 2.f, 1, 0, true)),
         y2 = make_one_cond(copy1(pred) + 2, copy1(x) * 3.f, 1, 0, true),
         z = opr::Copy::make(merge_one_out({y0, y1, y2}, MergeMode::SUM),
                             cns[0]),
         loss = opr::reduce_sum_sqr(z, z.make_scalar(1)),
         gx = cg::grad(loss, x);
    ASSERT_TRUE(cg::is_static_var_value(z.node()));
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});

    auto run = [&](int pv, Maybe<float> coeff) {
        host_pred->ptr<float>()[0] = pv;
        host_gx = {};
        func->execute();
        if (!coeff.valid()) {
            ASSERT_TRUE(host_gx.empty());
            return;
        }
        HostTensorND expect;
        expect.copy_from(*host_x);
        auto ptr = expect.ptr<float>();
        auto c = coeff.val();
        c = c * c * 2;
        for (size_t i = 0, it = host_x->shape().total_nr_elems(); i < it; ++i) {
            ptr[i] = ptr[i] * c;
        }
        MGB_ASSERT_TENSOR_EQ(expect, host_gx);
    };

    run(-1, 3.f);
    run(0, 2.f);
    run(1, 1.f);
    run(2, None);
}

TEST(TestCondExec, SyncForMultiCN) {
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    HostTensorGenerator<> gen;
    auto host_pred = gen({1}, cns[0]), host_x = gen({2, 3}, cns[0]);
    host_pred->ptr<float>()[0] = 0;
    auto copy1 = [&cns](SymbolVar x) { return opr::Copy::make(x, cns[1]); };
    auto pred = opr::Host2DeviceCopy::make_no_value_infer(*graph, host_pred),
         x = opr::Host2DeviceCopy::make(*graph, host_x),
         y0 = make_one_cond(pred, x, 1),
         y1 = make_one_cond(copy1(pred) + 1, copy1(x) * 2.f),
         y2 = make_one_cond(copy1(pred) + 2, copy1(x) * 3.f),
         y12 = opr::Copy::make(merge_one_out({y1, y2}, MergeMode::SUM_COND_OUT),
                               cns[0]),
         z = merge_one_out({y12, y0}, MergeMode::EXACT_ONE),
         loss = opr::reduce_sum_sqr(z, z.make_scalar(1)),
         gx = cg::grad(loss, x);
    ASSERT_FALSE(cg::is_static_var_value(z.node()));
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});

    auto run = [&](int pv, Maybe<float> coeff) {
        host_pred->ptr<float>()[0] = pv;
        host_gx = {};
        opr::Sleep::sleep(cns[0], 0.1);  // sleep to delay h2d copy
        func->execute();
        if (!coeff.valid()) {
            ASSERT_TRUE(host_gx.empty());
            return;
        }
        HostTensorND expect;
        expect.copy_from(*host_x);
        auto ptr = expect.ptr<float>();
        auto c = coeff.val();
        c = c * c * 2;
        for (size_t i = 0, it = host_x->shape().total_nr_elems(); i < it; ++i) {
            ptr[i] = ptr[i] * c;
        }
        MGB_ASSERT_TENSOR_EQ(expect, host_gx);
    };

    run(-1, 3.f);
    run(0, 2.f);
    run(1, 1.f);
}

TEST(TestCondExec, AsyncCondAccess) {
    constexpr float SLEEP_TIME = 0.2;
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    graph->options().graph_opt_level = 0;
    auto allocator = std::make_shared<DynamicMemLeakChecker>();
    graph->set_device_memory_allocator(allocator);
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_pred = gen({1});
    auto cn1 = host_x->comp_node().change_stream(1);
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         xmark = make_one_cond(pred, x),
         xmark_delay = opr::Sleep::make(xmark, SLEEP_TIME, {}, cn1),
         xp1 = (x + 1).rename("xp1"),
         y = opr::Elemwise::make({xmark_delay + 2.3f, xp1},
                                 opr::Elemwise::Mode::ADD, cn1);

    host_pred->ptr<float>()[0] = 0;

    set_priority(xp1, 100);

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    check_waiting_spec(y, {xp1});
    ASSERT_FALSE(cg::is_static_var_storage(xp1.node()));

    RealTimer timer;
    func->execute().wait();
    ASSERT_TRUE(host_y.empty());
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    auto use_time = timer.get_secs();
    if (use_time >= SLEEP_TIME / 2) {
        mgb_log_warn("expect time [%f < %f], got %f", use_time, SLEEP_TIME / 2,
                     use_time);
    }
    ASSERT_EQ(0u, allocator->nr_alive());

    host_pred->ptr<float>()[0] = 1;
    func->execute().wait();
    use_time = timer.get_secs();
    if (use_time <= SLEEP_TIME) {
        mgb_log_warn("expect time [%f > %f], got %f", use_time, SLEEP_TIME,
                     use_time);
    }
    HostTensorND expect;
    graph->compile({make_callback_copy(x * 2 + 3.3f, expect)})->execute();
    MGB_ASSERT_TENSOR_EQ(expect, host_y);
    ASSERT_EQ(0u, allocator->nr_alive());
}

TEST(TestCondExec, VolatilePtr) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int;
    HostTensorND expect;
    auto host_pred = gen_int({1});
    auto dev_x = std::make_shared<DeviceTensorND>();
    auto assign = [&](int br) {
        host_pred->ptr<int>()[0] = br;
        expect = *gen({2, 3});
        auto hold = *dev_x;
        *dev_x = {};
        // ensure a different ptr
        dev_x->copy_from(expect).sync();
        auto p = expect.ptr<float>();
        for (size_t i = 0; i < 6; ++i) {
            p[i] = p[i] + (br == 0 ? 1.2f : 2.1f);
        }
    };

    assign(0);

    auto x = opr::VolatileSharedDeviceTensor::make(*graph, dev_x),
         pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         xc0 = make_one_cond(pred + 1, x), xc1 = make_one_cond(pred, x),
         y = merge_one_out({xc0 + 1.2f, xc1 + 2.1f},
                           MergeMode::EXACT_ONE_SAME_SHAPE);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    auto run = [&](int br) {
        assign(br);
        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_y);
        if (br == 0) {
            ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(xc0));
        } else {
            ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(xc1));
        }
    };

    run(0);
    run(1);
    run(1);
    run(0);
}

TEST(TestCondExec, MultiShape) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2}), host_d2 = gen({2}), host_d3 = gen({3});
    //! return y conditioned on shape of \p x equaling \p shp
    auto enable_if_shape = [](SymbolVar x, size_t shp) {
        auto y = make_one_cond(x.symshape() - static_cast<int>(shp - 1), x);
        // static shape inference is always performed regardless of cond
        // exec mark, so we add a reshape here to hint the true shape of y, to
        // ensure that shape inference of oprs depending on y could succeed
        // TODO: remove this if static infer considers execution mask
        y = y.reshape(TensorShape{shp});
        return y;
    };
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x),
              d2 = opr::Host2DeviceCopy::make(*graph, host_d2),
              d3 = opr::Host2DeviceCopy::make(*graph, host_d3),
              xc0 = enable_if_shape(x, 2) + d2,
              xc1 = enable_if_shape(x, 3) + d3,
              merged = merge_one_out({xc0, xc1}, MergeMode::EXACT_ONE),
              loss = opr::reduce_sum_sqr(merged, merged.make_scalar(1)),
              gx = cg::grad(loss, x);
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    auto check = [&](const std::shared_ptr<HostTensorND>& host_delta) {
        auto pd = host_delta->ptr<float>();
        HostTensorND expect;
        auto pe = expect.copy_from(*host_x).ptr<float>();
        for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it; ++i) {
            pe[i] = 2 * (pe[i] + pd[i]);
        }
        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_gx);
    };

    check(host_d2);
    *host_x = *gen({3});
    check(host_d3);

    *host_x = *gen({3});
    check(host_d3);

    *host_x = *gen({2});
    check(host_d2);
}

TEST(TestCondExec, EmptyShape) {
    HostTensorGenerator<> gen;
    auto host_pred = gen({1});
    host_pred->ptr<float>()[0] = 0;
    static auto empty_in_empty_out = [](SymbolVar x) {
        return x;
    };
    static auto empty_in_scalar_out = [](SymbolVar x) {
        return opr::Concat::make({x, x.make_scalar(1.f)}, 0);
    };
    static auto scalar_in_empty_out = [](SymbolVar x) {
        return opr::CondTake::make(x, x, {})[0]; // whether eq 0
    };
    { // EXACT_ONE
    auto graph = ComputingGraph::make();
    auto pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         empty = opr::ImmutableTensor::make(*graph, *gen({0})),
         scalar = pred.make_scalar(1.f),
         y0 = empty_in_empty_out(make_one_cond(pred + 1, empty)),
         y1 = empty_in_scalar_out(make_one_cond(pred, empty)),
         y2 = scalar_in_empty_out(make_one_cond(pred - 1, scalar)),
         z = merge_one_out({y0, y1, y2}, MergeMode::EXACT_ONE);

    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_TRUE(host_z.layout().is_empty());

    host_pred->ptr<float>()[0] = 1;
    func->execute();
    ASSERT_EQ(1.f, host_z.ptr<float>()[0]);

    host_pred->ptr<float>()[0] = 2;
    func->execute();
    ASSERT_TRUE(host_z.layout().is_empty());
    }
    { // SUM
    auto graph = ComputingGraph::make();
    host_pred->ptr<float>()[0] = 1;
    auto pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         empty = opr::ImmutableTensor::make(*graph, *gen({0})),
         scalar = pred.make_scalar(1.f),
         y0 = empty_in_empty_out(make_one_cond(pred, empty)),
         y1 = scalar_in_empty_out(make_one_cond(pred, scalar)),
         z = merge_one_out({y0, y1}, MergeMode::SUM);

    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_TRUE(host_z.layout().is_empty());
    }
    { // TAKE GRAD
    auto graph = ComputingGraph::make();
    host_pred->ptr<float>()[0] = 0;
    auto pred = opr::Host2DeviceCopy::make(*graph, host_pred),
         x = pred.make_scalar(1.2f),
         y0 = opr::CondTake::make(make_one_cond(pred + 1, x), pred, {})[0],
         y1 = make_one_cond(pred, x.make_scalar(3.4f)),
         z = merge_one_out({y0, y1}, MergeMode::EXACT_ONE),
         g = cg::grad(z, x);

    HostTensorND host_z, host_g;
    auto func = graph->compile({
        make_callback_copy(z, host_z), make_callback_copy(g, host_g)});
    func->execute();
    ASSERT_EQ(1.2f, host_z.ptr<float>()[0]);
    ASSERT_EQ(1.f, host_g.ptr<float>()[0]);

    host_pred->ptr<float>()[0] = 1;
    func->execute();
    ASSERT_EQ(3.4f, host_z.ptr<float>()[0]);
    ASSERT_EQ(0.f, host_g.ptr<float>()[0]);
    }
}
#endif  // MGB_ENABLE_COND_EXEC

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
