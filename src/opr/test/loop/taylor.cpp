/**
 * \file src/opr/test/loop/taylor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */


#include "./taylor.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/loop.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"

#include <cmath>

using namespace mgb;
using namespace mgb::test::loop;

namespace {
using OutputMode = opr::Loop::Desc::OutputMode;

constexpr float TAYLOR_MAX_ERR = 1e-6;

void test_taylor(thin_function<SymbolVar(SymbolVar)> tmaker,
        thin_function<float(float)> raw_f,
        thin_function<float(float)> raw_g) {

    constexpr float MAX_ERR = 1e-6;
    HostTensorGenerator<> gen;

    auto host_x = gen({7}), host_loss_p = gen({7});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         loss_p = opr::Host2DeviceCopy::make(*graph, host_loss_p),
         y = tmaker(x),
         loss = opr::Dot::make(y, loss_p),
         grad = cg::grad(loss, x);
    HostTensorND host_y, host_grad;
    auto func = graph->compile({make_callback_copy(y, host_y),
            make_callback_copy(grad, host_grad)});

    for (size_t SIZE: {1, 23}) {
        host_x->copy_from(*gen({SIZE}));
        host_loss_p->copy_from(*gen({SIZE}));
        func->execute();
        ASSERT_EQ(host_x->shape(), host_y.shape());
        ASSERT_EQ(host_x->shape(), host_grad.shape());
        for (size_t i = 0; i < SIZE; i ++) {
            auto x = host_x->ptr<float>()[i],
                 loss_p = host_loss_p->ptr<float>()[i];
            MGB_ASSERT_FLOAT_NEAR(raw_f(x), host_y.ptr<float>()[i],
                    MAX_ERR * 10)
                << ssprintf("i: %zd; x: %.4f", i, x);
            MGB_ASSERT_FLOAT_NEAR(raw_g(x) * loss_p, host_grad.ptr<float>()[i],
                    MAX_ERR * 10)
                << ssprintf("i: %zd; x: %.4f; loss_p: %.4f", i, x, loss_p);
        }
    }
}


/*!
 * \brief calc a complex expression involving two vars
 *
 * z = sum(e^(x/k) * sin(y^2 + (x + k*y)*sum(p, 1<=p<=k))/exp(0.3*k) , k >= 1)
 */
void test_two_var_coupled(
        thin_function<SymbolVar(SymbolVar)> sym_exp,
        thin_function<SymbolVar(SymbolVar)> sym_sin) {
    set_rand_seed(19931102);

    constexpr float MAX_TERM_VAL = 1e-6, MAX_ERR = 1e-3;

    HostTensorGenerator<> gen;
    auto host_x = gen({1}), host_y = gen({1}),
         host_loss_p = gen({1});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y"),
         loss_p = opr::Host2DeviceCopy::make(*graph, host_loss_p).rename("lp");

    auto desc_maker = [&](opr::Loop::Desc &desc) {
        auto t1_cup = desc.add_input_assignable(x + y),
             t1_cdown = desc.add_input_assignable(x.make_scalar(1)),
             k = desc.get_counter_var() + 1,
             t0 = sym_exp(desc.add_input(x) / k).rename("t0"),
             t1 = sym_sin(desc.add_input(y * y) +
                     t1_cup / t1_cdown).rename("t1"),
             term = (t0 * t1 / sym_exp(k * 0.3f)).rename("term"),
             err_elem = (term * term >
                     MAX_TERM_VAL * MAX_TERM_VAL).rename("err"),
             result = desc.add_input_assignable(
                     x.fill_retain_dtype(0)).rename("sum"),
             result_next = result + term;
        desc.assign(result, result_next);
        desc.assign(t1_cup, t1_cup + desc.add_input(y));
        desc.assign(t1_cdown, t1_cdown + k + 1);
        desc.set_loop_condition(opr::Dot::make(err_elem, err_elem));
        desc.add_output(result_next, OutputMode::LAST);
    };

    auto z = opr::Loop::make(desc_maker).at(0),
         loss = opr::Dot::make(z, loss_p),
         gx = cg::grad(loss, x),
         gy = cg::grad(loss, y);

    HostTensorND host_z, host_gx, host_gy;
    auto func = graph->compile({
            make_callback_copy(z, host_z),
            make_callback_copy(gx, host_gx),
            make_callback_copy(gy, host_gy)});

    for (size_t SIZE: {1, 23}) {
        host_x->copy_from(*gen({SIZE}));
        host_y->copy_from(*gen({SIZE}));
        host_loss_p->copy_from(*gen({SIZE}));
        auto px = host_x->ptr<float>(), py = host_y->ptr<float>();

        func->execute();
        ASSERT_EQ(host_x->shape(), host_z.shape());
        ASSERT_EQ(host_x->shape(), host_gx.shape());
        ASSERT_EQ(host_x->shape(), host_gy.shape());
        auto pz = host_z.ptr<float>(), pgx = host_gx.ptr<float>(),
             pgy = host_gy.ptr<float>();
        std::vector<float> vraw_z(SIZE), vraw_gx(SIZE), vraw_gy(SIZE);

        for (float k = 1; ; k ++) {
            bool done = true;
            for (size_t i = 0; i < SIZE; i ++) {
                auto x = px[i], y = py[i];
                float &raw_z = vraw_z[i],
                      &raw_gx = vraw_gx[i],
                      &raw_gy = vraw_gy[i];
                float t0 = exp(x / k),
                      cdown = k * (k + 1) / 2,
                      t1_inner = y*y + (x + k*y) / cdown,
                      t1 = sin(t1_inner),
                      gt1 = cos(t1_inner),
                      f = 1 / exp(0.3 * k),
                      term = t0 * t1 * f;
                raw_z += term;
                raw_gx += (1/k * t0*t1 + 1/cdown * t0*gt1) * f;
                raw_gy += (2*y + k/cdown) * t0*gt1 *f;
                if (fabs(term) > MAX_TERM_VAL)
                    done = false;
            }
            if (done)
                break;
        }

        for (size_t i = 0; i < SIZE; i ++) {
            auto x = px[i], y = py[i],
                 raw_z = vraw_z[i], raw_gx = vraw_gx[i], raw_gy = vraw_gy[i];
            auto lp = host_loss_p->ptr<float>()[i];
            MGB_ASSERT_FLOAT_NEAR(raw_z, pz[i], MAX_ERR) <<
                ssprintf("failed at %zd/%zd: x=%g y=%g lp=%g",
                        i, SIZE, x, y, lp);
            MGB_ASSERT_FLOAT_NEAR(raw_gx * lp, pgx[i], MAX_ERR) <<
                ssprintf("failed at %zd/%zd: x=%g y=%g lp=%g",
                        i, SIZE, x, y, lp);
            MGB_ASSERT_FLOAT_NEAR(raw_gy * lp, pgy[i], MAX_ERR) <<
                ssprintf("failed at %zd/%zd: x=%g y=%g lp=%g",
                        i, SIZE, x, y, lp);

        }
    }
}

} // anonymous namespace

/*!
 *\brief calc sin(x) = sum((-1)^k * x^(1+2k) / (1+2k)!, k >= 0)
 */
SymbolVar mgb::test::loop::sin_by_taylor(SymbolVar x) {
    auto desc_maker = [x](opr::Loop::Desc &desc) {
        auto term = desc.add_input_assignable(x).rename("term"),
             x_sqr_neg = desc.add_input(-x * x).rename("x_sqr_neg"),
             err_elem = term * term > TAYLOR_MAX_ERR * TAYLOR_MAX_ERR,
             err = opr::Dot::make(err_elem, err_elem);

        desc.assign(term, term * x_sqr_neg /
                ((desc.get_counter_var() * 2 + 2) *
                 (desc.get_counter_var() * 2 + 3)));

        desc.set_loop_condition(err * (desc.get_counter_var() < 100));

        desc.add_output(term, OutputMode::SUM);
    };
    return opr::Loop::make(desc_maker).at(0);
}

/*!
 *\brief calc exp(x) = sum(x^k / k!, k >= 0)
 */
SymbolVar mgb::test::loop::exp_by_taylor(SymbolVar x) {
    auto desc_maker = [x](opr::Loop::Desc &desc) {
        auto term = desc.add_input_assignable(
                x.fill_retain_dtype(1)).rename("term"),
             err_elem = term * term > TAYLOR_MAX_ERR * TAYLOR_MAX_ERR,
             err = opr::Dot::make(err_elem, err_elem);

        desc.assign(term, term * desc.add_input(x) /
                (desc.get_counter_var() + 1));

        desc.set_loop_condition(err * (desc.get_counter_var() < 100));

        desc.add_output(term, OutputMode::SUM);
    };
    return opr::Loop::make(desc_maker).at(0);
}

// fails in msvc without function pointer
typedef float (*math_fp)(float);
TEST(TestOprLoop, TaylorSin) {
    math_fp s = std::sin, c = std::cos;
    test_taylor(sin_by_taylor, s, c);
}

TEST(TestOprLoop, TaylorExp) {
    math_fp e = std::exp;
    test_taylor(exp_by_taylor, e, e);
}

TEST(TestOprLoop, Coupled) {
    test_two_var_coupled(
            [](SymbolVar x){return opr::exp(x);},
            [](SymbolVar x){return opr::sin(x);}
            );
}

TEST(TestOprLoop, CoupledNested) {
    test_two_var_coupled(exp_by_taylor, sin_by_taylor);
}

/*!
 * sum(sin(x / k), 1 <= k <= LOOP_TIME)
 */
TEST(TestOprLoop, TaylorSinNested) {
    constexpr size_t SIZE = 23, LOOP_TIME = 2;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({SIZE});
    host_x->ptr<float>()[0] = 0;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");
    auto desc_maker = [&](opr::Loop::Desc &desc) {
        auto lx = desc.add_input(x).rename("lx");
        auto k = (desc.get_counter_var() + 1).rename("k");
        auto term = sin_by_taylor(lx / k).rename("term");
        desc.add_output(term, OutputMode::SUM);
        desc.set_loop_condition(k < int(LOOP_TIME));
    };
    auto y = opr::Loop::make(desc_maker).at(0).rename("y"),
         loss = opr::Reduce::make(
                 y, {opr::Reduce::Mode::SUM, 0}).rename("loss");
    HostTensorND host_y, host_gx;
    auto func = graph->compile({
            make_callback_copy(cg::grad(loss, x), host_gx),
            make_callback_copy(y, host_y)});
    func->execute();

    ASSERT_EQ(host_x->shape(), host_y.shape());
    ASSERT_EQ(host_x->shape(), host_gx.shape());

    auto ptr_x = host_x->ptr<float>(), ptr_y = host_y.ptr<float>(),
         ptr_gx = host_gx.ptr<float>();
    for (size_t i = 0; i < SIZE; i ++) {
        float x = ptr_x[i], y_expect = 0, gx_expect = 0;
        for (size_t j = 1; j <= LOOP_TIME; j ++) {
            y_expect += sin(x / j);
            gx_expect += cos(x / j) / j;
        }
        MGB_ASSERT_FLOAT_NEAR(y_expect, ptr_y[i], TAYLOR_MAX_ERR * 50) <<
            ssprintf("y failed at %zd: x=%g", i, x);
        MGB_ASSERT_FLOAT_NEAR(gx_expect, ptr_gx[i], TAYLOR_MAX_ERR * 50) <<
            ssprintf("gx failed at %zd: x=%g", i, x);
    }
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

