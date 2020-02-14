/**
 * \file src/opr/test/loop/record_internal.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/host_static_calc.h"

#include "megbrain/opr/loop.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/rand.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"

#include <cmath>

using namespace mgb;

using LoopDesc = opr::Loop::Desc;
using OutputMode = LoopDesc::OutputMode;

TEST(TestOprLoopRecordInternal, ImpureOprRNG) {
    constexpr int LOOP_TIME = 3;
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM>
        genx{1e-3, 1.5};

    auto host_x = genx({SIZE}),
         host_loss_p = gen({SIZE});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto desc_maker = [&](LoopDesc &desc) {
        auto xl = desc.add_input_assignable(x.fill_retain_dtype(1.f)),
             rand = opr::UniformRNG::make(opr::GetVarShape::make(xl));
        desc.assign(xl, xl * opr::pow(desc.add_input(x), rand * 2));
        desc.add_output(xl, OutputMode::LAST);
        desc.set_loop_condition(desc.get_counter_var() < LOOP_TIME);
    };
    auto y = opr::Loop::make(desc_maker)[0];
    auto loss = opr::Dot::make(
            y, opr::Host2DeviceCopy::make(*graph, host_loss_p)),
         gx = cg::grad(loss, x);
    HostTensorND host_gx, host_y;
    auto func = graph->compile({
            make_callback_copy(y, host_y),
            make_callback_copy(gx, host_gx)});
    func->execute();

    HostTensorND host_rand;
    func = graph->compile({make_callback_copy(
                opr::UniformRNG::make(opr::GetVarShape::make(x)), host_rand)});
    HostTensorND gx_expect, y_expect;
    gx_expect.copy_from(*host_x);
    y_expect.copy_from(*host_x);
    auto pgx = gx_expect.ptr<float>();
    memset(pgx, 0, sizeof(float) * SIZE);
    for (int i = 0; i < LOOP_TIME; ++ i) {
        func->execute();
        auto pr = host_rand.ptr<float>();
        for (size_t j = 0; j < SIZE; ++ j) {
            pgx[j] += pr[j] * 2;
        }
    }
    auto py = y_expect.ptr<float>(), plp = host_loss_p->ptr<float>();
    for (size_t i = 0; i < SIZE; ++ i) {
        float x = py[i], e = pgx[i];
        py[i] = std::pow(x, e);
        pgx[i] = plp[i] * e * std::pow(x, e - 1);
    }

    MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
    MGB_ASSERT_TENSOR_EQ(gx_expect, host_gx);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

