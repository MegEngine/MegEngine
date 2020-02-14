/**
 * \file src/tensorrt/test/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/gopt/framework.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/rand.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/tensorrt/opr_replace.h"
#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/basic_arith.h"

using namespace mgb;
using namespace opr;
using namespace tensorrt;

void TrtReplaceChecker::ensure_init_graph() {
    if (m_trt_y.node())
        return;

    SymbolVarArray inputs(m_nr_input);
    for (size_t i = 0; i < m_nr_input; ++i) {
        if (m_mark_inp_const.count(i)) {
            inputs[i] =
                    opr::SharedDeviceTensor::make(*m_graph, *m_inputs_val[i])
                            .rename(ssprintf("inp%zu", i));
        } else {
            inputs[i] = opr::Host2DeviceCopy::make(*m_graph, m_inputs_val[i])
                                .rename(ssprintf("inp%zu", i));
        }

        auto dt = m_idx2dtype.find(i);
        if (dt != m_idx2dtype.end()) {
            inputs[i] = opr::TypeCvt::make(inputs[i], dt->second);
        }
    }
    m_truth_y = m_exp_func(inputs);

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .apply({{m_truth_y}})
                          .endpoint_vars(),
                  m_trt_y);

    size_t nr_trt_opr = 0;
    cg::DepOprIter{[&nr_trt_opr, this](cg::OperatorNodeBase* opr) {
        if (opr->same_type<TensorRTOpr>()) {
            ++nr_trt_opr;
        }
    }}
            .add(m_trt_y.node());
    mgb_assert(nr_trt_opr >= 1);

    ComputingGraph::OutputSpec outspec(2);
    outspec[0] =
            make_callback_copy(m_truth_y, std::get<0>(m_output_val), false);
    outspec[1] = make_callback_copy(m_trt_y, std::get<1>(m_output_val), false);

    m_graph->options().graph_opt.tensorrt = false;
    m_func = m_graph->compile(outspec);
}

TrtReplaceChecker& TrtReplaceChecker::run(
        const TensorShapeArray& input_shapes) {
    if (::testing::Test::HasFailure()) {
        return *this;
    }
    mgb_assert(input_shapes.size() == m_nr_input);
    if (m_inputs_val.empty()) {
        m_inputs_val.resize(m_nr_input);
        for (size_t i = 0; i < m_nr_input; ++i) {
            auto rng_gen = m_idx2rng_gen.find(i);
            if (rng_gen != m_idx2rng_gen.end()) {
                m_inputs_val[i] = rng_gen->second->operator()(input_shapes[i]);
            } else
                m_inputs_val[i] = m_input_gen(input_shapes[i]);
        }
    } else {
        for (size_t i = 0; i < m_nr_input; ++i) {
            *m_inputs_val[i] = *m_input_gen(input_shapes[i]);
        }
    }

    ensure_init_graph();
    m_func->execute().wait();
    auto chk = [this]() {
        MGB_ASSERT_TENSOR_NEAR(std::get<0>(m_output_val),
                               std::get<1>(m_output_val), m_epsilon);
    };
    chk();
    return *this;
}

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
