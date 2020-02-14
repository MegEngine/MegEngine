/**
 * \file src/opr/test/dnn/legacy_checker.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/test/helper.h"
#include "megbrain/tensor.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"

#include <cmath>

namespace mgb {
namespace opr {
namespace test{

template <class Opr, size_t nr_inputs, class Param = typename Opr::Param>
class MakeProxy {
    public:
        SymbolVar operator()(const SymbolVarArray &/*inputs*/,
                const Param &/*param*/) {
            mgb_throw(GraphError, "MakeProxy unimplemented");
        }
};

template <class Opr, class Param>
class MakeProxy <Opr, 1, Param> {
    public:
        SymbolVar operator()(const SymbolVarArray &inputs,
                const Param &param) {
            return Opr::make(inputs[0], param);
        }
};

template <class Opr, class Param>
class MakeProxy <Opr, 2, Param> {
    public:
        SymbolVar operator()(const SymbolVarArray &inputs,
                const Param &param) {
            return Opr::make(inputs[0], inputs[1], param);
        }
};

template <class Opr, class Param>
class MakeProxy <Opr, 3, Param> {
    public:
        SymbolVar operator()(const SymbolVarArray &inputs,
                const Param &param) {
            return Opr::make(inputs[0], inputs[1], inputs[2], param);
        }
};

class Device2HostCallback {
    public:
        Device2HostCallback(std::shared_ptr<HostTensorND> host):
            m_host{host}
        {}
        void operator()(const DeviceTensorND &device) {
            m_host->copy_from(device).sync();
        }
    private:
        std::shared_ptr<HostTensorND> m_host;
};

template <class Opr, size_t nr_inputs,
         class Param = typename Opr::Param>
class BackwardChecker {
    private:
        static const int MAX_LEN = 10;
    public:
        /*!
         * \param eps threshold to determine float equal
         * \param delta step for computing numeric grad
         * \param strict checking policy
         *      if strict is set, a single wrong grad would fail the test;
         *      otherwise it requires #fail_ratio wrong grads to fail.
         */
        BackwardChecker(TensorShapeArray in_shapes,
                const Maybe<Param> &param,
                float eps = 1e-3, float delta = 1e-3, bool strict = true,
                float fail_ratio = 0.1):
            m_in_shapes(in_shapes),
            gen(0.0f, 1.0f),
            m_param(param.val_with_default()),
            m_eps(eps), m_delta(delta),
            m_strict(strict),
            m_fail_ratio(fail_ratio)
            {
            }

        void run() {
            ASSERT_EQ(nr_inputs, m_in_shapes.size());
            auto graph = ComputingGraph::make();
            // gen input data
            for (auto &&shape : m_in_shapes) {
                m_inputs.push_back(gen(shape));
            }
            // gen grad data
            for (size_t i = 0; i < m_in_shapes.size(); ++i) {
                m_grads.push_back(std::make_shared<HostTensorND>(
                            CompNode::load("xpu0")));
            }
            // gen cost data
            m_cost = std::make_shared<HostTensorND>(CompNode::load("xpu0"));
            // gen input symbol
            SymbolVarArray symbol_inputs;
            for (auto &&input : m_inputs) {
                symbol_inputs.push_back(
                        mgb::opr::Host2DeviceCopy::make(*graph, input));
            }
            SymbolVar symbol_output = MakeProxy<Opr, nr_inputs>()(
                    symbol_inputs, m_param);
            TensorShape output_shape = symbol_output.node()->shape();
            TensorShape flatten_shape = {output_shape.total_nr_elems()};
            // gen weight data
            m_weight = gen(flatten_shape);
            // gen weight symbol
            SymbolVar symbol_weight = mgb::opr::Host2DeviceCopy::make(*graph,
                    m_weight);
            // gen flatten symbol
            SymbolVar symbol_flatten = symbol_output.reshape(flatten_shape);
            // gen cost symbol
            SymbolVar symbol_cost = mgb::opr::Dot::make(
                    symbol_weight, symbol_flatten);
            // gen grad symbols
            SymbolVarArray symbol_grads;
            for (auto &&symbol_input : symbol_inputs) {
                symbol_grads.push_back(cg::grad(symbol_cost, symbol_input));
            }
            // gen callbacks
            using Callback = cg::ComputingGraph::Callback;
            using OutputSpec = cg::ComputingGraph::OutputSpec;
            OutputSpec spec;
            for (size_t i = 0; i < symbol_grads.size(); ++i) {
                Callback cb = Device2HostCallback{m_grads[i]};
                spec.push_back({symbol_grads[i], cb});
            }
            auto func = graph->compile(spec);
            func->execute();
            // here all grads complete
            // recompile func to calculate cost
            func = graph->compile({{
                    symbol_cost, Device2HostCallback{m_cost}
                    }});
            func->execute();
            float before = m_cost->ptr<float>()[0];
            for (size_t in_idx = 0; in_idx < m_in_shapes.size(); ++in_idx) {
                std::shared_ptr<HostTensorND> input = m_inputs[in_idx];
                size_t len = input->shape().total_nr_elems();
                size_t corrupted = 0;
                size_t total_nr_elems = len;
                for (size_t offset = 0; offset < len && offset < MAX_LEN; ++offset) {
                    float &cur = input->ptr<float>()[offset];
                    float backup = cur;
                    float cur_delta = m_delta;
                    cur += cur_delta;
                    func->execute();
                    float after = m_cost->ptr<float>()[0];
                    float empirical_grad = (after - before) / cur_delta;
                    float mgb_grad = m_grads[in_idx]->
                        template ptr<float>()[offset];
                    float diff = std::abs(empirical_grad - mgb_grad);
                    if (m_strict) {
                        MGB_ASSERT_FLOAT_NEAR(empirical_grad, mgb_grad,
                                m_eps) << "differ at input(" << in_idx << "," <<
                            offset << ")" << std::endl;
                    } else {
                        if (diff > m_eps) {
                            ++corrupted;
                        }
                    }
                    cur = backup;
                }
                if (!m_strict) {
                    float corrupted_ratio = static_cast<float>(corrupted) /
                        total_nr_elems;
                    ASSERT_LE(corrupted_ratio, m_fail_ratio) << "input(" <<
                        in_idx << "): " << std::setprecision(2) <<
                        corrupted << "/" << total_nr_elems << "(" <<
                        corrupted_ratio << ") grads corrupted." << std::endl;
                }
            }
        }

    private:
        TensorShapeArray m_in_shapes;
        std::vector<std::shared_ptr<HostTensorND>> m_inputs;
        std::vector<std::shared_ptr<HostTensorND>> m_grads;
        std::shared_ptr<HostTensorND> m_cost;
        std::shared_ptr<HostTensorND> m_weight;

        HostTensorGenerator<> gen;
        Param m_param;
        float m_eps, m_delta;
        bool m_strict;
        float m_fail_ratio;
};


template <class Opr, size_t nr_inputs,
         class Param = typename Opr::Param>
class ForwardChecker {
     public:
         using RefFunc = void (*)(
                 const std::vector<std::shared_ptr<HostTensorND>> &in_tensor,
                 std::shared_ptr<HostTensorND> &out_tensor,
                 const Param &param);
         /*!
          * \param eps threshold to determine float equal
          * \param strict checking policy
          *      if strict is set, a single wrong grad would fail the test;
          *      otherwise it requires #fail_ratio wrong grads to fail.
          */
         ForwardChecker(TensorShapeArray in_shapes,
                 RefFunc ref_func,
                 const Maybe<Param> &param, float eps = 1e-5,
                 bool strict = true, float fail_ratio = 0.1):
             m_in_shapes(in_shapes),
             gen(1.0f, 1.0f),
             m_ref_func(ref_func),
             m_param(param.val_with_default()),
             m_eps(eps),
             m_strict(strict),
             m_fail_ratio(fail_ratio)
             {
             }

         void get_mgb_output()
         {
             auto graph = ComputingGraph::make();
             // generate SymbolVar for each input
             SymbolVarArray symbol_inputs;
             for (auto &&host_tensor : m_inputs) {
                 symbol_inputs.push_back(
                         mgb::opr::Host2DeviceCopy::make(*graph, host_tensor)
                         );
             }
             // generate output
             SymbolVar symbol_output = MakeProxy<Opr, nr_inputs>()(
                     symbol_inputs, m_param);
             m_mgb_output = std::make_shared<HostTensorND>(
                     CompNode::load("xpu0"));
             auto func = graph->compile({{
                     symbol_output, Device2HostCallback{m_mgb_output}
                     }});
             func->execute();
         }
         void get_ref_output()
         {
             m_ref_func(m_inputs, m_ref_output, m_param);
         }
         void run() {
             ASSERT_EQ(nr_inputs, m_in_shapes.size());

             // generate input data
             for (auto &&shape : m_in_shapes) {
                 m_inputs.push_back(gen(shape));
             }

             get_ref_output();
             get_mgb_output();

             TensorShape mgb_shape = m_mgb_output->shape();
             TensorShape ref_shape = m_ref_output->shape();
             ASSERT_TRUE(mgb_shape.eq_shape(ref_shape)) <<
                 "mgb_shape=" << mgb_shape.to_string() <<
                 "\nref_shape=" << ref_shape.to_string() << std::endl;

             size_t total_nr_elems = mgb_shape.total_nr_elems();
             ASSERT_GE(total_nr_elems, static_cast<size_t>(0));

             size_t corrupted = 0;
             for (size_t i = 0; i < total_nr_elems; ++i) {
                 float mgb_val = m_mgb_output->ptr<float>()[i];
                 float ref_val = m_ref_output->ptr<float>()[i];
                 float diff = std::abs(mgb_val - ref_val);
                 if (m_strict) {
                     MGB_ASSERT_FLOAT_NEAR(ref_val, mgb_val, m_eps)
                         << "differ at position " << i << std::endl;
                 } else {
                     if (diff > m_eps) {
                         ++corrupted;
                     }
                 }
             }
             if (!m_strict) {
                 float corrupted_ratio = static_cast<float>(corrupted) /
                     total_nr_elems;
                 ASSERT_LE(corrupted_ratio, m_fail_ratio) <<
                     corrupted << "/" << total_nr_elems << "(" <<
                     corrupted_ratio << ") values corrupted." <<
                     std::endl;
             }
         }

     private:
         TensorShapeArray m_in_shapes;
         HostTensorGenerator<> gen;

         // storages
         std::vector<std::shared_ptr<HostTensorND>> m_inputs;
         std::shared_ptr<HostTensorND> m_mgb_output;
         std::shared_ptr<HostTensorND> m_ref_output;

         RefFunc m_ref_func;
         Param m_param;
         float m_eps;
         bool m_strict;
         float m_fail_ratio;
};

} // namespace test
} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

