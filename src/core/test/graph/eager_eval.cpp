/**
 * \file src/core/test/graph/eager_eval.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <memory>
#include "megbrain/graph/symbol_var.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/comp_node_env.h"

#include "megbrain/tensor.h"
#include "megbrain/test/helper.h"

using namespace mgb;

namespace {
class TestGraphEagerEvalBase : public ::testing::Test {
    std::shared_ptr<ComputingGraph> m_graph_eager;
protected:
    auto graph_eager() {
        if (!m_graph_eager) {
            m_graph_eager = ComputingGraph::make();
            m_graph_eager->options().eager_evaluation = true;
        }
        return m_graph_eager.get();
    }
    void make_graph_normal(
        const SmallVector<std::shared_ptr<HostTensorND>>& inputs,
        const thin_function<SymbolVarArray(const SymbolVarArray&)>& make_graph,
        std::shared_ptr<cg::AsyncExecutable>& func_normal,
        SmallVector<HostTensorND>& out_normal) {
        auto graph_normal = ComputingGraph::make();
        SymbolVarArray inp_normal(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++ i) {
            inp_normal[i] = opr::Host2DeviceCopy::make(
                    *graph_normal, inputs[i]);
        }
        auto symout_normal = make_graph(inp_normal);
        out_normal.resize(symout_normal.size());
        ComputingGraph::OutputSpec out_spec_normal(symout_normal.size());
        for (size_t i = 0; i < symout_normal.size(); ++i) {
            out_spec_normal[i] =
                    make_callback_copy(symout_normal[i], out_normal[i]);
        }
        func_normal = graph_normal->compile(out_spec_normal);
    }
    SymbolVarArray make_graph_eager(
        const SmallVector<std::shared_ptr<HostTensorND>>& inputs,
        const thin_function<SymbolVarArray(const SymbolVarArray&)>& make_graph,
        bool allow_inp_fwd) {
        SymbolVarArray inp_eager(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (allow_inp_fwd) {
                inp_eager[i] = opr::Host2DeviceCopy::make(
                        *graph_eager(), inputs[i]);
            } else {
                inp_eager[i] = opr::Host2DeviceCopy::make_no_fwd(
                        *graph_eager(), inputs[i]);
            }
        }
        return make_graph(inp_eager);
    }
    inline void check_results(const SymbolVarArray& symout_eager,
        const SmallVector<HostTensorND>& expected) {
        mgb_assert(expected.size() == symout_eager.size());
        for (size_t i = 0; i < symout_eager.size(); ++i) {
            HostTensorND val;
            val.copy_from(symout_eager[i].eager_eval_get_value()).sync();
            MGB_ASSERT_TENSOR_EQ(expected[i], val);
        }
    }
};

class TestGraphEagerEval : public TestGraphEagerEvalBase {
protected:
    void run_eager_eval_test(
        const SmallVector<std::shared_ptr<HostTensorND>>& inputs,
        const thin_function<SymbolVarArray(const SymbolVarArray&)>& make_graph,
        bool allow_inp_fwd = true) {

        std::shared_ptr<cg::AsyncExecutable> func_normal;
        SmallVector<HostTensorND> out_normal;
        make_graph_normal(inputs, make_graph, func_normal, out_normal);
        func_normal->execute();
        auto symout_eager = make_graph_eager(inputs, make_graph, allow_inp_fwd);
        check_results(symout_eager, out_normal);
    }
};

class TestGraphEagerReeval : public TestGraphEagerEvalBase {
protected:
    using InputGenerator =
        thin_function<std::shared_ptr<HostTensorND>(const TensorShape&)>;
    struct TestSpec {
        SmallVector<TensorShape> shapes;
        int nr_oprs_delta; // set to a negative value to skip check
    };
    void run_eager_reeval_test(
        const SmallVector<TestSpec>& specs,
        const thin_function<SymbolVarArray(const SymbolVarArray&)>& make_graph,
        const SmallVector<InputGenerator>& generators = {}) {

        size_t nr_iter = specs.size();
        mgb_assert(nr_iter);
        size_t nr_inputs = specs[0].shapes.size();
        mgb_assert(nr_inputs);
        bool use_glob_generator = generators.empty();
        HostTensorGenerator<> gen_glob;
        auto gen = [&](size_t iter, size_t idx) {
            auto &&shape = specs[iter].shapes[idx];
            if (use_glob_generator) {
                return gen_glob(shape);
            }
            return generators.at(idx)(shape);
        };

        SmallVector<std::shared_ptr<HostTensorND>> inputs(nr_inputs);
        for (size_t i = 0; i < nr_inputs; ++ i) {
            inputs[i] = gen(0, i);
        }

        std::shared_ptr<cg::AsyncExecutable> func_normal;
        SmallVector<HostTensorND> out_normal;
        make_graph_normal(inputs, make_graph, func_normal, out_normal);

        int prev_nr_oprs = 0, cur_nr_oprs, nr_oprs_delta = -1;
        for (size_t i = 0; i < nr_iter; ++ i) {
            if (i) {
                auto &&spec = specs[i];
                nr_oprs_delta = spec.nr_oprs_delta;
                mgb_assert(spec.shapes.size() == nr_inputs);
                for (size_t j = 0; j < nr_inputs; ++ j) {
                    auto host_val = gen(i, j);
                    inputs[j]->copy_from(*host_val).sync();
                }
            }
            func_normal->execute();
            auto symout_eager = make_graph_eager(inputs, make_graph, false);
            check_results(symout_eager, out_normal);
            cur_nr_oprs = graph_eager()->nr_oprs_in_graph();
            if (nr_oprs_delta >= 0) { // skip first execution
                ASSERT_EQ(nr_oprs_delta + prev_nr_oprs, cur_nr_oprs);
            }
            prev_nr_oprs = cur_nr_oprs;
        }
    }
};

MGB_DEFINE_OPR_CLASS(EmptyShapeOpr,
                           cg::SingleCNOutshapePureByInshapeOprBase) // {
    bool m_allow_empty;

public:
    EmptyShapeOpr(VarNode* input, bool allow_empty)
            : Super{input->owner_graph(),
                    OperatorNodeConfig{},
                    "empty_shape",
                    {input}},
              m_allow_empty{allow_empty} {
        add_input({input});
        add_output(None)->dtype(dtype::Byte());
        add_equivalence_component<PODHash<bool>>(&m_allow_empty);
    }

    static SymbolVar make(SymbolVar input, bool allow_empty) {
        return input.insert_single_output_opr<EmptyShapeOpr>(input.node(),
                                                             allow_empty);
    }

private:
    void scn_do_execute() override {}

    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override {
        out_shape[0] = {1};
    }

    NodeProp* do_make_node_prop() const override {
        auto ret = Super::do_make_node_prop();
        if (m_allow_empty) {
            ret->add_dep_type_existing_var(
                    input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
        }
        return ret;
    }
};  // namespace
MGB_DYN_TYPE_OBJ_FINAL_IMPL(EmptyShapeOpr);

}  // anonymous namespace

TEST_F(TestGraphEagerEval, APlusB) {
    HostTensorGenerator<> gen;
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        return {inputs[0] + inputs[1]};
    };
    run_eager_eval_test({gen({2, 8}), gen({2, 1})}, make_graph);
}

#if MGB_ENABLE_EXCEPTION
TEST_F(TestGraphEagerEval, Exception) {
    class Exc {};
    HostTensorGenerator<> gen;
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto cb = [](DeviceTensorND&) { mgb_throw_raw(Exc{}); };
        return {inputs[0] + opr::CallbackInjector::make(inputs[1], cb)};
    };
    ASSERT_THROW(run_eager_eval_test({gen({2, 8}), gen({2, 1})}, make_graph),
                 Exc);
}
#endif

TEST_F(TestGraphEagerEval, MultiCn) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto make_graph = [&](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto i1 = opr::Copy::make(opr::Sleep::make(inputs[1], 0.2), cns[0]);
        return {inputs[0] + i1};
    };
    run_eager_eval_test({gen({2, 8}, cns[0]), gen({2, 1}, cns[1])}, make_graph);
}

TEST_F(TestGraphEagerEval, NonContig) {
    HostTensorGenerator<> gen;
    std::shared_ptr<ComputingGraph> graph_refhold;
    SymbolVar chk_x, chk_sub0, chk_sub1;
    auto make_graph = [&](const SymbolVarArray& inputs) -> SymbolVarArray {
        using AIdx = opr::indexing::AxisIndexer;
        auto x = inputs[0], w = inputs[1],
             xsub0 = opr::Subtensor::make(
                     x, {AIdx::make_interval(1, x.make_scalar(1), None, None)}),
             xsub1 = opr::Subtensor::make(
                     xsub0,
                     {AIdx::make_interval(1, None, x.make_scalar(-1), None)});
        if (x.node()->owner_graph()->options().eager_evaluation) {
            chk_x = x;
            chk_sub0 = xsub0;
            chk_sub1 = xsub1;
            graph_refhold = x.node()->owner_graph()->shared_from_this();
        }
        return {opr::Convolution::make(xsub1, w)};
    };
    run_eager_eval_test({gen({5, 5, 6, 7}), gen({4, 3, 2, 2})}, make_graph,
                        false);
    auto x0 = chk_x.eager_eval_get_value().raw_ptr(),
         x1 = x0 + chk_x.eager_eval_get_value().layout().span().dist_byte();
    auto chk_range = [&](SymbolVar var) {
        auto ptr = var.eager_eval_get_value().raw_ptr();
        return ptr >= x0 && ptr < x1;
    };
    ASSERT_TRUE(chk_range(chk_sub0));
    ASSERT_FALSE(chk_range(chk_sub1));
}

TEST_F(TestGraphEagerEval, DynShape) {
    HostTensorGenerator<> gen;
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto i0 = opr::MarkDynamicVar::make(inputs[0]),
             i1 = opr::MarkDynamicVar::make(inputs[1]),
             tmp = i0 * i1;
        auto tmp1 = i0 * i1; // dedup
        EXPECT_EQ(tmp, tmp1);
        return {i0 * i1 + opr::GetVarShape::make(i0, 1)};
    };
    run_eager_eval_test({gen({2, 8}), gen({2, 1})}, make_graph);
}

TEST_F(TestGraphEagerEval, DynValueNeeded) {
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int{-5, 5};
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto i0 = inputs[0], i1 = opr::MarkDynamicVar::make(inputs[1]);
        using S = opr::Subtensor;
        auto o0 = S::make(i0, {S::AxisIndexer::make_index(0, i1)}),
             o1 = S::make(i0, {S::AxisIndexer::make_index(0, i1 + 1)});
        return {o0, o1};
    };
    run_eager_eval_test({gen({8, 2}), gen_int({1})}, make_graph);
}

TEST_F(TestGraphEagerEval, Grad) {
    HostTensorGenerator<> gen;
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto x = inputs[0], w = inputs[1], y = opr::MatrixMul::make(x, w),
             loss = opr::reduce_sum_sqr(y, y.make_scalar(1)) /
                    opr::TypeCvt::make(opr::GetVarShape::make(y, 0), dtype::Float32());
        return cg::grad(loss, {x, w});
    };
    run_eager_eval_test({gen({123, 321}), gen({321, 345})}, make_graph);
}

TEST_F(TestGraphEagerEval, EagerGradWithoutDep) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("xpu0");
    auto x = opr::Host2DeviceCopy::make(*graph_eager(), gen({32, 16}));
    auto w1 = opr::SharedDeviceTensor::make(
            *graph_eager(), std::make_shared<DeviceTensorND>(cn, TensorShape{16, 128}));
    auto b1 = opr::SharedDeviceTensor::make(
            *graph_eager(), std::make_shared<DeviceTensorND>(cn, TensorShape{128}));
    auto fc1 = opr::MatrixMul::make(x, w1) + b1;
    auto w2 = opr::SharedDeviceTensor::make(
            *graph_eager(), std::make_shared<DeviceTensorND>(cn, TensorShape{128, 1}));
    auto b2 = opr::SharedDeviceTensor::make(
            *graph_eager(), std::make_shared<DeviceTensorND>(cn, TensorShape{1}));
    auto fc2 = opr::MatrixMul::make(fc1, w2) + b2;
    auto loss = opr::reduce_sum(fc2, fc2.make_scalar(1));
    auto symout_eager = cg::grad(loss, {b1, b2});
    SmallVector<std::shared_ptr<HostTensorND>> inputs{};
    for (size_t i = 0; i < symout_eager.size(); ++i) {
        HostTensorND val;
        val.copy_from(symout_eager[i].eager_eval_get_value()).sync();
    }
}

TEST_F(TestGraphEagerEval, VarRecvInfo) {
    HostTensorGenerator<> gen;
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto y = inputs[0] + inputs[1];
        auto og = y.node()->owner_graph();
        auto chk = [&]() {
            auto&& info = og->var_receiver_in_current_comp_seq(y.node());
            ASSERT_TRUE(info.value_needed());
            ASSERT_TRUE(info.is_empty_allowed());
        };
        if (og->options().eager_evaluation) {
            chk();
        }
        return {y};
    };
    run_eager_eval_test({gen({2, 8}), gen({2, 1})}, make_graph);
}

TEST_F(TestGraphEagerEval, EmptyShape) {
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 0});
    auto graph = ComputingGraph::make();
    graph->options().eager_evaluation = true;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    EmptyShapeOpr::make(x, true);
    ASSERT_THROW(EmptyShapeOpr::make(x, false), MegBrainError);
}

TEST_F(TestGraphEagerEval, Compile) {
    HostTensorGenerator<> gen;
    constexpr size_t length = 42;
    auto host_x = gen({42});
    auto normal_graph = ComputingGraph::make();

    int flag = 0;
    auto cb = [&](DeviceTensorND &) {
        ++ flag;
    };

    auto x = opr::Host2DeviceCopy::make(*graph_eager(), host_x),
         y = x + 1, z = y * 2,
         x_cb = opr::CallbackInjector::make(x, cb);
    graph_eager()->options().extra_vardeps[z.node()].push_back(x_cb.node());

    auto output = cg::replace_vars_comp_graph({z}, normal_graph.get());
    HostTensorND host_res;
    auto func = normal_graph->compile({
        make_callback_copy(output[0], host_res)
    });
    func->execute().wait();
    for (size_t i = 0; i < length; ++ i) {
        MGB_ASSERT_FLOAT_EQ((host_x->ptr<float>()[i] + 1) * 2,
                host_res.ptr<float>()[i]);
    }
    ASSERT_EQ(flag, 2);
}

TEST_F(TestGraphEagerReeval, Basic) {
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto x = inputs[0] / 2.f, y = x + 1.f, y0 = x + 1.f, z = y * 2.f;
        EXPECT_EQ(y, y0);
        return {z};
    };
    size_t nr_execs = 0;
    auto callback = [&nr_execs](const cg::event::OprExecStart &e) {
        ++ nr_execs;
    };
    auto handle = graph_eager()->event().register_receiver<cg::event::OprExecStart>(callback);
    size_t iter = 3;
    SmallVector<TestSpec> specs(3, {{{42}}, 0});
    run_eager_reeval_test(specs, make_graph);
    // 2 const-src: immutable[1.0], immutable[2.0]
    // 1 mutable-src: DataProvider[x]
    // 3 mid-node: x = inputs[i] / 2, y = x + 1, z = y * 2
    ASSERT_EQ(nr_execs, iter * 4 + 2);
}

TEST_F(TestGraphEagerReeval, DynShape) {
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto x = inputs[0],
             y = opr::MarkDynamicVar::make(x),
             z = opr::GetVarShape::make(y, 0) + x * 2;
        return {z};
    };
    SmallVector<TestSpec> specs;
    for (size_t i = 0; i < 5; ++ i) {
        specs.push_back({{{42 + (i & 3)}}, 0});
    }
    run_eager_reeval_test(specs, make_graph);
}

TEST_F(TestGraphEagerReeval, MultiCn) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto make_graph = [&](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto i1 = opr::Copy::make(opr::Sleep::make(inputs[1], 0.2), cns[0]);
        return {inputs[0] + i1};
    };
    SmallVector<TestSpec> specs(3, {{{2, 8}, {2, 1}}, 0});
    using namespace std::placeholders;
    run_eager_reeval_test(specs, make_graph,
        {std::bind(gen, _1, cns[0]), std::bind(gen, _1, cns[1])});
}

TEST_F(TestGraphEagerReeval, Grad) {
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto x = inputs[0], w = inputs[1], y = opr::MatrixMul::make(x, w),
             loss = opr::reduce_sum_sqr(y, y.make_scalar(1)) /
                    opr::TypeCvt::make(opr::GetVarShape::make(y, 0), dtype::Float32());
        return cg::grad(loss, {x, w});
    };
    SmallVector<TestSpec> specs(3, {{{123, 321}, {321, 345}}, 0});
    for (size_t i = 0; i < 5; ++ i) { // DynShape
        specs.push_back({{{123 + (i & 3), 321}, {321, 345}}, 0});
    }
    run_eager_reeval_test(specs, make_graph);
}

TEST_F(TestGraphEagerReeval, LayoutConstraint) {
    auto make_graph = [](const SymbolVarArray& inputs) -> SymbolVarArray {
        auto x = inputs[0], w = inputs[1], y = opr::MatrixMul::make(x, w),
             // when query for gradient a broadcast from shape(1) to shape(n, m)
             // would generate and it would be used as input to MatrixMul which
             // require all input layouts are contiguous.
             loss = opr::reduce_sum(y, y.make_scalar(1));
        return cg::grad(loss, {x, w});
    };
    SmallVector<TestSpec> specs(3, {{{123, 321}, {321, 345}}, 0});
    run_eager_reeval_test(specs, make_graph);
}

TEST_F(TestGraphEagerReeval, ReuseAfterRelease) {
    auto graph = ComputingGraph::make();
    graph->options().eager_evaluation = true;
    HostTensorGenerator<> gen;
    constexpr int SIZE = 123;
    auto host_x = gen({SIZE});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = x + 1; // Typecvt(int2float)
    graph->clear_device_memory();
    x = opr::Host2DeviceCopy::make(*graph, host_x);
    y = x + 1;
    HostTensorND host_y;
    host_y.copy_from(y.eager_eval_get_value()).sync();
    for (size_t i = 0; i < SIZE; ++ i) {
        ASSERT_FLOAT_EQ(host_x->ptr<float>()[i] + 1.f, host_y.ptr<float>()[i]);

    }
}

#if MGB_CUDA
TEST_F(TestGraphEagerReeval, MemoryAlloc) {
    REQUIRE_GPU(1);
    CompNode::load("gpux").activate();

    size_t reserve;
    {
        size_t free, tot;
        MGB_CUDA_CHECK(cudaMemGetInfo(&free, &tot));
        reserve = free * 0.92;
    }
    auto reserve_setting = ssprintf("b:%zu", reserve);

    auto run = [this, reserve] {
        CompNode::try_coalesce_all_free_memory();
        CompNode::finalize();
        auto cn = CompNode::load("gpux");
        cn.sync();

        // 1 -> 2 -> 4 -> 8 -> x(16) -> x0(fwd) -> 8 -|
        //                      |                     +--> 8 -> 8
        //                      |-----> x1(fwd) -> 8 -|
        // total usage : 63 + (16 after the first iteration)
        // x might has iteration i's memory, but x0/x1 foward i-1's memory
        size_t length = reserve / (sizeof(dt_int32) * 5 * 16);
        auto host_x = std::make_shared<HostTensorND>(cn, dtype::Int32());
        HostTensorND host_val;
        dt_int32 expect = 0;

        auto set_input = [&] {
            auto px = host_x->resize({length}).ptr<dt_int32>();
            RNGxorshf rng{next_rand_seed()};
            expect = 0;
            for (size_t i = 0; i < length; ++ i) {
                expect += ((px[i] = rng()) + 1) * 2;
            }
            expect *= 16;
        };

        auto make_graph = [length](const SymbolVarArray& inputs) -> SymbolVarArray {
            auto x = inputs[0];
            for (size_t j = 4; j > 0; -- j) {
                x = opr::Concat::make({x, x}, 0);
            }
            using AIdx = opr::indexing::AxisIndexer;
            dt_int32 point = length * 8;
            auto x0 = opr::Subtensor::make(x,
                {AIdx::make_interval(0, None, x.make_scalar(point), None)}),
                 x1 = opr::Subtensor::make(x,
                {AIdx::make_interval(0, x.make_scalar(point), None, None)});
            auto y0 = x0 + 1, y1 = x1 + 1, z = (y0 + y1) * 2,
                 out = opr::reduce_sum(z, z.make_scalar(1));
            return {out};
        };

        for (size_t iter = 0; iter < 5; ++ iter) {
            set_input();
            auto out = make_graph_eager({host_x}, make_graph, false);
            host_val.copy_from(out[0].eager_eval_get_value()).sync();
            ASSERT_EQ(expect, host_val.ptr<dt_int32>()[0]);
        }

    };

    // reserve memory explicitly to avoid uncontrollable factors
    constexpr const char* KEY = "MGB_CUDA_RESERVE_MEMORY";
    auto old_value = getenv(KEY);
    setenv(KEY, reserve_setting.c_str(), 1);
    MGB_TRY {
        run();
    } MGB_FINALLY(
        if (old_value) {
            setenv(KEY, old_value, 1);
        } else {
            unsetenv(KEY);
        }
        CompNode::try_coalesce_all_free_memory();
        CompNode::finalize();
    );
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
