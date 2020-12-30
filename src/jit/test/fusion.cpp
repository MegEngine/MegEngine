/**
 * \file src/jit/test/fusion.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#include "megbrain_build_config.h"

#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/misc.h"
#include "megbrain/graph/cg.h"
#include "megbrain/jit/ast_c.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/jit/fusion_pass.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/dnn/convolution.h"

#if MGB_JIT

using namespace mgb;
using namespace jit;

#define FOREACH_CASE(cb)                                                       \
    cb(basic) cb(shape_change) cb(large_num_inps) cb(simple_exp)               \
    cb(complex_exp) cb(exp_pow) cb(cache) cb(all_oprs)                         \
    cb(expand_jit_executor) cb(multi_device) cb(multi_shape)                   \
    cb(non_contig) cb(visit_complexity) cb(imm_scalar)                         \
    cb(jit_grad) cb(concat_input) cb(special_graph_input)

namespace {
#define def_tag(x) \
    struct x {};
FOREACH_CASE(def_tag)
#undef def_tag

#define t(n) n,
using test_types = ::testing::Types<FOREACH_CASE(t) void>;
#undef t

template <typename tag>
void run(Backend backend, CompNode cn);

template <typename T>
size_t find_opr_num(SymbolVar endpoint) {
    size_t opr_num = 0;
    auto cb = [&opr_num](cg::OperatorNodeBase* opr) {
        if (opr->same_type<T>()) {
            opr_num++;
        }
    };
    cg::DepOprIter{cb}.add(endpoint.node()->owner_opr());
    return opr_num;
}

template <typename T>
SmallVector<T*> find_oprs(SymbolVar endpoint) {
    SmallVector<T*> res;
    auto cb = [&res](cg::OperatorNodeBase* opr) {
        if (opr->same_type<T>()) {
            auto ptr = &(opr->cast_final_safe<T>());
            res.push_back(ptr);
        }
    };
    cg::DepOprIter{cb}.add(endpoint.node()->owner_opr());
    return res;
}

template <typename T>
SmallVector<T*> find_oprs(cg::AsyncExecutable& func) {
    SmallVector<T*> res;
    auto cb = [&res](cg::OperatorNodeBase* opr) {
        if (opr->same_type<T>()) {
            auto ptr = &(opr->cast_final_safe<T>());
            res.push_back(ptr);
        }
        return true;
    };
    func.iter_opr_seq(cb);
    return res;
}

//! make a pair of functions with and without JIT optimization
std::pair<std::unique_ptr<cg::AsyncExecutable>,
          std::unique_ptr<cg::AsyncExecutable>>
make_func_pair(HostTensorND& dst0, HostTensorND& dst1,
               thin_function<SymbolVar(ComputingGraph&)> make_dst,
               uint8_t jit_level) {
    auto g0 = ComputingGraph::make();
    g0->options().graph_opt_level = 0;
    auto f0 = g0->compile({make_callback_copy(make_dst(*g0), dst0)});

    auto g1 = ComputingGraph::make();
    g1->options().graph_opt_level = 3;
    g1->options().graph_opt.jit = jit_level;
    auto f1 = g1->compile({make_callback_copy(make_dst(*g1), dst1)});

    EXPECT_FALSE(find_oprs<JITExecutor>(*f1).empty());
    return {std::move(f0), std::move(f1)};
}

template <>
void run<void>(Backend, CompNode) {}

template <>
void run<basic>(Backend backend, CompNode cn) {
    set_backend(backend);

    HostTensorGenerator<> gen;
    auto host_x0 = gen({3, 3}, cn), host_x1 = gen({3, 1}, cn),
         host_x2 = gen({1, 1}, cn), host_x3 = gen({3, 1}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto a = opr::Host2DeviceCopy::make(graph, host_x0),
             b = opr::Host2DeviceCopy::make(graph, host_x1),
             c = opr::Host2DeviceCopy::make(graph, host_x2),
             d = opr::Host2DeviceCopy::make(graph, host_x3);
        return a * b + c * a + d + d + d;
    };
    HostTensorND host_z1, host_z2;
    auto funcs = make_func_pair(host_z1, host_z2, make_dst, 2);
    funcs.first->execute();
    funcs.second->execute();
    MGB_ASSERT_TENSOR_EQ(host_z1, host_z2);
    auto jits = find_oprs<JITExecutor>(*funcs.second);
    ASSERT_EQ(2u, jits.size());
    // only one broadcast is allowed in JIT fusion
    ASSERT_EQ(1u, jits[0]->input().size());
    ASSERT_EQ(4u, jits[1]->input().size());

    //! check memfwd
    ASSERT_EQ(prev_dev_ptr(jits[0]->input(0)),
              prev_dev_ptr(jits[0]->output(0)));
    ASSERT_EQ(prev_dev_ptr(jits[1]->input(0)),
              prev_dev_ptr(jits[1]->output(0)));
}

template <>
void run<shape_change>(Backend backend, CompNode cn) {
    set_backend(backend);

    HostTensorGenerator<> gen;
    auto host_x0 = gen({3, 3}, cn), host_x1 = gen({3, 1}, cn),
         host_x2 = gen({1, 1}, cn), host_x3 = gen({1, 3}, cn);

    auto run_gen = [&](size_t n, bool dim = false, bool swap = false) {
        if (dim) {
            host_x0->copy_from(*gen({n, n, 3}, cn));
            host_x1->copy_from(*gen({n, 1, 1}, cn));
            host_x2->copy_from(*gen({1, 1, 3}, cn));
            host_x3->copy_from(*gen({1, n, 1}, cn));
        } else {
            host_x0->copy_from(*gen({n, n}, cn));
            host_x1->copy_from(*gen({n, 1}, cn));
            host_x2->copy_from(*gen({1, 1}, cn));
            host_x3->copy_from(*gen({1, n}, cn));
        }
        if (swap) {
            std::swap(*host_x1, *host_x3);
        }
    };

    using JITOprArr = std::array<JITExecutor*, 2>;
    auto make_func = [&](HostTensorND& out, JITOprArr* jit) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
             b = opr::Host2DeviceCopy::make(*graph, host_x1),
             c = opr::Host2DeviceCopy::make(*graph, host_x2),
             d = opr::Host2DeviceCopy::make(*graph, host_x3);

        auto y = opr::abs(a) * (b + c) * d - (b + c) * c * b;
        if (jit) {
            graph->options().graph_opt_level = 3;
        }
        auto func = graph->compile({make_callback_copy(y, out)});
        if (jit) {
            unpack_vector(find_oprs<JITExecutor>(*func), (*jit)[0], (*jit)[1]);
        }
        return func;
    };
    JITOprArr jits;
    HostTensorND host_y1, host_y2;
    auto func1 = make_func(host_y1, nullptr), func2 = make_func(host_y2, &jits);

    auto run = [&]() -> std::array<Executable*, 2> {
        func1->execute();
        func2->execute();
        auto chk = [&]() { MGB_ASSERT_TENSOR_EQ(host_y1, host_y2); };
        chk();
        return {jits[0]->executable(), jits[1]->executable()};
    };

    auto exe_shp3 = run();

    {
        run_gen(5);
        auto exe_shp5 = run();
        if (backend == Backend::HALIDE) {
            ASSERT_NE(exe_shp3, exe_shp5);
        } else {
            ASSERT_EQ(exe_shp3, exe_shp5);
        }
    }

    // change ndim
    run_gen(3, true);
    ASSERT_NE(exe_shp3, run());

    // change bcast pattern
    {
        run_gen(3, false, true);
        auto exe_chg = run();
        if (backend == Backend::HALIDE) {
            ASSERT_NE(exe_shp3, exe_chg);
        } else {
            ASSERT_EQ(exe_shp3, exe_chg);
        }
    }

    run_gen(3);
    ASSERT_EQ(exe_shp3, run());
}

template <>
void run<large_num_inps>(Backend backend, CompNode cn) {
    set_backend(backend);

    HostTensorGenerator<> gen;
    int inp_nr = 120;
    std::vector<std::shared_ptr<HostTensorND>> host_xs;
    for (int i = 0; i < inp_nr; i++)
        host_xs.push_back(gen({4, 3, 2, 1}, cn));

    auto make_dst = [&](ComputingGraph& graph) {
        std::vector<SymbolVar> dev_xs;
        for (int i = 0; i < inp_nr; i++)
            dev_xs.push_back(opr::Host2DeviceCopy::make(graph, host_xs[i]));

        auto y = dev_xs[0] + dev_xs[1];
        for (int i = 2; i < inp_nr; i++)
            y = y + dev_xs[i];
        return y;
    };
    HostTensorND host_y1, host_y2;
    auto funcs = make_func_pair(host_y1, host_y2, make_dst, 2);
    funcs.first->execute();
    funcs.second->execute();
    MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);

    ASSERT_GT(find_oprs<JITExecutor>(*funcs.second).size(), 1u);
}

template <>
void run<concat_input>(Backend backend, CompNode cn) {
    set_backend(backend);
    FusionChecker checker{
            4,
            [](const SymbolVarArray& inp) -> SymbolVar {
                auto spl = opr::Split::make(
                        inp[0],
                        opr::Split::Options::make_partition(inp[0], 1, {1, 1}));
                return spl[1] * inp[1] + inp[2] * spl[1] + inp[3] + inp[3];
            },
            cn};
    checker.disable_opr_type_check().run({TensorShape{3, 2}, {3, 1}, {3, 1}, {3, 1}});
}

template <>
void run<simple_exp>(Backend backend, CompNode cn) {
    set_backend(backend);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              return inp[0] + inp[1];
                          },
                          cn};
    checker.enable_direct_build().run({TensorShape{3, 3}, {3, 3}});
}

template <>
void run<jit_grad>(Backend backend, CompNode cn) {
    set_backend(backend);

    FusionChecker checker{
            1,
            [](const SymbolVarArray& inp) -> SymbolVar { return inp[0] + 1; },
            cn};
    checker.enable_direct_build().run({TensorShape{3, 1}});
}

template <>
void run<exp_pow>(Backend backend, CompNode cn) {
    set_backend(backend);

    FusionChecker checker{
            3,
            [](const SymbolVarArray& inp) -> SymbolVar {
                auto iabs = opr::abs(inp[0]) + .23f;
                return opr::exp(inp[0]) + opr::exp(inp[1]) -
                       opr::exp(inp[2]) * opr::pow(opr::abs(inp[1]) + 0.2f,
                                                   opr::abs(inp[2]) + 0.1f) +
                       opr::powf(inp[0], 2) - opr::powf(inp[0], -3) +
                       opr::powf(iabs, 1.f / 3.f) +
                       opr::PowC::make(iabs, -1.f / 3.f) +
                       opr::PowC::make(iabs, .5f) + opr::PowC::make(iabs, -.5f);
            },
            cn};
    checker.run({TensorShape{2, 3}, {2, 3}, {2, 3}});
}

template <>
void run<complex_exp>(Backend backend, CompNode cn) {
    set_backend(backend);

    FusionChecker checker{4,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              return opr::abs(inp[0]) * (inp[1] + inp[2]) *
                                             inp[3] -
                                     (inp[1] + inp[2]) * inp[2] / inp[1];
                          },
                          cn};
    checker.run({TensorShape{3, 3}, {1, 3}, {3, 1}, {1, 3}});
}

template <>
void run<cache>(Backend backend, CompNode cn) {
    set_backend(backend);

    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_a = gen({1}, cn), host_b = gen({1}, cn), host_c = gen({1}, cn);
    auto a = opr::Host2DeviceCopy::make(*graph, host_a),
         b = opr::Host2DeviceCopy::make(*graph, host_b),
         c = opr::Host2DeviceCopy::make(*graph, host_c), x = opr::sin(a + 1),
         y = opr::cos(b + 1), z = opr::sin(c + 1);

    gopt::GraphOptimizer gopt;
    gopt.add_pass<gopt::JITFusionPass>();
    VarNodeArray vars{x.node(), y.node(), z.node()};
    gopt.apply_inplace(vars);

    ASSERT_NE(vars[0], vars[1]);
    ASSERT_NE(vars[0], vars[2]);
    ASSERT_NE(vars[1], vars[2]);

    auto func = graph->compile({{vars[0], {}}, {vars[1], {}}, {vars[2], {}}});
    func->execute();

    auto get_exe = [](SymbolVar var) {
        return var.node()
                ->owner_opr()
                ->cast_final_safe<JITExecutor>()
                .executable();
    };
    auto ex0 = get_exe(vars[0]), ex1 = get_exe(vars[1]), ex2 = get_exe(vars[2]);
    ASSERT_EQ(ex0, ex2);
    ASSERT_NE(ex0, ex1);
}

template <>
void run<all_oprs>(Backend backend, CompNode cn) {
    // test all supported modes in multiple threads
    set_backend(backend);

    std::vector<std::pair<const char*, thin_function<void()>>> tasks;

    static auto itrans_none = [](SymbolVar* data, size_t size) {};
    static auto itrans_pos = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::abs(data[i]) + float(0.1f + 0.23f * i);
        }
    };
    static auto itrans_clip1 = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::max(opr::min(data[i], data[i].make_scalar_dt(0.9f)),
                               data[i].make_scalar_dt(-0.9f));
        }
    };
    static auto itrans_gt0 = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::max(data[i], data[i].make_scalar_dt(0.1f));
        }
    };
    static auto itrans_ne0 = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            auto mask = opr::abs(data[i]) < 0.1f;
            data[i] = data[i] * (1.f - mask) + mask * (data[i] + 1.f);
        }
    };

#define DO_CHK_ELEM(_mode, _arity, _do_grad, _itrans, _shps...)         \
    tasks.emplace_back(#_mode, [cn]() {                                 \
        FusionChecker chk{_arity,                                       \
                          [](SymbolVarArray inp) -> SymbolVar {         \
                              itrans_##_itrans(inp.data(), inp.size()); \
                              return opr::Elemwise::make(               \
                                      inp, opr::Elemwise::Mode::_mode); \
                          },                                            \
                          cn};                                          \
        chk.enable_direct_build();                                      \
        if (!_do_grad) {                                                \
            chk.disable_inp_grad();                                     \
        }                                                               \
        chk.run({_shps});                                               \
    })

#define CHECK_ELEM1(_mode, _do_grad, _itrans) \
    DO_CHK_ELEM(_mode, 1, _do_grad, _itrans, TensorShape{9, 12, 7})
#define CHECK_ELEM2(_mode, _do_grad, _itrans)                       \
    DO_CHK_ELEM(_mode, 2, _do_grad, _itrans, TensorShape{9, 12, 7}, \
                TensorShape{9, 1, 7})
#define CHECK_ELEM3(_mode, _do_grad, _itrans)                       \
    DO_CHK_ELEM(_mode, 3, _do_grad, _itrans, TensorShape{9, 12, 7}, \
                TensorShape{9, 1, 7}, TensorShape{1, 12, 7})
#define CHECK_ELEM4(_mode, _do_grad, _itrans)                       \
    DO_CHK_ELEM(_mode, 4, _do_grad, _itrans, TensorShape{9, 12, 7}, \
                TensorShape{9, 1, 7}, TensorShape{1, 12, 7},        \
                TensorShape{9, 12, 1})

    CHECK_ELEM1(RELU, true, none);
    CHECK_ELEM1(ABS, true, none);
    CHECK_ELEM1(ACOS, true, clip1);
    CHECK_ELEM1(ASIN, true, clip1);
    CHECK_ELEM1(CEIL, false, none);
    CHECK_ELEM1(COS, true, none);
    CHECK_ELEM1(EXP, true, none);
    CHECK_ELEM1(EXPM1, true, none);
    CHECK_ELEM1(FLOOR, false, none);
    CHECK_ELEM1(LOG, true, gt0);
    CHECK_ELEM1(LOG1P, true, gt0);
    CHECK_ELEM1(NEGATE, true, none);
    CHECK_ELEM1(SIGMOID, true, none);
    CHECK_ELEM1(SIN, true, none);
    CHECK_ELEM1(TANH, true, none);
    CHECK_ELEM1(ERF, true, none);
    CHECK_ELEM1(ERFC, true, none);
    CHECK_ELEM1(H_SWISH, true, none);

    CHECK_ELEM2(ABS_GRAD, true, none);
    CHECK_ELEM2(ADD, true, none);
    CHECK_ELEM2(FLOOR_DIV, false, ne0);
    CHECK_ELEM2(MAX, true, none);
    CHECK_ELEM2(MIN, true, none);
    CHECK_ELEM2(MOD, false, ne0);
    CHECK_ELEM2(MUL, true, none);
    CHECK_ELEM2(POW, true, pos);
    CHECK_ELEM2(SIGMOID_GRAD, true, none);
    CHECK_ELEM2(SUB, true, none);
    CHECK_ELEM2(SWITCH_GT0, true, none);
    CHECK_ELEM2(TANH_GRAD, true, none);
    CHECK_ELEM2(TRUE_DIV, true, ne0);
    CHECK_ELEM2(LOG_SUM_EXP, true, none);
    CHECK_ELEM2(H_SWISH_GRAD, false, none);

    CHECK_ELEM2(LT, false, none);
    CHECK_ELEM2(LEQ, false, none);
    CHECK_ELEM2(EQ, false, none);

    CHECK_ELEM2(ATAN2, true, gt0);

    CHECK_ELEM3(COND_LEQ_MOV, false, none);
    CHECK_ELEM3(FUSE_MUL_ADD3, true, none);

    CHECK_ELEM4(FUSE_MUL_ADD4, true, none);

    CHECK_ELEM2(FUSE_ADD_RELU, true, none);
    CHECK_ELEM2(FUSE_ADD_SIGMOID, true, none);
    CHECK_ELEM2(FUSE_ADD_TANH, true, none);
    CHECK_ELEM2(FUSE_ADD_H_SWISH, true, none);

    ASSERT_EQ(ast_c::elem_opr_generator().size(), tasks.size());

    auto type_cvt_test = [&](const char* name, DType src_dtype,
                             DType dst_dtype) {
        tasks.emplace_back(name, [cn, src_dtype, dst_dtype]() {
            FusionChecker checker{
                    1,
                    [dst_dtype](const SymbolVarArray& inp) -> SymbolVar {
                        return opr::TypeCvt::make(inp[0], dst_dtype);
                    },
                    cn};
            checker.enable_direct_build();
            checker.set_dtype(0, src_dtype).run({TensorShape{4, 7, 99, 1}});
        });
    };

    type_cvt_test("f16->f32", dtype::Float16(), dtype::Float32());
    type_cvt_test("f32->f16", dtype::Float32(), dtype::Float16());

#undef CHECK_ELEM1
#undef CHECK_ELEM2
#undef CHECK_ELEM3
#undef CHECK_ELEM4
#undef DO_CHK_ELEM

    std::vector<std::thread> workers;
    std::atomic_size_t finished_tasks{0};
    auto worker = [&tasks, &finished_tasks](int wid) {
        for (;;) {
            size_t id = finished_tasks.fetch_add(1);
            if (id >= tasks.size()) {
                return;
            }
            if (!::testing::Test::HasFailure()) {
                mgb_log("going to run %s on worker %d", tasks[id].first, wid);
                ASSERT_NO_THROW(tasks[id].second())
                        << "failed for " << tasks[id].first;
            }
        }
    };
    int nr_worker;
    if (auto set = MGB_GETENV("MGB_JIT_TEST_WORKER")) {
        nr_worker = std::stoi(set);
    } else {
        nr_worker = CompNode::get_device_count(CompNode::DeviceType::CPU) / 2;
    }

    if (nr_worker == 1) {
        worker(-1);
    } else {
        for (int i = 0; i < nr_worker; ++i) {
            workers.emplace_back(worker, i);
        }
        for (auto&& i : workers) {
            i.join();
        }
    }

    ASSERT_GE(finished_tasks.load(), tasks.size());
}

template <>
void run<expand_jit_executor>(Backend backend, CompNode cn) {
    set_backend(backend);

    auto make_jit = [](SymbolVar target, const SymbolVarArray& inputs) {
        auto y = target.node();
        auto ig_gen = std::make_unique<InternalGraphGenerator>(y->owner_opr());
        auto inputs_vptr = cg::to_var_node_array(inputs);
        for (auto i : get_rev_topo_order(
                     target, {inputs_vptr.begin(), inputs_vptr.end()})) {
            ig_gen->add_opr(i);
        }
        auto igraph = ig_gen->generate();
        return JITExecutor::make(igraph, ig_gen->orig_inps());
    };

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 3;
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 3}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto type_cvt_x = opr::TypeCvt::make(x, dtype::Float16());
    auto relu_x = opr::relu(type_cvt_x);
    auto sin_x = opr::sin(relu_x);

    auto host_y = gen({3, 3}, cn);
    auto y = opr::Host2DeviceCopy::make(*graph, host_y);
    auto type_cvt_y = opr::TypeCvt::make(y, dtype::Float16());
    auto relu_y = opr::relu(type_cvt_y);
    auto sin_y = opr::sin(relu_y);

    auto fusion_x = make_jit(sin_x, {relu_x});
    auto fusion_y = make_jit(sin_y, {type_cvt_y});

    auto z = fusion_x + fusion_y;

    // expanding at endpoint
    auto fusion0_x = make_jit(sin_x, {type_cvt_x});
    auto fusion1_x = make_jit(fusion0_x, {x});
    auto fusion2_x = make_jit(sin_x, {x});
    ASSERT_EQ(fusion1_x, fusion2_x);

    // expand mulitple JITExecutor
    auto fusion_z = make_jit(z, {x, y});
    auto fusion_z_expected = make_jit(sin_x + sin_y, {x, y});
    ASSERT_EQ(fusion_z, fusion_z_expected);
}

SymbolVar jit_stop(SymbolVar x) {
    return opr::Sleep::make(x, 1e-3);
}

template <>
void run<multi_device>(Backend backend, CompNode cn) {
    set_backend(backend);

    auto loc = cn.locator_logical();
    mgb_assert(loc.device >= 0);
    loc.device += 1;
    if (loc.device >= static_cast<int>(CompNode::get_device_count(loc.type))) {
        return;
    }

    HostTensorGenerator<> gen;
    auto cn1 = CompNode::load(loc);
    auto host_x = gen({42, 23}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x),
             a = opr::tanh(x) + opr::sin(x), y = opr::Copy::make(x, cn1),
             b = opr::tanh(y) + opr::sin(y);
        return jit_stop(a) + opr::Copy::make(b, cn);
    };
    HostTensorND host_z1, host_z2;
    auto funcs = make_func_pair(host_z1, host_z2, make_dst, 2);
    for (int i = 0; i < 8; ++i) {
        funcs.first->execute();
        funcs.second->execute();
        if (i == 4) {
            host_x->copy_from(*gen({10, 20, 3}, cn));
        } else {
            host_x->copy_from(*gen(host_x->shape(), cn));
        }
        MGB_ASSERT_TENSOR_EQ(host_z1, host_z2);
    }

    auto jits = find_oprs<JITExecutor>(*funcs.second);
    ASSERT_EQ(2u, jits.size());
    ASSERT_EQ(jits[0]->internal_graph().output(),
              jits[1]->internal_graph().output());
}

template <>
void run<multi_shape>(Backend backend, CompNode cn) {
    // multiple shapes of same computing expr
    set_backend(backend);

    HostTensorGenerator<> gen;
    auto host_x = gen({4, 2, 3}, cn), host_y = gen({4, 2}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x).rename("x"),
             y = opr::Host2DeviceCopy::make(graph, host_y).rename("y"),
             jit0 = jit_stop(opr::sin(x) * x),
             a = opr::AxisAddRemove::make(
                     opr::Reduce::make(jit0,
                                       {opr::Reduce::Param::Mode::SUM, 2}),
                     {opr::AxisAddRemove::AxisDesc::make_remove(2)}),
             jit1 = jit_stop(opr::sin(a) + opr::sin(y)),
             jit2 = opr::sin(jit1) * jit1;
        return jit2;
    };
    HostTensorND host_z1, host_z2;
    auto funcs = make_func_pair(host_z1, host_z2, make_dst, 2);
    auto jits = find_oprs<JITExecutor>(*funcs.second);
    ASSERT_EQ(3u, jits.size());
    ASSERT_EQ(jits[0]->internal_graph().output(),
              jits[2]->internal_graph().output());
    for (int i = 0; i < 8; ++i) {
        funcs.first->execute();
        funcs.second->execute();
        if (i == 4) {
            host_x->copy_from(*gen({3, 7, 5}, cn));
            host_y->copy_from(*gen({3, 7}, cn));
        } else {
            host_x->copy_from(*gen(host_x->shape(), cn));
            host_y->copy_from(*gen(host_y->shape(), cn));
        }
        MGB_ASSERT_TENSOR_EQ(host_z1, host_z2);
    }
}

template <>
void run<non_contig>(Backend backend, CompNode cn) {
    set_backend(backend);

    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}, cn);
    SmallVector<std::pair<SymbolVar, SymbolVar>> subs;
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x),
             y = opr::Subtensor::make(
                     x, {opr::Subtensor::AxisIndexer::make_interval(
                                1, x.make_scalar(1), x.make_scalar(3), None)});
        subs.emplace_back(x, y);
        return opr::sin(y) * y;
    };
    HostTensorND y0, y1;
    auto funcs = make_func_pair(y0, y1, make_dst, 2);
    for (size_t s : {4, 7}) {
        *host_x = *gen({3, s});
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_EQ(y0, y1);
    }

    ASSERT_EQ(2u, subs.size());
    for (int i = 0; i < 2; ++i) {
        auto p0 = static_cast<const float*>(prev_dev_ptr(subs[i].first)) + 1,
             p1 = static_cast<const float*>(prev_dev_ptr(subs[i].second));
        if (backend != Backend::HALIDE || !i) {
            ASSERT_EQ(p0, p1);
        } else {
            ASSERT_NE(p0, p1);
        }
    }
}

template <>
void run<visit_complexity>(Backend backend, CompNode cn) {
    // build a graph that would have exponential complexity if graph visiting is
    // not correctly implemented
    set_backend(backend);

    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{0.01f,
                                                                         0.02f};
    auto host_x = gen({3, 4}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x);
        auto y = x;
        for (int i = 0; i < 32; ++i) {
            y = y * y + y;
        }
        return y;
    };
    HostTensorND host_y1, host_y2;
    auto funcs = make_func_pair(host_y1, host_y2, make_dst, 2);
    funcs.first->execute();
    funcs.second->execute();
    MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);

    ASSERT_EQ(1u, find_oprs<JITExecutor>(*funcs.second).size());
    ASSERT_TRUE(find_oprs<opr::Elemwise>(*funcs.second).empty());
}

template <>
void run<imm_scalar>(Backend backend, CompNode cn) {
    set_backend(backend);

    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3, 4}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x);
        return (x * x + 1.f) / (opr::sin(x) + 1.2f) * .3f;
    };
    HostTensorND host_y1, host_y2;
    auto funcs = make_func_pair(host_y1, host_y2, make_dst, 2);

    funcs.first->execute();
    funcs.second->execute();
    MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);

    JITExecutor* jit;
    unpack_vector(find_oprs<JITExecutor>(*funcs.second), jit);
    ASSERT_TRUE(find_oprs<opr::Elemwise>(*funcs.second).empty());

    ASSERT_EQ(1u, jit->input().size());
    ASSERT_TRUE(jit->input(0)->owner_opr()->same_type<opr::Host2DeviceCopy>());
}

template <>
void run<special_graph_input>(Backend backend, CompNode cn) {
    set_backend(backend);

    HostTensorGenerator<> gen;
    auto host_x = gen({3, 3}, cn);
    auto host_y = gen({2, 1}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x);
        auto y = opr::Host2DeviceCopy::make(graph, host_y);
        auto spl = opr::Split::make(x,
                        opr::Split::Options::make_partition(x, 1, {1, 2}));
        auto mat = mgb::opr::MatrixMul::make(spl[1], y);
        return (spl[0] * spl[0] + 1.f) / (mat + 1.2f) * .3f;
    };
    HostTensorND host_y1, host_y2;
    auto funcs = make_func_pair(host_y1, host_y2, make_dst, 2);

    funcs.first->execute();
    funcs.second->execute();
    MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);

    JITExecutor* jit;
    unpack_vector(find_oprs<JITExecutor>(*funcs.second), jit);
    ASSERT_TRUE(find_oprs<opr::Elemwise>(*funcs.second).empty());
    ASSERT_EQ(2u, jit->input().size());
}

}  // namespace

#if MGB_JIT_HALIDE
TEST(TestJITFusionHalide, SimpleReduce) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 3;
    graph->options().graph_opt.jit = 2;
    HostTensorGenerator<> gen;
    auto host_x0 = gen({3, 3}), host_x1 = gen({3, 1});
    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         y = opr::reduce_sum(a + b, opr::GetVarShape::make(b)),
         z = opr::reduce_sum(a * b, opr::GetVarShape::make(a)) + y;

    SymbolVar z_opt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_preset_passes(true, nullptr, &(graph->options()))
                          .apply({{z}})
                          .endpoint_vars(),
                  z_opt);
    ASSERT_EQ(2u, find_opr_num<mgb::jit::JITExecutor>(z_opt));
    HostTensorND h;
    graph->compile({make_callback_copy(z_opt, h)})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestJITFusionHalide.SimpleReduce.json"));
}

TEST(TestJITFusionHalide, JITExecutor) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 3;
    graph->options().graph_opt.jit = 2;
    HostTensorGenerator<> gen;
    auto host_x0 = gen({3, 3}), host_x1 = gen({3, 1}), host_x2 = gen({3, 3}),
         host_x3 = gen({3, 1});
    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         c = opr::Host2DeviceCopy::make(*graph, host_x2),
         d = opr::Host2DeviceCopy::make(*graph, host_x3),
         shape_of_b = opr::GetVarShape::make(b),
         shape_of_a = opr::GetVarShape::make(a),
         y = opr::reduce_sum(a + b, shape_of_b),
         z = opr::reduce_sum(a * b, shape_of_a);
    auto ig_gen_1 =
            std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());
    auto ig_gen_2 =
            std::make_unique<InternalGraphGenerator>(z.node()->owner_opr());
    {
        ThinHashSet<VarNode*> nd_set;
        nd_set.insert(a.node());
        nd_set.insert(b.node());
        nd_set.insert(shape_of_b.node());
        auto topo = get_rev_topo_order(y, nd_set);
        for (auto opr : topo) {
            ig_gen_1->add_opr(opr);
        }
    }
    {
        ThinHashSet<VarNode*> nd_set;
        nd_set.insert(a.node());
        nd_set.insert(b.node());
        nd_set.insert(shape_of_a.node());
        auto topo = get_rev_topo_order(z, nd_set);
        for (auto opr : topo) {
            ig_gen_2->add_opr(opr);
        }
    }
    auto ig_1 = ig_gen_1->generate(), ig_2 = ig_gen_2->generate();
    auto jit_1 = JITExecutor::make(ig_1, ig_gen_1->orig_inps());
    auto jit_2 = JITExecutor::make(ig_2, ig_gen_2->orig_inps());
    auto w = opr::reduce_sum(a * b + c * d, opr::GetVarShape::make(a)),
         x = w + jit_1, u = x * jit_2;

    SymbolVar u_opt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_preset_passes(true, nullptr, &(graph->options()))
                          .apply({{u}})
                          .endpoint_vars(),
                  u_opt);
    ASSERT_EQ(2u, find_opr_num<mgb::jit::JITExecutor>(u_opt));
    ASSERT_GT(1u, find_opr_num<opr::Elemwise>(u_opt));
    HostTensorND h;
    graph->compile({make_callback_copy(u_opt, h)})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestJITFusionHalide.JITExecutor.json"));
}

TEST(TestJITFusionHalide, BatchNormalization) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    auto graph1 = ComputingGraph::make();
    graph1->options().graph_opt_level = 3;
    graph1->options().graph_opt.jit = 2;
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{0.1,
                                                                         1};
    size_t n = 32, c = 24, h = 28, w = 28;
    auto host_x0 = gen({n, c, h, w});
    auto host_tshp = std::make_shared<HostTensorND>(host_x0->comp_node(),
                                                    dtype::Int32());
    host_tshp->resize({4});
    host_tshp->ptr<int>()[0] = 1;
    host_tshp->ptr<int>()[1] = c;
    host_tshp->ptr<int>()[2] = 1;
    host_tshp->ptr<int>()[3] = 1;
    auto host_pow = std::make_shared<HostTensorND>(host_x0->comp_node(),
                                                   dtype::Float32());
    host_pow->resize({1});
    host_pow->ptr<float>()[0] = -0.5;
    auto pow = opr::Host2DeviceCopy::make(*graph1, host_pow, {"pow"});
    auto x = opr::Host2DeviceCopy::make(*graph1, host_x0, {"x"}),
         tshp = opr::Host2DeviceCopy::make(*graph1, host_tshp, {"tshp"});
    auto xshp = opr::GetVarShape::make(x);
    auto reduce_size = opr::reduce_prod(xshp, xshp.make_scalar(1)) /
                       opr::reduce_prod(tshp, tshp.make_scalar(1));
    auto xx = opr::Elemwise::make({2 * x}, opr::Elemwise::Param::Mode::RELU);
    auto x1 = opr::reduce_sum(xx, tshp);
    auto x2 = opr::reduce_sum_sqr(xx, tshp);
    auto var = (x2 - x1 * x1 / reduce_size) / (reduce_size - 1),
         regular_var = var + (float)(1e-5);
    auto invsqrt_var = opr::Elemwise::make({regular_var, pow},
                                           opr::Elemwise::Param::Mode::POW);
    auto ovar = (x - x1 / reduce_size) * invsqrt_var;
    HostTensorND h_ovar;

    using Callback = thin_function<void(DeviceTensorND&)>;
    using OutputSpecItem = std::pair<SymbolVar, Callback>;
    using OutputSpec = std::vector<OutputSpecItem>;
    OutputSpec out_spec;
    out_spec.push_back(make_callback_copy(ovar, h_ovar));
    HostTensorND h_grad;
    bool do_grad = true;
    if (do_grad) {
        auto reduce_ovar = opr::reduce_sum(ovar * ovar, ovar.make_scalar(1));
        auto grad = cg::grad(reduce_ovar, x);
        out_spec.push_back(make_callback_copy(grad, h_grad));
    }
    auto func1 = graph1->compile(out_spec);
    func1->to_json()->writeto_fpath(
            output_file("TestJITFusionHalide.BatchNormalization.json"));
    func1->execute();

    auto graph2 = ComputingGraph::make();
    graph2->options().graph_opt_level = 0;
    auto pow_ = opr::Host2DeviceCopy::make(*graph2, host_pow, {"pow"});
    auto x_ = opr::Host2DeviceCopy::make(*graph2, host_x0, {"x"}),
         tshp_ = opr::Host2DeviceCopy::make(*graph2, host_tshp, {"tshp"});
    auto xshp_ = opr::GetVarShape::make(x_);
    auto reduce_size_ = opr::reduce_prod(xshp_, xshp_.make_scalar(1)) /
                        opr::reduce_prod(tshp_, tshp_.make_scalar(1));
    auto xx_ = opr::Elemwise::make({2 * x_}, opr::Elemwise::Param::Mode::RELU);
    auto x1_ = opr::reduce_sum(xx_, tshp_);
    auto x2_ = opr::reduce_sum_sqr(xx_, tshp_);
    auto var_ = (x2_ - x1_ * x1_ / reduce_size_) / (reduce_size_ - 1),
         regular_var_ = var_ + (float)(1e-5);
    auto invsqrt_var_ = opr::Elemwise::make({regular_var_, pow_},
                                            opr::Elemwise::Param::Mode::POW);
    auto ovar_ = (x_ - x1_ / reduce_size_) * invsqrt_var_;
    HostTensorND h_ovar_;

    OutputSpec out_spec_;
    out_spec_.push_back(make_callback_copy(ovar_, h_ovar_));
    HostTensorND h_grad_;
    if (do_grad) {
        auto reduce_ovar = opr::reduce_sum(ovar_ * ovar_, ovar_.make_scalar(1));
        auto grad = cg::grad(reduce_ovar, x_);
        out_spec_.push_back(make_callback_copy(grad, h_grad_));
    }
    auto func2 = graph2->compile(out_spec_);
    func2->execute();

    MGB_ASSERT_TENSOR_NEAR(h_ovar_, h_ovar, 3e-5);
    if (do_grad){
        MGB_ASSERT_TENSOR_NEAR(h_grad_, h_grad, 3e-4);
    }
}

TEST(TestJITFusionHalide, ReduceShapeManip) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);
    auto cn = CompNode::load("gpu0");
    HostTensorGenerator<> gen;

    auto do_chk = [&](bool dyn_shape) {
        auto host_x = gen({7, 8, 9}, cn);
        // TODO: handle opr fusion without shape constraints, and test dynamic
        // shape case where target shape can be inferred
        auto make_dst = [&host_x, dyn_shape](ComputingGraph& cg) {
            auto x = opr::Host2DeviceCopy::make(cg, host_x), xm2 = x * 2,
                 one = x.make_scalar(1),
                 tshp = opr::Concat::make(
                         {one,
                          opr::GetVarShape::make(
                                  dyn_shape ? opr::MarkDynamicVar::make(xm2)
                                            : xm2,
                                  1),
                          one},
                         0),
                 y = opr::reduce_sum(xm2, tshp) + 3;
            return y;
        };

        HostTensorND host_y0, host_y1;
        auto funcs = make_func_pair(host_y0, host_y1, make_dst, 2);
        auto run = [&]() {
            funcs.first->execute();
            funcs.second->execute();
            MGB_ASSERT_TENSOR_NEAR(host_y0, host_y1, 1e-5);
        };
        funcs.second->to_json()->writeto_fpath(output_file(ssprintf(
                "TestJITFusionHalide.ReduceShapeManip%d.json", dyn_shape)));
        run();
        host_x->copy_from(*gen({13, 4, 5}, cn));
        run();

        if (!dyn_shape) {
            JITExecutor* jit;
            unpack_vector(find_oprs<JITExecutor>(*funcs.second), jit);
            ASSERT_TRUE(jit->input(0)
                                ->owner_opr()
                                ->same_type<opr::Host2DeviceCopy>());
            ASSERT_EQ(2u, jit->input().size());
            auto dep_type = jit->node_prop().dep_map().at(jit->input(1));
            ASSERT_EQ(cg::OperatorNodeBase::NodeProp::DepType::HOST_VALUE,
                      dep_type);
            ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
        }
    };
    do_chk(false);
    do_chk(true);
}

TEST(TestJITFusionHalide, ReduceExp) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{
            2,
            [](const SymbolVarArray& inp) -> SymbolVar {
                auto var1 =
                        opr::reduce_sum(inp[0], opr::GetVarShape::make(inp[1]));
                auto var2 = opr::reduce_sum_sqr(inp[0] + inp[1],
                                                opr::GetVarShape::make(inp[1]));
                return var1 + var2;
            },
            CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
    checker.run({TensorShape{3, 3}, {1}});  // to scalar
}

TEST(TestJITFusionHalide, ReduceO16xC32) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    using DataType = opr::Reduce::Param::DataType;
    FusionChecker checker{
            2,
            [](const SymbolVarArray& inp) -> SymbolVar {
                auto var1 = opr::Reduce::make(
                        inp[0],
                        {opr::Reduce::Mode::SUM, 1, DataType::FLOAT_O16xC32},
                        {});
                auto var2 = opr::Reduce::make(inp[0],
                                              {opr::Reduce::Mode::SUM_SQR, 1,
                                               DataType::FLOAT_O16xC32},
                                              {});
                return var1 + var2;
            },
            CompNode::load("gpu0")};
    checker.disable_inp_grad().run({TensorShape{3, 3}, {3, 1}});
}

TEST(TestJITFusionHalide, ReduceSum) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              auto var1 = opr::reduce_sum(
                                      inp[0], opr::GetVarShape::make(inp[1]));
                              return var1 + inp[1];
                          },
                          CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
    checker.run({TensorShape{3, 3}, {1}});  // test reduce to scalar
}

TEST(TestJITFusionHalide, ReduceSumSqr) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              auto var1 = opr::reduce_sum_sqr(
                                      inp[0], opr::GetVarShape::make(inp[1]));
                              return var1 + inp[1];
                          },
                          CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
    checker.run({TensorShape{3, 3}, {3, 3}});  // test side effect
}

TEST(TestJITFusionHalide, ReduceMax) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              auto var1 = opr::reduce_max(
                                      inp[0], opr::GetVarShape::make(inp[1]));
                              return var1 + inp[1];
                          },
                          CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
}

TEST(TestJITFusionHalide, ReduceMin) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              auto var1 = opr::reduce_min(
                                      inp[0], opr::GetVarShape::make(inp[1]));
                              return var1 + inp[1];
                          },
                          CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
}

TEST(TestJITFusionHalide, ReduceProduct) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              auto var1 = opr::reduce_prod(
                                      inp[0], opr::GetVarShape::make(inp[1]));
                              return var1 + inp[1];
                          },
                          CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
}

TEST(TestJITFusionHalide, ReduceMean) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);

    FusionChecker checker{2,
                          [](const SymbolVarArray& inp) -> SymbolVar {
                              auto var1 = opr::Reduce::make(
                                      inp[0], opr::Reduce::Param::Mode::MEAN,
                                      opr::GetVarShape::make(inp[1]));
                              return var1 + inp[1];
                          },
                          CompNode::load("gpu0")};
    checker.run({TensorShape{3, 3}, {3, 1}});
}

TEST(TestJITFusionHalide, SameGradOpr) {
    REQUIRE_GPU(1);
    set_backend(Backend::HALIDE);
    auto cn = CompNode::load("gpu0");

    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x0 = gen({3, 3}, cn), host_x1 = gen({3, 1}, cn),
         host_x2 = gen({3, 3}, cn);
    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         c = opr::Host2DeviceCopy::make(*graph, host_x2);

    auto y = (a + b) * c;
    auto reduce_y = opr::reduce_sum(y * y, y.make_scalar(1));
    auto a_grad = opr::VirtualGrad::make(reduce_y.node(), a.node());
    auto b_grad = opr::VirtualGrad::make(reduce_y.node(), b.node());
    auto c_grad = opr::VirtualGrad::make(reduce_y.node(), c.node());

    gopt::GraphOptimizer gopt;
    gopt.add_pass<gopt::JITFusionPass>(true);
    gopt.add_pass<gopt::ExpandVirtualGradPass>();

    VarNodeArray vars{y.node(), a_grad.node(), b_grad.node(), c_grad.node()};
    gopt.apply_inplace(vars);
    ASSERT_EQ(vars[1]->owner_opr()->input(0), vars[2]->owner_opr()->input(0));
    ASSERT_NE(vars[1]->owner_opr()->input(0), vars[3]->owner_opr()->input(0));
}

template <typename tag>
class TestJITHalideFusionCuda : public ::testing::Test {};
TYPED_TEST_CASE(TestJITHalideFusionCuda, test_types);
TYPED_TEST(TestJITHalideFusionCuda, run) {
    set_backend(Backend::NONE);

    REQUIRE_GPU(1);
    run<TypeParam>(Backend::HALIDE, CompNode::load("gpu0"));

    set_backend(Backend::NONE);
}
#endif  // MGB_JIT_HALIDE

template <typename tag>
class TestJITNvrtcFusion : public ::testing::Test {};
TYPED_TEST_CASE(TestJITNvrtcFusion, test_types);
TYPED_TEST(TestJITNvrtcFusion, run) {
    set_backend(Backend::NONE);

    REQUIRE_GPU(1);
    run<TypeParam>(Backend::NVRTC, CompNode::load("gpu0"));

    set_backend(Backend::NONE);
}

TEST(TestJITNvrtcFusion, SourceCache) {
    REQUIRE_GPU(1);
    set_backend(Backend::NVRTC);

    std::string cache_cat;
    std::vector<std::string> sources;
    auto on_cache_get = [&](const std::string& category, const void* key,
                            size_t key_size, const void*, size_t) {
        if (cache_cat.empty()) {
            cache_cat = category;
        } else {
            ASSERT_EQ(cache_cat, category);
        }
        sources.push_back(std::string{static_cast<const char*>(key), key_size});
    };
    PersistentCacheHook cache_hook{on_cache_get};

    auto cn = CompNode::load("gpu0");

    auto run = [cn]() {
        HostTensorGenerator<> gen;
        auto host_x = gen({2, 3}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto x = opr::Host2DeviceCopy::make(graph, host_x),
                 y = jit_stop(x * opr::sin(x)), z = y + opr::tanh(y);
            return z;
        };
        HostTensorND host_y1, host_y2;
        auto funcs = make_func_pair(host_y1, host_y2, make_dst, 2);
        ASSERT_EQ(2u, find_oprs<JITExecutor>(*funcs.second).size());
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);
    };

    for (size_t i = 0; i < 4; ++i) {
        run();
        ASSERT_EQ((i + 1) * 2, sources.size());
        ASSERT_EQ(sources[0], sources[i * 2]);
        ASSERT_EQ(sources[1], sources[i * 2 + 1]);
    }
}

TEST(TestJITNvrtc, DimshuffleFusion) {
    REQUIRE_GPU(1);
    set_backend(Backend::NVRTC);
    auto cn = CompNode::load("gpu0");
    HostTensorGenerator<> gen;
    // single dimshuffle
    {
        auto host_x = gen({2, 3, 8, 8}, cn);
        auto host_w = gen({3, 3, 1, 1}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto data = opr::SharedDeviceTensor::make(graph, *host_x);
            auto w = opr::SharedDeviceTensor::make(graph, *host_w);
            opr::Convolution::Param param;
            auto x = opr::Convolution::make(data, w, param);
            x = opr::relu(x);
            x = opr::Dimshuffle::make(x, {1, 2, 3, 0});
            x = opr::TypeCvt::make(x, dtype::Float16{});
            return x;
        };
        HostTensorND host_y1, host_y2;
        auto funcs = make_func_pair(host_y1, host_y2, make_dst, 1);

        ASSERT_EQ(1u, find_oprs<JITExecutor>(*funcs.second).size());
        ASSERT_EQ(1u, find_oprs<opr::Convolution>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Dimshuffle>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::TypeCvt>(*funcs.second).size());
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);
    }
    // multi dimshuffle in one branch
    {
        auto host_x = gen({3, 4, 6}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto data = opr::SharedDeviceTensor::make(graph, *host_x);
            auto x = opr::relu(data);
            x = opr::Dimshuffle::make(x, {2, 0, 1});
            x = opr::sigmoid(x);
            x = opr::Dimshuffle::make(x, {1, 0, -1, 2});
            x = opr::TypeCvt::make(x, dtype::Float16{});
            return x;
        };
        HostTensorND host_y1, host_y2;
        auto funcs = make_func_pair(host_y1, host_y2, make_dst, 1);
        ASSERT_EQ(1u, find_oprs<JITExecutor>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Dimshuffle>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::TypeCvt>(*funcs.second).size());
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);
    }

    // multi dimshuffle in two branch
    {
        auto host_x = gen({3, 4, 6}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto data = opr::SharedDeviceTensor::make(graph, *host_x);
            auto x = opr::relu(data);
            x = opr::Dimshuffle::make(x, {2, 0, 1});
            x = opr::sigmoid(x);
            x = opr::Dimshuffle::make(x, {1, 0, -1, 2});
            x = opr::TypeCvt::make(x, dtype::Float16{});

            auto y = opr::sigmoid(data);
            y = opr::Dimshuffle::make(y, {0, 2, -1, 1});
            y = opr::TypeCvt::make(y, dtype::Float16{});

            auto z = x + y;
            return z;
        };
        HostTensorND host_y1, host_y2;
        auto funcs = make_func_pair(host_y1, host_y2, make_dst, 1);
        ASSERT_EQ(1u, find_oprs<JITExecutor>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Dimshuffle>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::TypeCvt>(*funcs.second).size());
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_NEAR(host_y1, host_y2, 1e-3);
    }

    // dimshuffle pattern length > 4
    {
        auto host_x = gen({4, 3, 4, 6}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto data = opr::SharedDeviceTensor::make(graph, *host_x);
            auto x = opr::relu(data);
            x = opr::Dimshuffle::make(x, {2, 1, 0, -1, 3});
            x = opr::TypeCvt::make(x, dtype::Float16{});

            return x;
        };
        HostTensorND host_y1, host_y2;
        auto g0 = ComputingGraph::make();
        g0->options().graph_opt_level = 0;
        auto f0 = g0->compile({make_callback_copy(make_dst(*g0), host_y1)});

        auto g1 = ComputingGraph::make();
        g1->options().graph_opt_level = 3;
        g1->options().graph_opt.jit = 1;
        auto f1 = g1->compile({make_callback_copy(make_dst(*g1), host_y2)});

        EXPECT_TRUE(find_oprs<JITExecutor>(*f1).empty());
        f0->execute();
        f1->execute();
        MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);
    }

    // dimshuffle is endpoint
    {
        auto host_x = gen({4, 3, 4, 6}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto x = opr::TypeCvt::make(
                    opr::Host2DeviceCopy::make(graph, host_x),
                    dtype::Float16{});
            auto y = opr::Dimshuffle::make(x, {3, 0, 1, 2});
            return y;
        };
        HostTensorND host_y;
        auto g1 = ComputingGraph::make();
        g1->options().graph_opt_level = 3;
        g1->options().graph_opt.jit = 1;
        auto f1 = g1->compile({make_callback_copy(make_dst(*g1), host_y)});
        EXPECT_TRUE(find_oprs<JITExecutor>(*f1).empty());
    }
}

TEST(TestJITNvrtc, DimshuffleGrad) {
    REQUIRE_GPU(1);
    set_backend(Backend::NVRTC);
    auto cn = CompNode::load("gpu0");
    HostTensorGenerator<> gen;
    // single dimshuffle
    {
        auto host_x = gen({2, 3, 8, 8}, cn);
        auto host_w = gen({3, 3, 1, 1}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto data = opr::SharedDeviceTensor::make(graph, *host_x);
            auto w = opr::SharedDeviceTensor::make(graph, *host_w);
            opr::Convolution::Param param;
            auto x = opr::Convolution::make(data, w, param);
            x = opr::relu(x);
            x = opr::Dimshuffle::make(x, {1, 2, 3, 0});
            x = opr::TypeCvt::make(x, dtype::Float16{});
            auto loss = opr::reduce_sum(x, x.make_scalar(1));
            auto grad = cg::grad(loss, w);
            return grad;
        };
        HostTensorND host_y1, host_y2;
        auto funcs = make_func_pair(host_y1, host_y2, make_dst, 1);

        ASSERT_EQ(1u, find_oprs<JITExecutor>(*funcs.second).size());
        ASSERT_EQ(1u, find_oprs<opr::Convolution>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Dimshuffle>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::TypeCvt>(*funcs.second).size());
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);
    }
    // multi dimshuffle in two branch
    {
        auto host_x = gen({3, 4, 6}, cn);
        auto make_dst = [&](ComputingGraph& graph) {
            auto data = opr::SharedDeviceTensor::make(graph, *host_x);
            auto x = opr::relu(data);
            x = opr::Dimshuffle::make(x, {2, 0, 1});
            x = opr::sigmoid(x);
            x = opr::Dimshuffle::make(x, {1, 0, -1, 2});
            x = opr::TypeCvt::make(x, dtype::Float16{});

            auto y = opr::sigmoid(data);
            y = opr::Dimshuffle::make(y, {0, 2, -1, 1});
            y = opr::TypeCvt::make(y, dtype::Float16{});

            auto z = x + y;
            auto loss = opr::reduce_sum(z, z.make_scalar(1));
            auto grad = cg::grad(loss, data);
            return grad;
        };
        HostTensorND host_y1, host_y2;
        auto funcs = make_func_pair(host_y1, host_y2, make_dst, 1);
        ASSERT_EQ(1u, find_oprs<JITExecutor>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::Dimshuffle>(*funcs.second).size());
        ASSERT_EQ(0u, find_oprs<opr::TypeCvt>(*funcs.second).size());
        funcs.first->execute();
        funcs.second->execute();
        MGB_ASSERT_TENSOR_NEAR(host_y1, host_y2, 1e-3);
    }
    {
        FusionChecker checker{2,
            [](const SymbolVarArray& inp) -> SymbolVar {
                auto var = opr::Dimshuffle::make(inp[0], {1, 2, 3, 0});
                return inp[1] * var;
            },
            CompNode::load("gpu0")};
        checker.set_jit_level(1)
               .run({TensorShape{1, 2, 3, 4}, {2, 3, 4, 1}})
               .run({TensorShape{3, 4, 1, 2}, {4, 1, 2, 3}})
               .run({TensorShape{4, 6, 3, 5}, {6, 3, 5, 4}});
    }
}

TEST(TestJITExecutor, GradBehavior) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    HostTensorGenerator<> gen;
    {
        set_backend(Backend::NVRTC);
        auto graph = ComputingGraph::make();
        auto host_a = gen({2, 3, 4}, cn);
        auto a = opr::Host2DeviceCopy::make(*graph, host_a),
            x = opr::exp(a + 1);

        gopt::GraphOptimizer gopt;
        gopt.add_pass<gopt::JITFusionPass>();
        VarNodeArray dest_vars{x.node()};
        gopt.apply_inplace(dest_vars);
        x = opr::reduce_sum(dest_vars[0], a.make_scalar_dt(1));
        SmallVector<jit::JITExecutor*> jits;
        auto on_opr = [&jits](cg::OperatorNodeBase* op) {
            if (auto jit = op->try_cast_final<jit::JITExecutor>()) {
                jits.push_back(jit);
            }
        };
        auto grad_a = cg::grad(x, a);
        cg::DepOprIter{on_opr}.add(grad_a);
        ASSERT_EQ(jits.size(), 2);
        // input of forward jit executor: host_a
        ASSERT_EQ(jits[0]->input().size(), 1);
        // input of grad jit executor:
        //      output of forward jit executor, output grad
        ASSERT_EQ(jits[1]->input().size(), 2);
        // internal graph is (input: og, out | output: og * out)
        size_t nr_ph = 0, nr_mul = 0;
        cg::DepOprIter{
            [&nr_ph, &nr_mul](cg::OperatorNodeBase* op) {
                if (op->same_type<jit::JITPlaceholder>()) {
                    ++ nr_ph;
                    return;
                }
                if(auto mul = op->try_cast_final<opr::Elemwise>()) {
                    using Mode = opr::Elemwise::Mode;
                    if (mul->param().mode == Mode::MUL) {
                        ++ nr_mul;
                        return;
                    }
                }
                mgb_throw(MegBrainError, "unexpected op %s", op->cname());
            }}
            .add(jits[1]->internal_graph_ptr()->output());
        ASSERT_EQ(nr_ph, 2);
        ASSERT_EQ(nr_mul, 1);
    }
#if MGB_JIT_HALIDE
    {
        set_backend(Backend::HALIDE);
        auto graph = ComputingGraph::make();
        auto host_a = gen({2, 3, 4}, cn);
        auto a = opr::Host2DeviceCopy::make(*graph, host_a),
            x = opr::exp(a + 1);

        gopt::GraphOptimizer gopt;
        gopt.add_pass<gopt::JITFusionPass>();
        VarNodeArray dest_vars{x.node()};
        gopt.apply_inplace(dest_vars);
        x = opr::reduce_sum(dest_vars[0], a.make_scalar_dt(1));
        size_t nr_ops = 0, nr_jits = 0;
        auto on_opr = [&nr_jits, &nr_ops](cg::OperatorNodeBase* op) {
            if (op->same_type<jit::JITExecutor>()) {
                ++ nr_jits;
            }
            ++ nr_ops;
        };
        auto grad_a = cg::grad(x, a);
        cg::DepOprIter{on_opr}.add(grad_a);
        // in Halide backend, grad internal graph would be expanded into
        // original graph, so there was only one JITExecutor
        ASSERT_EQ(nr_jits, 1);
        // the grad of a is broadcast(JITExecutor.output(0), a.shape()),
        // so the oprs depended by grad_a are H2D(a), JITExecutor,
        // GetVarShape(a) and broadcast
        ASSERT_EQ(nr_ops, 4);
    }
#endif // MGB_JIT_HALIDE
    {
        set_backend(Backend::NVRTC);
        auto graph = ComputingGraph::make();
        auto host_a = gen({2, 3, 4}, cn);
        auto a = opr::SharedDeviceTensor::make(*graph, *host_a),
            x = a * 2 + 1;

        gopt::GraphOptimizer gopt;
        gopt.add_pass<gopt::JITFusionPass>();
        VarNodeArray dest_vars{x.node()};
        gopt.apply_inplace(dest_vars);
        x = opr::reduce_sum(dest_vars[0], a.make_scalar_dt(1));
        auto grad_a = cg::grad(x, a);
        // all inputs of grad jit executor are const, its internal graph
        // would be expanded into original graph for more optimizations,
        // so no JITExecutor can be found
        cg::DepOprIter{[](cg::OperatorNodeBase* op) {
            ASSERT_FALSE(op->same_type<jit::JITExecutor>());}
        }.add(grad_a);
    }
}

#if MGB_JIT_MLIR

void run_mlir(CompNode cn) {
    set_backend(Backend::MLIR);

    HostTensorGenerator<> gen;
    auto host_x0 = gen({23, 42}, cn), host_x1 = gen({23, 1}, cn),
         host_x2 = gen({1, 42}, cn), host_x3 = gen({23, 42}, cn),
         host_x4 = gen({1, 42}, cn), host_x5 = gen({23, 1}, cn);

    auto make_dst = [&](ComputingGraph& graph) {
        auto a = opr::Host2DeviceCopy::make(graph, host_x0),
         b = opr::Host2DeviceCopy::make(graph, host_x1),
         c = opr::Host2DeviceCopy::make(graph, host_x2),
         d = opr::Host2DeviceCopy::make(graph, host_x3),
         e = opr::Host2DeviceCopy::make(graph, host_x4);
        return a + opr::max(b, c) + opr::max(d, e);
    };
    HostTensorND host_y1, host_y2;
    auto funcs = make_func_pair(host_y1, host_y2, make_dst, 2);

    funcs.first->execute();
    funcs.second->execute();
    MGB_ASSERT_TENSOR_EQ(host_y1, host_y2);

    JITExecutor* jit;
    unpack_vector(find_oprs<JITExecutor>(*funcs.second), jit);
    ASSERT_EQ(0u, find_oprs<opr::Elemwise>(*funcs.second).size());
    ASSERT_EQ(5u, jit->input().size());
}

TEST(TestJITExecutor, TestJITMlirFusion) {
    run_mlir(CompNode::load("cpu0"));
}

TEST(TestJITExecutor, TestJITMlirFusionGpu) {
    REQUIRE_GPU(1);
    run_mlir(CompNode::load("gpu0"));
}

#endif // MGB_JIT_MLIR

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
