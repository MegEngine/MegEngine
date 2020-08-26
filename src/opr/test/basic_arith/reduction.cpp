/**
 * \file src/opr/test/basic_arith/reduction.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/autocheck.h"
#include "megbrain/test/megdnn_helper.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/tensor_manip.h"

#include "megdnn/tensor_iter.h"

#include <algorithm>

using namespace mgb;

namespace {

    using Mode = opr::Reduce::Mode;
    using DataType = opr::Reduce::Param::DataType;

    template<Mode mode, typename ctype>
    struct ImplTrait {
    };

    template<typename ctype>
    struct ImplTrait<Mode::SUM, ctype> {
        static constexpr float GRAD_MAXERR = 1e-4, GRAD_EPS = 1;

        static ctype init() {
            return 0;
        }

        static ctype reduce(ctype accum, ctype v) {
            return accum + v;
        }

        ctype finalize(ctype result) {
            return result;
        }
    };

    template<typename ctype>
    struct ImplTrait<Mode::SUM_SQR, ctype> {
        static constexpr float GRAD_MAXERR = 1e-3, GRAD_EPS = 0.01;

        static ctype init() {
            return 0;
        }

        static ctype reduce(ctype accum, ctype v) {
            return accum + v * v;
        }

        ctype finalize(ctype result) {
            return result;
        }
    };

    template<typename ctype>
    struct ImplTrait<Mode::PRODUCT, ctype> {
        static constexpr float GRAD_MAXERR = 1e-4, GRAD_EPS = 0.01;

        static ctype init() {
            return 1;
        }

        static ctype reduce(ctype accum, ctype v) {
            return accum * v;
        }

        ctype finalize(ctype result) {
            return result;
        }
    };

    template<typename ctype>
    struct ImplTrait<Mode::MAX, ctype> {
        static constexpr float GRAD_MAXERR = 1e-2, GRAD_EPS = 1e-3;

        static ctype init() {
           return std::numeric_limits<ctype>::lowest();
        }

        static ctype reduce(ctype accum, ctype v) {
            return std::max(accum, v);
        }

        ctype finalize(ctype result) {
            return result;
        }
    };

    template<typename ctype>
    struct ImplTrait<Mode::MIN, ctype> {
        static constexpr float GRAD_MAXERR = 1e-2, GRAD_EPS = 1e-3;

        static ctype init() {
            return std::numeric_limits<ctype>::max();
        }

        static ctype reduce(ctype accum, ctype v) {
            return std::min(accum, v);
        }

        ctype finalize(ctype result) {
            return result;
        }
    };

    template<typename ctype>
    struct ImplTrait<Mode::MEAN, ctype> {
        static constexpr float GRAD_MAXERR = 1e-4, GRAD_EPS = 1e-2;
        size_t nr_elems;

        ctype init() {
            nr_elems = 0;
            return 0;
        }

        ctype reduce(ctype accum, ctype v) {
            nr_elems ++;
            return accum + v;
        }

        ctype finalize(ctype result) {
            return result / static_cast<ctype>(nr_elems);
        }
    };

    template<Mode mode, typename ctype>
    void reduce_raw(HostTensorND &dest, const HostTensorND &src) {
        auto tshp = dest.shape();
        using Impl = ImplTrait<mode, ctype>;

        if (tshp.is_scalar()) {
            if (src.shape().is_scalar()) {
                dest.copy_from_fixlayout(src);
                return;
            }

            Impl impl;
            ctype val = impl.init();
            for (auto i: megdnn::tensor_iter_valonly<ctype>(src.as_megdnn()))
                val = impl.reduce(val, i);
            dest.ptr<ctype>()[0] = impl.finalize(val);
            return;
        }

        mgb_assert(tshp.ndim == src.shape().ndim);

        std::vector<size_t> axis_to_use;
        for (size_t i = 0; i < tshp.ndim; i ++) {
            if (tshp.shape[i] != src.shape(i)) {
                mgb_assert(tshp.shape[i] == 1);
                axis_to_use.push_back(i);
            }
        }

        if (axis_to_use.empty()) {
            dest.copy_from_fixlayout(src);
            return;
        }
        TensorLayout sub_layout{dest.dtype()};
        sub_layout.ndim = axis_to_use.size();
        for (size_t i = 0; i < axis_to_use.size(); i ++) {
            sub_layout.shape[i] = src.layout().shape[axis_to_use[i]];
            sub_layout.stride[i] = src.layout().stride[axis_to_use[i]];
        }

        auto diter_maker = megdnn::tensor_iter<ctype>(dest.as_megdnn());
        for (auto iter = diter_maker.begin(), iter_end = diter_maker.end();
                iter != iter_end; ++ iter) {
            ptrdiff_t offset = 0;
            for (size_t i = 0; i < tshp.ndim; i ++)
                offset += iter.idx()[i] * src.layout().stride[i];

            Impl impl;
            ctype val = impl.init();
            auto subspec = SubTensorSpec::make_from_offset_elem(
                    sub_layout, offset);
            HostTensorND subt = const_cast<HostTensorND&>(src).sub(subspec);
            for (ctype i:
                    megdnn::tensor_iter_valonly<ctype>(subt.as_megdnn())) {
                val = impl.reduce(val, i);
            }
            *iter = impl.finalize(val);
        }
    }

    template<Mode mode, class dtype>
    void do_test_correctness() {
        using ctype = typename DTypeTrait<dtype>::ctype;
        using Impl = ImplTrait<mode, ctype>;

        using Checker = AutoOprChecker<1, 1, dtype>;
        constexpr int AXIS = 1;

        auto make_graph = [&](const typename Checker::SymInpArray &inputs) ->
            typename Checker::SymOutArray
        {
            return {opr::Reduce::make(inputs[0], {mode, AXIS})};
        };
        auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
            TensorShape oshp = inp[0]->shape();
            oshp.shape[1] = 1;
            dest[0].resize(oshp);
            reduce_raw<mode, ctype>(dest[0], *inp[0]);
        };

        typename Checker::RunOptions opt;
        opt.numdiff_eps = Impl::GRAD_EPS;
        opt.numdiff_max_err = Impl::GRAD_MAXERR;
        using S = TensorShape;
        Checker{make_graph, fwd}.
            run({S{2, 3, 4}}, opt).
            run({S{2, 2, 3, 4}}, opt).
            run({S{2, 3, 4, 3}}, opt);
    }

    template<Mode mode>
    void test_correctness() {
        set_rand_seed(19931102);
        do_test_correctness<mode, dtype::Float32>();
        do_test_correctness<mode, dtype::Int32>();
    }

    void test_base_impl(bool dyn_inp, bool dyn_tshp) {
        HostTensorGenerator<> gen;
        auto host_x = gen({10});
        auto host_tshp = std::make_shared<HostTensorND>(
                host_x->comp_node(), dtype::Int32());

        host_tshp->resize({1}).ptr<int>()[0] = 1;
        HostTensorND host_y, expected{host_x->comp_node(), dtype::Float32()};
        DeviceTensorND static_calc_x{CompNode::default_cpu()},
                       static_calc_workspace{CompNode::default_cpu()},
                       static_calc_y{CompNode::default_cpu()};
        auto static_calc_opr = opr::intl::create_megdnn_opr<megdnn::Reduce>(
                CompNode::default_cpu());
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
             tshp = opr::Host2DeviceCopy::make(*graph, host_tshp, {"tshp"});
        if (dyn_inp)
            x = opr::MarkDynamicVar::make(x);
        if (dyn_tshp)
            tshp = opr::MarkDynamicVar::make(tshp);
        auto y = opr::reduce_sum(x, tshp);
        auto func = graph->compile({make_callback_copy(y, host_y)});

        if (!dyn_tshp) {
            ASSERT_TRUE(cg::is_static_var_shape(y.node()));
        }
        if (!dyn_inp && !dyn_tshp) {
            ASSERT_TRUE(cg::is_static_var_value(y.node()));
        }

        bool check_succ = false;
        auto do_check = [&](const TensorShape &ishp,
                const std::vector<size_t> &reduce_axes) {
            check_succ = false;
            host_x->copy_from(*gen(ishp));
            auto oshp = ishp;
            if (reduce_axes.size() == 1 && reduce_axes[0] == (size_t)-1) {
                oshp.shape[0] = 1;
                oshp.ndim = 1;
            } else {
                for (auto i: reduce_axes)
                    oshp.shape[i] = 1;
            }
            {
                DeviceTensorND tmp;
                cg::copy_shape_to_tensor_value(tmp, oshp);
                host_tshp->copy_from(tmp);
            }
            func->execute();

            if (reduce_axes.empty() && !(!dyn_inp && dyn_tshp)) {
                ASSERT_EQ(x.node()->prev_dev_ptr(), y.node()->prev_dev_ptr());
            }

            expected.resize(oshp);
            reduce_raw<Mode::SUM, float>(expected, *host_x);

            MGB_ASSERT_TENSOR_NEAR(expected, host_y, 1e-5);

            static_calc_x.copy_from(*host_x);
            opr::Reduce::perform(
                    Mode::SUM, static_calc_y, static_calc_workspace,
                    static_calc_x, oshp, static_calc_opr);
            host_y.ptr<float>()[0] ++;
            host_y.copy_from(static_calc_y);
            MGB_ASSERT_TENSOR_NEAR(expected, host_y, 1e-5);

            check_succ = true;
        };

        auto check = [&](const TensorShape &ishp,
                const std::vector<size_t> &reduce_axes) {
            do_check(ishp, reduce_axes);
            mgb_assert(check_succ);
        };

        check({1, 2}, {size_t(-1)});
        check({1, 2}, {});
        check({1}, {});

        check({2}, {0});
        check({2, 3}, {0, 1});
        check({2, 3, 4}, {0, 1, 2});
        check({2, 3, 4, 5}, {0, 1, 2, 3});
        check({2, 3, 4, 5, 6}, {0, 1, 2, 3, 4});
        check({2, 3, 4, 5, 6}, {size_t(-1)});
        check({1, 1, 1}, {size_t(-1)});

        check({1, 2, 3, 4}, {});

        for (size_t i = 0; i < 4; i ++)
            check({3, 2, 5, 6}, {i});

        for (size_t i = 0; i < 4; i ++)
            for (size_t j = i + 1; j < 4; j ++)
                check({4, 2, 6, 7}, {i, j});

        for (size_t i = 0; i < 5; i ++)
            for (size_t j = i + 1; j < 5; j ++)
                for (size_t k = j + 1; k < 5; k ++)
                    check({4, 5, 2, 7, 2}, {i, j, k});

        check({100, 100, 32}, {1});
    }

} // anonymous namespace

TEST(TestBasicArithReduction, BaseImpl00) {
    test_base_impl(false, false);
}

TEST(TestBasicArithReduction, BaseImpl01) {
    test_base_impl(false, true);
}

TEST(TestBasicArithReduction, BaseImpl10) {
    test_base_impl(true, false);
}

TEST(TestBasicArithReduction, BaseImpl11) {
    test_base_impl(true, true);
}

TEST(TestBasicArithReduction, AxisOnly) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 6, 7, 8});
    for (bool dyn: {false, true}) {
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        if (dyn)
            x = opr::MarkDynamicVar::make(x);
        auto y = opr::Reduce::make(x, {Mode::SUM, 1});
        HostTensorND host_y, expected{host_x->comp_node(), host_x->dtype()};
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        expected.resize({2, 1, 7, 8});
        reduce_raw<Mode::SUM, float>(expected, *host_x);
        MGB_ASSERT_TENSOR_EQ(expected, host_y);
    }
}

TEST(TestBasicArithReduction, NegativeAxis) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 6, 7, 8});
    for (bool dyn : {false, true})
        for (int i = 0; i < 4; i++) {
            auto graph = ComputingGraph::make();
            auto x = opr::Host2DeviceCopy::make(*graph, host_x);
            if (dyn)
                x = opr::MarkDynamicVar::make(x);
            auto y = opr::Reduce::make(x, {Mode::SUM, i - 4});
            HostTensorND host_y, expected{host_x->comp_node(), host_x->dtype()};
            auto func = graph->compile({make_callback_copy(y, host_y)});
            func->execute();
            megdnn::TensorShape tshp({2, 6, 7, 8});
            tshp.shape[i] = 1;
            expected.resize(tshp);
            reduce_raw<Mode::SUM, float>(expected, *host_x);
            MGB_ASSERT_TENSOR_EQ(expected, host_y);
        }
}

TEST(TestBasicArithReduction, NonCont) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    for (int dyn = 0; dyn < 4; ++ dyn) {
        auto host_x = gen({2, 1});
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             xnt = x.broadcast({2, 4}),
             tshp = x.make_scalar(1);
        if (dyn & 3)
            xnt = opr::MarkDynamicVar::make(xnt);
        if (dyn & 1)
            tshp = opr::MarkDynamicVar::make(tshp);
        auto y = opr::reduce_sum(xnt, tshp);
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_TRUE(host_y.shape().is_scalar());
        auto xp = host_x->ptr<float>();
        MGB_ASSERT_FLOAT_EQ((xp[0] + xp[1]) * 4, host_y.ptr<float>()[0]);
    }
}

TEST(TestBasicArithReduction, NonContFwd) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    for (int dyn = 0; dyn < 4; ++ dyn) {
        auto host_x = gen({2, 1});
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             xnt = x.broadcast({2, 4}),
             tshp = xnt.symshape();
        if (dyn & 3)
            xnt = opr::MarkDynamicVar::make(xnt);
        if (dyn & 1)
            tshp = opr::MarkDynamicVar::make(tshp);
        auto y = opr::reduce_sum(xnt, tshp);
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({2, 4}), host_y.shape());
        for (size_t i = 0; i < 2; ++ i) {
            for (size_t j = 0; j < 4; ++ j) {
                MGB_ASSERT_FLOAT_EQ(
                        host_x->ptr<float>()[i],
                        host_y.ptr<float>({i, j})[0]);
            }
        }
        if (!dyn) {
            ASSERT_EQ(dev_ptr(x), dev_ptr(xnt));
            ASSERT_EQ(dev_ptr(x), dev_ptr(y));
        }
        if (dyn == 3) {
            ASSERT_EQ(xnt.node()->prev_dev_ptr(), y.node()->prev_dev_ptr());
        }
    }
}

TEST(TestBasicArithReduction, NonContPerform) {
    DeviceTensorND x{CompNode::default_cpu(), dtype::Float32()},
                   y{x.comp_node(), x.dtype()}, workspace;
    x.resize({1}).ptr<float>()[0] = 2.3;
    x.reset(x.storage(), x.layout().broadcast({5, 5}));
    auto opr = opr::intl::create_megdnn_opr<megdnn::Reduce>(x.comp_node());

    float x0_val = 2.3;
    for (auto mode: {Mode::SUM, Mode::SUM_SQR}) {
        for (auto &&tshp:
                TensorShapeArray{{5, 1}, {1, 5}, {1, 1}, {1}, {5, 5}}) {

            opr::Reduce::perform(mode, y, workspace, x, tshp, opr);
            ASSERT_TRUE(y.layout().is_contiguous());
            ASSERT_EQ(tshp, y.shape());
            size_t nr = tshp.total_nr_elems();
            float expect = x0_val * 25 / nr;
            auto py = y.ptr<float>();
            for (size_t i = 0; i < nr; ++ i)
                MGB_ASSERT_FLOAT_EQ(expect, py[i]);
        }
        x0_val *= 2.3;
    }
}

TEST(TestBasicArithReduction, SideEffect) {
    using Checker = AutoOprChecker<1, 2>;

    auto make_graph = [&](const Checker::SymInpArray &inputs, bool scalar) ->
        Checker::SymOutArray
    {
        auto x = inputs[0];
        auto y0_shp = opr::GetVarShape::make(x);
        opr::Subtensor::IndexDesc desc{
            opr::Subtensor::AxisIndexer::make_index(0, x.make_scalar(1))};
        auto y1_shp = opr::SetSubtensor::make(y0_shp.fill_retain_dtype(1),
                opr::Subtensor::make(y0_shp, desc), desc);
        if (scalar) {
            y1_shp = y1_shp.make_scalar(1);
        }
        return {opr::reduce_sum_sqr(x, y0_shp), opr::reduce_sum_sqr(x, y1_shp)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp, bool scalar) {
        auto&& x = *inp[0], &&y0 = dest[0], &&y1 = dest[1];
        y0.copy_from(x);
        auto py0 = y0.ptr<float>();
        for (size_t i = 0, it = x.shape().total_nr_elems(); i < it; ++ i) {
            py0[i] *= py0[i];
        }

        auto y1_shp = y0.shape();
        for (size_t i = 0; i < y1_shp.ndim; ++ i) {
            if (i != 1)
                y1_shp[i] = 1;
        }
        if (scalar) {
            y1_shp.ndim = 1;
            y1_shp[0] = 1;
        }
        reduce_raw<opr::Reduce::Mode::SUM, dt_float32>(y1.resize(y1_shp), y0);
    };

    using S = TensorShape;
    for(auto &&scalar: {false, true}) {
        using namespace std::placeholders;
        Checker{std::bind(make_graph, _1, scalar),
                std::bind(fwd, _1, _2, scalar)}.
            run({S{2, 3, 4}}).
            run({S{2, 2, 3, 4}}).
            run({S{3, 3, 2, 3}}).
            run({S{1, 1}});
    }
}

TEST(TestBasicArithReduction, DifferentNDim) {
    HostTensorGenerator<> gen;
    for (size_t first_dim = 1; first_dim <= 2; ++ first_dim) {
        auto host_x = gen({first_dim, 64, 22, 22});
        auto host_tshp =
                std::make_shared<HostTensorND>(host_x->comp_node(), dtype::Int32());
        host_tshp->resize({3});
        host_tshp->ptr<int>()[0] = 64;
        host_tshp->ptr<int>()[1] = 22;
        host_tshp->ptr<int>()[2] = 22;

        auto host_tshp_equal =
                std::make_shared<HostTensorND>(host_x->comp_node(), dtype::Int32());
        host_tshp_equal->resize({4});
        host_tshp_equal->ptr<int>()[0] = 1;
        host_tshp_equal->ptr<int>()[1] = 64;
        host_tshp_equal->ptr<int>()[2] = 22;
        host_tshp_equal->ptr<int>()[3] = 22;

        using namespace opr;

        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;

        auto x = opr::relu(
                     opr::Host2DeviceCopy::make(*graph, host_x, {"x"}));
        auto tshp = opr::Host2DeviceCopy::make(*graph, host_tshp, {"tshp"});
        auto tshp_equal = opr::Host2DeviceCopy::make(*graph, host_tshp_equal, {"tshp_equal"});

        auto check_mode = [&](Reduce::Mode mode) {
            Reduce::Param param_default{mode, MEGDNN_MAX_NDIM,
                                        Reduce::Param::DataType::DEFAULT};
            auto reduce_default = opr::Reduce::make(x, param_default, tshp);
            auto reduce_equal = opr::Reshape::make(opr::Reduce::make(x, param_default, tshp_equal), tshp);

            HostTensorND host_default;
            HostTensorND host_equal;
            auto func = graph->compile(
                    {make_callback_copy(reduce_default, host_default),
                    make_callback_copy(reduce_equal, host_equal)});

            func->execute();
            MGB_ASSERT_TENSOR_EQ(host_default, host_equal);
        };

        for (auto mode :
             {Reduce::Mode::PRODUCT, Reduce::Mode::MAX, Reduce::Mode::MIN,
              Reduce::Mode::SUM, Reduce::Mode::SUM_SQR, Reduce::Mode::MEAN}) {
            check_mode(mode);
        }
    }
}

TEST(TestBasicArithReduction, MultiType) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 64, 22, 22});
    auto host_tshp =
            std::make_shared<HostTensorND>(host_x->comp_node(), dtype::Int32());

    host_tshp->resize({4});
    host_tshp->ptr<int>()[0] = 1;
    host_tshp->ptr<int>()[1] = 64;
    host_tshp->ptr<int>()[2] = 1;
    host_tshp->ptr<int>()[3] = 1;

    using namespace opr;

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto x_fp16 = opr::relu(opr::TypeCvt::make(
                 opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
                 dtype::Float16())),
         tshp = opr::Host2DeviceCopy::make(*graph, host_tshp, {"tshp"});

    auto x = opr::TypeCvt::make(x_fp16, dtype::Float32());

    auto check_mode = [&](Reduce::Mode mode) {
        Reduce::Param param_default{mode, MEGDNN_MAX_NDIM,
                                    Reduce::Param::DataType::DEFAULT};
        Reduce::Param param_i16_co32{mode, MEGDNN_MAX_NDIM,
                                     Reduce::Param::DataType::FLOAT_O32xC32};
        Reduce::Param param_io16_c32{mode, MEGDNN_MAX_NDIM,
                                     Reduce::Param::DataType::FLOAT_O16xC32};

        auto reduce_default = opr::Reduce::make(x, param_default, tshp);
        auto reduce_i16_co32 = opr::Reduce::make(x_fp16, param_i16_co32, tshp);
        auto reduce_io16_c32 = opr::Reduce::make(x_fp16, param_io16_c32, tshp);
        auto reduce_default_as16 =
                opr::TypeCvt::make(reduce_default, dtype::Float16());

        HostTensorND host_default, host_default_as16, host_i16_co32,
                host_io16_c32;

        auto func = graph->compile(
                {make_callback_copy(reduce_default, host_default),
                 make_callback_copy(reduce_i16_co32, host_i16_co32),
                 make_callback_copy(reduce_io16_c32, host_io16_c32),
                 make_callback_copy(reduce_default_as16, host_default_as16)});

        func->execute();

        MGB_ASSERT_TENSOR_EQ(host_default, host_i16_co32);
        MGB_ASSERT_TENSOR_EQ(host_default_as16, host_io16_c32);
    };

    for (auto mode :
         {//Reduce::Mode::PRODUCT, Reduce::Mode::MAX, Reduce::Mode::MIN,
         // Reduce::Mode::SUM,
          Reduce::Mode::SUM_SQR}) {
        check_mode(mode);
    }
    host_tshp->ptr<int>()[0] = 1;
    host_tshp->ptr<int>()[1] = 64;
    host_tshp->ptr<int>()[2] = 22;
    host_tshp->ptr<int>()[3] = 22;
    for (auto mode :
         {Reduce::Mode::PRODUCT, Reduce::Mode::MAX, Reduce::Mode::MIN,
          Reduce::Mode::SUM, Reduce::Mode::SUM_SQR, Reduce::Mode::MEAN}) {
        check_mode(mode);
    }
}

TEST(TestBasicArithReduction, C32VsC16) {
    HostTensorGenerator<> gen(1.f, 2.f);
    auto host_x = gen({1, 32, 100000, 2});
    auto host_tshp =
            std::make_shared<HostTensorND>(host_x->comp_node(), dtype::Int32());

    host_tshp->resize({4});
    host_tshp->ptr<int>()[0] = 1;
    host_tshp->ptr<int>()[1] = 32;
    host_tshp->ptr<int>()[2] = 1;
    host_tshp->ptr<int>()[3] = 1;

    using namespace opr;

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto x_fp16 = opr::relu(opr::TypeCvt::make(
                 opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
                 dtype::Float16())),
         tshp = opr::Host2DeviceCopy::make(*graph, host_tshp, {"tshp"});

    x_fp16 = opr::Concat::make({x_fp16, -x_fp16}, 0);

    auto x = opr::TypeCvt::make(x_fp16, dtype::Float32());

    Reduce::Param::Mode mode = Reduce::Param::Mode::SUM;

    Reduce::Param param_default{mode, MEGDNN_MAX_NDIM,
                                Reduce::Param::DataType::DEFAULT};
    Reduce::Param param_i16_co32{mode, MEGDNN_MAX_NDIM,
                                 Reduce::Param::DataType::FLOAT_O32xC32};
    Reduce::Param param_io16_c32{mode, MEGDNN_MAX_NDIM,
                                 Reduce::Param::DataType::FLOAT_O16xC32};

    auto reduce_default = opr::Reduce::make(x, param_default, tshp);
    auto reduce_i16_co32 = opr::Reduce::make(x_fp16, param_i16_co32, tshp);
    auto reduce_io16_c32 = opr::Reduce::make(x_fp16, param_io16_c32, tshp);
    auto reduce_default_as16 =
            opr::TypeCvt::make(reduce_default, dtype::Float16());
    auto bad = opr::Reduce::make(x_fp16, param_default, tshp);

    HostTensorND host_default, host_default_as16, host_i16_co32, host_io16_c32,
            host_bad;

    auto func = graph->compile(
            {make_callback_copy(reduce_default, host_default),
             make_callback_copy(reduce_i16_co32, host_i16_co32),
             make_callback_copy(reduce_io16_c32, host_io16_c32),
             make_callback_copy(reduce_default_as16, host_default_as16),
             make_callback_copy(bad, host_bad)});

    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_default, host_i16_co32);
    MGB_ASSERT_TENSOR_EQ(host_default_as16, host_io16_c32);

    for (size_t i = 0; i < host_io16_c32.shape().total_nr_elems(); ++i) {
        float a = host_io16_c32.ptr<dt_float16>()[i];
        float b = host_bad.ptr<dt_float16>()[i];
        ASSERT_TRUE(std::isfinite(a));
        ASSERT_FALSE(std::isfinite(b));
    }
}

TEST(TestBasicArithReduction, AutoCheck) {
    using Checker = AutoOprChecker<2, 1>;
    using Param = opr::Reduce::Param;

    Param param;

    auto make_graph = [&param](const Checker::SymInpArray& inputs, DType dtype)
            -> Checker::SymOutArray {
        auto inp = inputs[0];
        auto tshp = inputs[1].symshape();
        inp = opr::TypeCvt::make(inp, dtype);
        return {opr::Reduce::make(inp, param, tshp)};
    };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp,
                DType dtype) {
        auto cn = inp[0]->storage().comp_node();
        TensorShape out_shape = inp[1]->shape();
        dest[0] = HostTensorND{cn, out_shape, dtype::Float32()};
        HostTensorND tmp_inp{cn, inp[0]->shape(), dtype};
        HostTensorND new_inp{cn, inp[0]->shape(), dtype::Float32()};
        auto typecvt =
                megdnn_naive_handle()->create_operator<megdnn::TypeCvt>();
        typecvt->exec(inp[0]->as_megdnn(), tmp_inp.as_megdnn());
        typecvt->exec(tmp_inp.as_megdnn(), new_inp.as_megdnn());

#define dispatch_by_mode(CTYPE, MODE, in, out) \
    if (MODE == param.mode) {                  \
        reduce_raw<MODE, CTYPE>(out, in);      \
    }
#define dispatch_by_dtype(DTYPE, in, out)            \
    mgb_assert(DTYPE() == (in).dtype());             \
    typedef DTypeTrait<DTYPE>::ctype ctype;          \
    dispatch_by_mode(ctype, Mode::MIN, in, out);     \
    dispatch_by_mode(ctype, Mode::MAX, in, out);     \
    dispatch_by_mode(ctype, Mode::SUM, in, out);     \
    dispatch_by_mode(ctype, Mode::PRODUCT, in, out); \
    dispatch_by_mode(ctype, Mode::SUM_SQR, in, out); \
    dispatch_by_mode(ctype, Mode::MEAN, in, out);

        mgb_assert(param.data_type == Param::DataType::FLOAT_O32xC32);
        dispatch_by_dtype(dtype::Float32, new_inp, dest[0]);
#undef dispatch_by_mode
#undef dispatch_by_dtype
    };

    auto check = [&](Mode mode, Param::DataType data_type, DType dtype) {
        param.mode = mode;
        param.data_type = data_type;
        Checker::RunOptions opts;
        opts.outputs_max_err = 1e-3;
        opts.numdiff_max_err = 5e-1;
        using namespace std::placeholders;
        Checker checker(std::bind(make_graph, _1, dtype),
                        std::bind(fwd, _1, _2, dtype));
        if (dtype.category() == DTypeCategory::FLOAT) {
            checker.set_input_allow_grad(1, false);
        } else {
            checker.disable_grad_check();
        }
        checker.run({TensorShape{22, 21}, {22, 1}}, opts)
               .run({TensorShape{22, 21}, {1, 1}}, opts)
               .run({TensorShape{22, 21}, {22, 1}}, opts);
    };

    for (auto mode :
         {Mode::SUM, Mode::MAX, Mode::MIN, Mode::PRODUCT, Mode::MEAN}) {
        check(mode, Param::DataType::FLOAT_O32xC32, dtype::Float16());
        check(mode, Param::DataType::FLOAT_O32xC32, dtype::Int32());
    }
}

#define OPR_TEST(o) \
    TEST(TestBasicArithReduction, o) { test_correctness<Mode::o>(); }

OPR_TEST(SUM)
OPR_TEST(SUM_SQR)
OPR_TEST(PRODUCT)
OPR_TEST(MAX)
OPR_TEST(MIN)
OPR_TEST(MEAN)

TEST(TestBasicArithReduction, CompSeqRecordLevel2) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, CompNode::load("cpux"));
    auto host_tshp =
            std::make_shared<HostTensorND>(host_x->comp_node(), dtype::Int32());

    host_tshp->resize({1});
    host_tshp->ptr<int>()[0] = 1;

    using namespace opr;

    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    graph->options().comp_node_seq_record_level = 2;
    graph->options().graph_opt_level = 0;

    auto x_fp16 = opr::relu(opr::TypeCvt::make(
                 opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
                 dtype::Float16())),
         tshp = opr::Host2DeviceCopy::make(*graph, host_tshp, {"tshp"});

    auto mode = Reduce::Mode::SUM_SQR;
    auto x = opr::TypeCvt::make(x_fp16, dtype::Float32());

    Reduce::Param param_default{mode, MEGDNN_MAX_NDIM,
                                Reduce::Param::DataType::DEFAULT};
    Reduce::Param param_i16_co32{mode, MEGDNN_MAX_NDIM,
                                 Reduce::Param::DataType::FLOAT_O32xC32};

    auto reduce_default = opr::Reduce::make(x, param_default, tshp);
    auto reduce_i16_co32 = opr::Reduce::make(x_fp16, param_i16_co32, tshp);

    HostTensorND host_default, host_i16_co32;

    auto func = graph->compile({
            make_callback_copy(reduce_default, host_default, false),
            make_callback_copy(reduce_i16_co32, host_i16_co32, false),
    });
    ComputingGraph::assert_destroy(graph);

    EXPECT_NO_THROW(func->execute().wait());
    EXPECT_NO_THROW(func->execute().wait());
}

TEST(TestBasicArithReduction, StaticInferValue) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3, 4, 5});
    auto graph = ComputingGraph::make();
    using AI = opr::Subtensor::AxisIndexer;
    // h2d default param enable value infer
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         x_shape = opr::GetVarShape::make(x),
         x_shape_sub = opr::Subtensor::make(x_shape,
            {AI::make_interval(0, x.make_scalar(-2), nullptr ,nullptr)}),
         y = opr::reduce_sum(x, x_shape_sub);
    auto inferred_dev = graph->static_infer_manager().infer_value(y.node());
    HostTensorND expected{host_x->comp_node(), dtype::Float32()};
    // reduce_raw requires the same ndim between src and dest
    expected.resize({1, 1, 4, 5});
    reduce_raw<Mode::SUM, float>(expected, *host_x);
    // reshape as {4, 5}
    expected.reset(expected.storage(), inferred_dev.layout());
    HostTensorND inferred = HostTensorND::make_proxy(inferred_dev);
    MGB_ASSERT_TENSOR_EQ(inferred, expected);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
