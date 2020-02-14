/**
 * \file src/opr/test/nn_int.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"

using namespace mgb;

namespace {
using Checker31 = AutoOprChecker<3, 1>;

std::unique_ptr<Checker31> make_elemwise_multi_type_checker3(
        opr::ElemwiseMultiType::Mode mode, const std::array<DType, 3>& dtypes) {
    using Checker = Checker31;
    auto make_graph =
            [=](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto as_type = [&dtypes, &inputs](size_t i) {
            return opr::TypeCvt::make(inputs[i], dtypes[i]);
        };
        auto ovar = opr::ElemwiseMultiType::make(
                {as_type(0), as_type(1), as_type(2)}, mode);
        return {opr::TypeCvt::make(ovar, dtype::Float32{})};
    };
    auto fwd = [=](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()
                           ->create_operator<megdnn::ElemwiseMultiType>();
        auto opr_typecvt =
                megdnn_naive_handle()->create_operator<megdnn::TypeCvt>();
        opr->param() = {mode};
        megdnn::TensorShapeArray inp_shapes(3);
        megdnn::TensorNDArray inp_tensors(3);
        HostTensorND cvt_val[3];
        for (int i = 0; i < 3; ++i) {
            cvt_val[i]
                    .dtype(dtypes[i])
                    .comp_node(inp[i]->comp_node())
                    .resize(inp[i]->shape());
            opr_typecvt->exec(inp[i]->as_megdnn(), cvt_val[i].as_megdnn());
            inp_shapes[i] = inp[i]->shape();
            inp_tensors[i] = cvt_val[i].as_megdnn();
        }
        TensorShape out_shape;
        megdnn::Elemwise::deduce_shape(inp_shapes, out_shape);
        auto trait = megdnn::ElemwiseMultiType::ModeTrait::from_mode(mode);
        DType dtype;
        trait.check_out(dtype, false);
        HostTensorND tmp_out{inp[0]->comp_node(), out_shape, dtype};
        opr->exec(inp_tensors, tmp_out.as_megdnn());
        dest[0].resize(out_shape);
        opr_typecvt->exec(tmp_out.as_megdnn(), dest[0].as_megdnn());
    };
    return std::make_unique<Checker>(make_graph, fwd);
}
}  // anonymous namespace

TEST(TestOprElemwiseMultiType, Fma3Int16x32x32x32) {
    make_elemwise_multi_type_checker3(
            opr::ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32,
            {dtype::Int16{}, dtype::Int32{}, dtype::Int32{}})
            ->disable_grad_check()
            .run({TensorShape{3, 4, 5}, {1, 4, 1}, {1, 4, 1}})
            .run({TensorShape{1, 4, 5}, {1, 4, 1}, {1, 4, 1}})
            .run({TensorShape{3, 4, 5}, {3, 4, 1}, {3, 4, 1}});
}

TEST(TestOprElemwiseMultiType, Fma3IXxf32xf32xi8) {
    std::array<DType, 3> src_types{dtype::Int8{}, dtype::Int16{},
                                   dtype::Int32{}};
    for (auto src_type : src_types) {
        make_elemwise_multi_type_checker3(
                opr::ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8,
                {src_type, dtype::Float32{}, dtype::Float32{}})
                ->disable_grad_check()
                .run({TensorShape{3, 4}, {3, 4}, {3, 4}})
                .run({TensorShape{3, 4}, {1, 4}, {1, 4}})
                .run({TensorShape{9, 4, 8}, {1, 4, 8}, {1, 4, 8}});
    }
}

TEST(TestOprElemwiseMultiType, QuantizedModeBinary_IS8_OS32) {
    using Checker = AutoOprChecker<2, 1>;
    DType x_dtype = dtype::QuantizedS8(0.15f);
    DType y_dtype = dtype::QuantizedS8(0.20f);
    DType z_dtype = dtype::QuantizedS32(0.15f);
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode : {Mode::QFUSE_ADD_RELU, Mode::QADD, Mode::QMUL}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{z_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto b = opr::Copy::make(inputs[1], cpu);
            auto y = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype),
                     opr::TypeCvt::make(b, y_dtype)},
                    {mode}, config);
            y = opr::TypeCvt::make(y, dtype::Float32());
            return {y};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            auto y = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[1]),
                                        y_dtype);
            SymbolVar z;
            if (mode == Mode::QMUL) {
                z = opr::TypeCvt::make(x, dtype::Float32()) *
                    opr::TypeCvt::make(y, dtype::Float32());
                z = opr::TypeCvt::make(z, z_dtype);
            }
            if (mode == Mode::QADD) {
                z = opr::TypeCvt::make(x, dtype::Float32()) +
                    opr::TypeCvt::make(y, dtype::Float32());
                z = opr::TypeCvt::make(z, z_dtype);
            }
            if (mode == Mode::QFUSE_ADD_RELU) {
                z = opr::TypeCvt::make(x, dtype::Float32()) +
                    opr::TypeCvt::make(y, dtype::Float32());
                z = opr::Elemwise::make({z}, {opr::Elemwise::Mode::RELU});
                z = opr::TypeCvt::make(z, z_dtype);
            }
            z = opr::TypeCvt::make(z, dtype::Float32());
            auto func = cg->compile({make_callback_copy(z, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}, {3, 4}})
                .run({TensorShape{3, 4}, {1, 4}})
                .run({TensorShape{9, 4, 8}, {1, 4, 8}}, options);
    }
}

auto gen_postive = [](HostTensorND& dest) {
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM>
            mask_generator{0.f, FLT_MAX};
    dest = *mask_generator(dest.shape(), dest.comp_node());
};
//! \warning: asin and acos has lower precision,
//! they may produce nan.
auto gen_asin_acos = [](HostTensorND& dest) {
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM>
            mask_generator{-0.5f, 0.5f};
    dest = *mask_generator(dest.shape(), dest.comp_node());
};
//! \warning: erfinv and erfcinv has lower precision,
//! should give them more strict input.
auto gen_erfinv = [](HostTensorND& dest) {
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM>
            mask_generator{-0.5f, 0.5f};
    dest = *mask_generator(dest.shape(), dest.comp_node());
};
auto gen_erfcinv = [](HostTensorND& dest) {
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM>
            mask_generator{0.5f, 1.5f};
    dest = *mask_generator(dest.shape(), dest.comp_node());
};

#define MAKE_UNARY(_MODE)                                            \
    case Mode::Q##_MODE:                                             \
        d = opr::Elemwise::make({xf}, {opr::Elemwise::Mode::_MODE}); \
        break
TEST(TestOprElemwiseMultiType, QuantizedModeUnary_IS8_OS8) {
    using Checker = AutoOprChecker<1, 1>;
    DType x_dtype = dtype::QuantizedS8(1.15f);
    DType d_dtype = dtype::QuantizedS8(2.00f);
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode :
         {Mode::QRELU, Mode::QABS,    Mode::QSIGMOID, Mode::QEXP,
          Mode::QTANH, Mode::QNEGATE, Mode::QACOS,    Mode::QASIN,
          Mode::QCEIL, Mode::QCOS,    Mode::QEXPM1,   Mode::QFLOOR,
          Mode::QLOG,  Mode::QLOG1P,  Mode::QSIN,     Mode::QROUND,
          Mode::QERF,  Mode::QERFINV, Mode::QERFC,    Mode::QERFCINV,
          Mode::QFAST_TANH, Mode::QH_SWISH}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{d_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto d = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype)},
                    {mode}, config);
            d = opr::TypeCvt::make(d, dtype::Float32());
            return {d};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            SymbolVar d;
            auto xf = opr::TypeCvt::make(x, dtype::Float32());
            switch (mode) {
                MAKE_UNARY(RELU);
                MAKE_UNARY(ABS);
                MAKE_UNARY(SIGMOID);
                MAKE_UNARY(EXP);
                MAKE_UNARY(TANH);
                MAKE_UNARY(FAST_TANH);
                MAKE_UNARY(NEGATE);
                MAKE_UNARY(ACOS);
                MAKE_UNARY(ASIN);
                MAKE_UNARY(CEIL);
                MAKE_UNARY(COS);
                MAKE_UNARY(EXPM1);
                MAKE_UNARY(FLOOR);
                MAKE_UNARY(LOG);
                MAKE_UNARY(LOG1P);
                MAKE_UNARY(SIN);
                MAKE_UNARY(ROUND);
                MAKE_UNARY(ERF);
                MAKE_UNARY(ERFINV);
                MAKE_UNARY(ERFC);
                MAKE_UNARY(ERFCINV);
                MAKE_UNARY(H_SWISH);
                default:
                    mgb_throw(InternalError, "Unknown ElemwiseMultiType Mode\n");
                    break;
            }
            d = opr::TypeCvt::make(d, d_dtype);
            d = opr::TypeCvt::make(d, dtype::Float32());
            auto func = cg->compile({make_callback_copy(d, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        switch (mode) {
            case Mode::QACOS:
            case Mode::QASIN:
                checker.set_input_generator(0, gen_asin_acos);
                break;
            case Mode::QLOG:
            case Mode::QLOG1P:
                checker.set_input_generator(0, gen_postive);
                break;
            case Mode::QERFINV:
                checker.set_input_generator(0, gen_erfinv);
                break;
            case Mode::QERFCINV:
                checker.set_input_generator(0, gen_erfcinv);
                break;
            default:
                break;
        }
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}})
                .run({TensorShape{4, 8}})
                .run({TensorShape{9, 4, 8}}, options);
    }
}

TEST(TestOprElemwiseMultiType, QuantizedModeUnary_I8Asymm_O8Asymm) {
    using Checker = AutoOprChecker<1, 1>;
    DType x_dtype = dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128));
    DType d_dtype = dtype::Quantized8Asymm(2.00f, static_cast<uint8_t>(128));
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode :
         {Mode::QRELU, Mode::QABS,    Mode::QSIGMOID, Mode::QEXP,
          Mode::QTANH, Mode::QNEGATE, Mode::QACOS,    Mode::QASIN,
          Mode::QCEIL, Mode::QCOS,    Mode::QEXPM1,   Mode::QFLOOR,
          Mode::QLOG,  Mode::QLOG1P,  Mode::QSIN,     Mode::QROUND,
          Mode::QERF,  Mode::QERFINV, Mode::QERFC,    Mode::QERFCINV,
          Mode::QFAST_TANH}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{d_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto d = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype)},
                    {mode}, config);
            d = opr::TypeCvt::make(d, dtype::Float32());
            return {d};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            SymbolVar d;
            auto xf = opr::TypeCvt::make(x, dtype::Float32());
            switch (mode) {
                MAKE_UNARY(RELU);
                MAKE_UNARY(ABS);
                MAKE_UNARY(SIGMOID);
                MAKE_UNARY(EXP);
                MAKE_UNARY(TANH);
                MAKE_UNARY(FAST_TANH);
                MAKE_UNARY(NEGATE);
                MAKE_UNARY(ACOS);
                MAKE_UNARY(ASIN);
                MAKE_UNARY(CEIL);
                MAKE_UNARY(COS);
                MAKE_UNARY(EXPM1);
                MAKE_UNARY(FLOOR);
                MAKE_UNARY(LOG);
                MAKE_UNARY(LOG1P);
                MAKE_UNARY(SIN);
                MAKE_UNARY(ROUND);
                MAKE_UNARY(ERF);
                MAKE_UNARY(ERFINV);
                MAKE_UNARY(ERFC);
                MAKE_UNARY(ERFCINV);
                default:
                    mgb_throw(InternalError, "Unknown ElemwiseMultiType Mode\n");
                    break;
            }
            d = opr::TypeCvt::make(d, d_dtype);
            d = opr::TypeCvt::make(d, dtype::Float32());
            auto func = cg->compile({make_callback_copy(d, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        switch (mode) {
            case Mode::QACOS:
            case Mode::QASIN:
                checker.set_input_generator(0, gen_asin_acos);
                break;
            case Mode::QLOG:
            case Mode::QLOG1P:
                checker.set_input_generator(0, gen_postive);
                break;
            case Mode::QERFINV:
                checker.set_input_generator(0, gen_erfinv);
                break;
            case Mode::QERFCINV:
                checker.set_input_generator(0, gen_erfcinv);
                break;
            default:
                break;
        }
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}})
                .run({TensorShape{4, 8}})
                .run({TensorShape{9, 4, 8}}, options);
    }
}
#undef MAKE_UANRY

#define MAKE_BINARY(_MODE)                                               \
    case Mode::Q##_MODE:                                                 \
        d = opr::Elemwise::make({xf, yf}, {opr::Elemwise::Mode::_MODE}); \
        break
TEST(TestOprElemwiseMultiType, QuantizedModeBinary_IS8_OS8) {
    using Checker = AutoOprChecker<2, 1>;
    DType x_dtype = dtype::QuantizedS8(1.15f);
    DType y_dtype = dtype::QuantizedS8(2.0f);
    DType d_dtype = dtype::QuantizedS8(1.15f);
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode : {Mode::QFUSE_ADD_RELU, Mode::QADD, Mode::QMUL,
                      Mode::QMIN, Mode::QMAX, Mode::QSUB, Mode::QTRUE_DIV,
                      Mode::QFUSE_ADD_SIGMOID, Mode::QFUSE_ADD_TANH,
                      Mode::QABS_GRAD, Mode::QFLOOR_DIV,
                      Mode::QMOD, Mode::QSIGMOID_GRAD, Mode::QSWITCH_GT0,
                      Mode::QTANH_GRAD, Mode::QLT, Mode::QLEQ, Mode::QEQ,
                      Mode::QPOW, Mode::QLOG_SUM_EXP,
                      Mode::QFAST_TANH_GRAD, Mode::QATAN2}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{d_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto b = opr::Copy::make(inputs[1], cpu);
            auto d = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype),
                     opr::TypeCvt::make(b, y_dtype)},
                    {mode}, config);
            d = opr::TypeCvt::make(d, dtype::Float32());
            return {d};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            auto y = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[1]),
                                        y_dtype);
            SymbolVar d;
            auto xf = opr::TypeCvt::make(x, dtype::Float32());
            auto yf = opr::TypeCvt::make(y, dtype::Float32());
            switch (mode) {
                MAKE_BINARY(FUSE_ADD_RELU);
                MAKE_BINARY(ADD);
                MAKE_BINARY(MUL);
                MAKE_BINARY(MIN);
                MAKE_BINARY(MAX);
                MAKE_BINARY(SUB);
                MAKE_BINARY(TRUE_DIV);
                MAKE_BINARY(FUSE_ADD_SIGMOID);
                MAKE_BINARY(FUSE_ADD_TANH);
                MAKE_BINARY(ABS_GRAD);
                MAKE_BINARY(FLOOR_DIV);
                MAKE_BINARY(MOD);
                MAKE_BINARY(SIGMOID_GRAD);
                MAKE_BINARY(SWITCH_GT0);
                MAKE_BINARY(TANH_GRAD);
                MAKE_BINARY(LT);
                MAKE_BINARY(LEQ);
                MAKE_BINARY(EQ);
                MAKE_BINARY(POW);
                MAKE_BINARY(LOG_SUM_EXP);
                MAKE_BINARY(FAST_TANH_GRAD);
                MAKE_BINARY(ATAN2);
                default:
                    mgb_throw(InternalError, "Unknown ElemwiseMultiType Mode\n");
                    break;
            }
            d = opr::TypeCvt::make(d, d_dtype);
            d = opr::TypeCvt::make(d, dtype::Float32());
            auto func = cg->compile({make_callback_copy(d, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        switch (mode) {
            case Mode::QTRUE_DIV:
            case Mode::QMOD:
            case Mode::QFLOOR_DIV:
                checker.set_input_generator(1, gen_postive);
                break;
            default:
                break;
        }
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}, {3, 4}})
                .run({TensorShape{4, 8}, {1, 1}})
                .run({TensorShape{9, 4, 8}, {9, 4, 8}}, options);
    }
}

TEST(TestOprElemwiseMultiType, QuantizedModeBinary_I8Asymm_O8Asymm) {
    using Checker = AutoOprChecker<2, 1>;
    DType x_dtype = dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128));
    DType y_dtype = dtype::Quantized8Asymm(2.0f, static_cast<uint8_t>(128));
    DType d_dtype = dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128));
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode : {Mode::QFUSE_ADD_RELU,
                      Mode::QADD,
                      Mode::QMUL,
                      Mode::QMIN,
                      Mode::QMAX,
                      Mode::QSUB,
                      Mode::QTRUE_DIV,
                      Mode::QFUSE_ADD_SIGMOID,
                      Mode::QFUSE_ADD_TANH,
                      Mode::QFUSE_ADD_H_SWISH,
                      Mode::QABS_GRAD,
                      Mode::QFLOOR_DIV,
                      Mode::QMOD,
                      Mode::QSIGMOID_GRAD,
                      Mode::QSWITCH_GT0,
                      Mode::QTANH_GRAD,
                      Mode::QLT,
                      Mode::QLEQ,
                      Mode::QEQ,
                      Mode::QPOW,
                      Mode::QLOG_SUM_EXP,
                      Mode::QFAST_TANH_GRAD,
                      Mode::QATAN2}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{d_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto b = opr::Copy::make(inputs[1], cpu);
            auto d = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype),
                     opr::TypeCvt::make(b, y_dtype)},
                    {mode}, config);
            d = opr::TypeCvt::make(d, dtype::Float32());
            return {d};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            auto y = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[1]),
                                        y_dtype);
            SymbolVar d;
            auto xf = opr::TypeCvt::make(x, dtype::Float32());
            auto yf = opr::TypeCvt::make(y, dtype::Float32());
            switch (mode) {
                MAKE_BINARY(FUSE_ADD_RELU);
                MAKE_BINARY(ADD);
                MAKE_BINARY(MUL);
                MAKE_BINARY(MIN);
                MAKE_BINARY(MAX);
                MAKE_BINARY(SUB);
                MAKE_BINARY(TRUE_DIV);
                MAKE_BINARY(FUSE_ADD_SIGMOID);
                MAKE_BINARY(FUSE_ADD_TANH);
                MAKE_BINARY(FUSE_ADD_H_SWISH);
                MAKE_BINARY(ABS_GRAD);
                MAKE_BINARY(FLOOR_DIV);
                MAKE_BINARY(MOD);
                MAKE_BINARY(SIGMOID_GRAD);
                MAKE_BINARY(SWITCH_GT0);
                MAKE_BINARY(TANH_GRAD);
                MAKE_BINARY(LT);
                MAKE_BINARY(LEQ);
                MAKE_BINARY(EQ);
                MAKE_BINARY(POW);
                MAKE_BINARY(LOG_SUM_EXP);
                MAKE_BINARY(FAST_TANH_GRAD);
                MAKE_BINARY(ATAN2);
                default:
                    mgb_throw(InternalError, "Unknown ElemwiseMultiType Mode\n");
                    break;
            }
            d = opr::TypeCvt::make(d, d_dtype);
            d = opr::TypeCvt::make(d, dtype::Float32());
            auto func = cg->compile({make_callback_copy(d, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        switch (mode) {
            case Mode::QTRUE_DIV:
            case Mode::QMOD:
            case Mode::QFLOOR_DIV:
                checker.set_input_generator(1, gen_postive);
                break;
            default:
                break;
        }
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}, {3, 4}})
                .run({TensorShape{4, 8}, {1, 1}})
                .run({TensorShape{9, 4, 8}, {9, 4, 8}}, options);
    }
}
#undef MAKE_BINARY

#define MAKE_TERNARY(_MODE)                                                  \
    case Mode::Q##_MODE:                                                     \
        d = opr::Elemwise::make({xf, yf, zf}, {opr::Elemwise::Mode::_MODE}); \
        break
TEST(TestOprElemwiseMultiType, QuantizedModeTernary_IS8_OS8) {
    using Checker = AutoOprChecker<3, 1>;
    DType x_dtype = dtype::QuantizedS8(1.15f);
    DType y_dtype = dtype::QuantizedS8(2.0f);
    DType z_dtype = dtype::QuantizedS8(1.15f);
    DType d_dtype = dtype::QuantizedS8(1.15f);
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode : {Mode::QFUSE_MUL_ADD3, Mode::QCOND_LEQ_MOV}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{d_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto b = opr::Copy::make(inputs[1], cpu);
            auto c = opr::Copy::make(inputs[2], cpu);
            auto d = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype),
                     opr::TypeCvt::make(b, y_dtype),
                     opr::TypeCvt::make(c, z_dtype)},
                    {mode}, config);
            d = opr::TypeCvt::make(d, dtype::Float32());
            return {d};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            auto y = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[1]),
                                        y_dtype);
            auto z = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[2]),
                                        z_dtype);
            SymbolVar d;
            auto xf = opr::TypeCvt::make(x, dtype::Float32());
            auto yf = opr::TypeCvt::make(y, dtype::Float32());
            auto zf = opr::TypeCvt::make(z, dtype::Float32());
            switch (mode) {
                MAKE_TERNARY(FUSE_MUL_ADD3);
                MAKE_TERNARY(COND_LEQ_MOV);
                default:
                    mgb_throw(InternalError, "Unknown ElemwiseMultiType Mode\n");
                    break;
            }
            d = opr::TypeCvt::make(d, d_dtype);
            d = opr::TypeCvt::make(d, dtype::Float32());
            auto func = cg->compile({make_callback_copy(d, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}, {3, 4}, {3, 4}})
                .run({TensorShape{4, 8}, {4, 8}, {4, 8}})
                .run({TensorShape{9, 4, 8}, {9, 4, 8}, {9, 4, 8}}, options);
    }
}

TEST(TestOprElemwiseMultiType, QuantizedModeTernary_I8Asymm_O8Asymm) {
    using Checker = AutoOprChecker<3, 1>;
    DType x_dtype = dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128));
    DType y_dtype = dtype::Quantized8Asymm(2.0f, static_cast<uint8_t>(128));
    DType z_dtype = dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128));
    DType d_dtype = dtype::Quantized8Asymm(1.15f, static_cast<uint8_t>(128));
    using Mode = opr::ElemwiseMultiType::Param::Mode;
    for (auto mode : {Mode::QFUSE_MUL_ADD3, Mode::QCOND_LEQ_MOV}) {
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            OperatorNodeConfig config{d_dtype};
            auto cpu = CompNode::load("cpux");
            auto a = opr::Copy::make(inputs[0], cpu);
            auto b = opr::Copy::make(inputs[1], cpu);
            auto c = opr::Copy::make(inputs[2], cpu);
            auto d = opr::ElemwiseMultiType::make(
                    {opr::TypeCvt::make(a, x_dtype),
                     opr::TypeCvt::make(b, y_dtype),
                     opr::TypeCvt::make(c, z_dtype)},
                    {mode}, config);
            d = opr::TypeCvt::make(d, dtype::Float32());
            return {d};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto cg = ComputingGraph::make();
            cg->options().graph_opt_level = 0;
            auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[0]),
                                        x_dtype);
            auto y = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[1]),
                                        y_dtype);
            auto z = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*cg, inp[2]),
                                        z_dtype);
            SymbolVar d;
            auto xf = opr::TypeCvt::make(x, dtype::Float32());
            auto yf = opr::TypeCvt::make(y, dtype::Float32());
            auto zf = opr::TypeCvt::make(z, dtype::Float32());
            switch (mode) {
                MAKE_TERNARY(FUSE_MUL_ADD3);
                MAKE_TERNARY(COND_LEQ_MOV);
                default:
                    mgb_throw(InternalError, "Unknown ElemwiseMultiType Mode\n");
                    break;
            }
            d = opr::TypeCvt::make(d, d_dtype);
            d = opr::TypeCvt::make(d, dtype::Float32());
            auto func = cg->compile({make_callback_copy(d, dest[0])});
            func->execute().wait();
        };
        Checker checker{make_graph, fwd};
        Checker::RunOptions options;
        options.outputs_max_err = 0.2;
        checker.disable_grad_check()
                .run({TensorShape{3, 4}, {3, 4}, {3, 4}})
                .run({TensorShape{4, 8}, {4, 8}, {4, 8}})
                .run({TensorShape{9, 4, 8}, {9, 4, 8}, {9, 4, 8}}, options);
    }
}
#undef MAKE_TERNARY

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
