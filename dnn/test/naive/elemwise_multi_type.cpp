/**
 * \file dnn/test/naive/elemwise_multi_type.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn_int.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

namespace {
#define MODE(_MODE)                         \
    case ElemwiseMultiType::Mode::Q##_MODE: \
        return Elemwise::Mode::_MODE
Elemwise::Mode get_elem_mode(ElemwiseMultiType::Mode mode) {
    switch (mode) {
        MODE(ADD);
        MODE(FUSE_ADD_RELU);
        MODE(MUL);
        MODE(MIN);
        MODE(MAX);
        MODE(SUB);
        MODE(TRUE_DIV);
        MODE(FUSE_ADD_SIGMOID);
        MODE(FUSE_ADD_TANH);
        MODE(RELU);
        MODE(ABS);
        MODE(SIGMOID);
        MODE(EXP);
        MODE(TANH);
        MODE(FAST_TANH);
        MODE(FUSE_MUL_ADD3);
        MODE(NEGATE);
        MODE(ACOS);
        MODE(ASIN);
        MODE(CEIL);
        MODE(COS);
        MODE(EXPM1);
        MODE(FLOOR);
        MODE(LOG);
        MODE(LOG1P);
        MODE(SIN);
        MODE(ROUND);
        MODE(ERF);
        MODE(ERFINV);
        MODE(ERFC);
        MODE(ERFCINV);
        MODE(H_SWISH);
        MODE(ABS_GRAD);
        MODE(FLOOR_DIV);
        MODE(MOD);
        MODE(SIGMOID_GRAD);
        MODE(SWITCH_GT0);
        MODE(TANH_GRAD);
        MODE(LT);
        MODE(LEQ);
        MODE(EQ);
        MODE(POW);
        MODE(LOG_SUM_EXP);
        MODE(FAST_TANH_GRAD);
        MODE(ATAN2);
        MODE(COND_LEQ_MOV);

        MODE(H_SWISH_GRAD);
        MODE(FUSE_ADD_H_SWISH);
        default:
            megdnn_throw("unsupported elemwise mode");
    }
}
#undef MODE
}  // namespace

TEST_F(NAIVE, ELEMWISE_QUANTIZED_MODE_UNARY) {
    using Param = ElemwiseMultiType::Param;

    Checker<ElemwiseMultiType> checker(handle());
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));

    for (auto mode :
         {Param::Mode::QRELU,
          Param::Mode::QABS,
          Param::Mode::QACOS,
          Param::Mode::QASIN,
          Param::Mode::QCEIL,
          Param::Mode::QCOS,
          Param::Mode::QEXP,
          Param::Mode::QEXPM1,
          Param::Mode::QFLOOR,
          Param::Mode::QLOG,
          Param::Mode::QLOG1P,
          Param::Mode::QNEGATE,
          Param::Mode::QSIGMOID,
          Param::Mode::QSIN,
          Param::Mode::QTANH,
          Param::Mode::QFAST_TANH,
          Param::Mode::QROUND,
          Param::Mode::QERF,
          Param::Mode::QERFINV,
          Param::Mode::QERFC, 
          Param::Mode::QERFCINV,
          Param::Mode::QH_SWISH}) {
        Param param{mode};
        checker.set_param(param);

        auto extra_impl = [&](const TensorNDArray& tensors) {
            TensorNDArray float_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = tensors[i].layout;
                layout.dtype = dtype::Float32();
                float_tensors.emplace_back(malloc(layout.span().dist_byte()),
                                           std::move(layout));
            }
            auto typecvt = handle()->create_operator<TypeCvt>();
            typecvt->exec(tensors[0], float_tensors[0]);

            auto opr = handle()->create_operator<Elemwise>();
            opr->param().mode = get_elem_mode(mode);
            opr->exec({float_tensors[0]}, float_tensors[1]);

            typecvt->exec(float_tensors[1], tensors[1]);

            for (auto&& tensor : float_tensors) {
                free(tensor.raw_ptr);
            }
        };

        checker.set_extra_opr_impl(extra_impl);

        checker.set_dtype(1, dtype::QuantizedS8(0.35f));
        checker.execs({{3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 1}, {}});

        checker.set_dtype(1, dtype::QuantizedS32(0.35f));
        checker.execs({{3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 1}, {}});
    }
}

TEST_F(NAIVE, ELEMWISE_QUANTIZED_MODE_BINARY) {
    using Param = ElemwiseMultiType::Param;

    Checker<ElemwiseMultiType> checker(handle());
    checker.set_dtype(0, dtype::QuantizedS8(0.1f))
            .set_dtype(1, dtype::QuantizedS8(0.2f));

    for (auto mode : {
          Param::Mode::QABS_GRAD,
          Param::Mode::QADD,
          Param::Mode::QFLOOR_DIV,
          Param::Mode::QMAX,
          Param::Mode::QMIN,
          Param::Mode::QMOD,
          Param::Mode::QMUL,
          Param::Mode::QPOW,
          Param::Mode::QSIGMOID_GRAD,
          Param::Mode::QSUB,
          Param::Mode::QSWITCH_GT0,
          Param::Mode::QTANH_GRAD,
          Param::Mode::QTRUE_DIV,
          Param::Mode::QLOG_SUM_EXP,

          Param::Mode::QLT,
          Param::Mode::QLEQ,
          Param::Mode::QEQ,

          Param::Mode::QFUSE_ADD_RELU,
          Param::Mode::QFUSE_ADD_SIGMOID,
          Param::Mode::QFUSE_ADD_TANH,
          Param::Mode::QFAST_TANH_GRAD,
          Param::Mode::QATAN2,
          Param::Mode::QH_SWISH_GRAD,
          Param::Mode::QFUSE_ADD_H_SWISH}) {

        Param param{mode};
        checker.set_param(param);

        auto extra_impl = [&](const TensorNDArray& tensors) {
            TensorNDArray float_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = tensors[i].layout;
                layout.dtype = dtype::Float32();
                float_tensors.emplace_back(malloc(layout.span().dist_byte()),
                                           std::move(layout));
            }
            auto typecvt = handle()->create_operator<TypeCvt>();
            for (size_t i = 0; i < 2; ++i) {
                typecvt->exec(tensors[i], float_tensors[i]);
            }

            auto opr = handle()->create_operator<Elemwise>();
            opr->param().mode = get_elem_mode(mode);
            opr->exec({float_tensors[0], float_tensors[1]}, float_tensors[2]);

            typecvt->exec(float_tensors[2], tensors[2]);

            for (auto&& tensor : float_tensors) {
                free(tensor.raw_ptr);
            }
        };

        checker.set_extra_opr_impl(extra_impl);

        checker.set_dtype(2, dtype::QuantizedS8(0.35f));
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {10, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 6}, {20, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 1}, {2, 1, 1, 2}, {}});

        checker.set_dtype(2, dtype::QuantizedS32(0.35f));
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {10, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 6}, {20, 4, 5, 6}, {}});
        checker.execs({{1, 4, 5, 1}, {2, 1, 1, 2}, {}});
    }
}

TEST_F(NAIVE, ELEMWISE_QUANTIZED_MODE_TERNARY) {
    using Param = ElemwiseMultiType::Param;

    Checker<ElemwiseMultiType> checker(handle());
    checker.set_dtype(0, dtype::QuantizedS8(0.1f))
            .set_dtype(1, dtype::QuantizedS8(0.2f))
            .set_dtype(2, dtype::QuantizedS8(0.3f));

    for (auto mode : {Param::Mode::QFUSE_MUL_ADD3,
                      Param::Mode::QCOND_LEQ_MOV}) {
        Param param{mode};
        checker.set_param(param);

        auto extra_impl = [&](const TensorNDArray& tensors) {
            TensorNDArray float_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = tensors[i].layout;
                layout.dtype = dtype::Float32();
                float_tensors.emplace_back(malloc(layout.span().dist_byte()),
                                           std::move(layout));
            }
            auto typecvt = handle()->create_operator<TypeCvt>();
            for (size_t i = 0; i < 3; ++i) {
                typecvt->exec(tensors[i], float_tensors[i]);
            }

            auto opr = handle()->create_operator<Elemwise>();
            opr->param().mode = get_elem_mode(mode);
            opr->exec({float_tensors[0], float_tensors[1], float_tensors[2]},
                      float_tensors[3]);

            typecvt->exec(float_tensors[3], tensors[3]);

            for (auto&& tensor : float_tensors) {
                free(tensor.raw_ptr);
            }
        };

        checker.set_extra_opr_impl(extra_impl);

        checker.set_dtype(3, dtype::QuantizedS8(0.35f));
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {10, 4, 5, 6}, {10, 4, 5, 6}, {}});

        checker.set_dtype(3, dtype::QuantizedS32(0.35f));
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {10, 4, 5, 6}, {10, 4, 5, 6}, {}});
    }
}

// vim: syntax=cpp.doxygen
