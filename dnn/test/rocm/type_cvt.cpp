/**
 * \file dnn/test/rocm/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(ROCM, TYPE_CVT) {
    UniformFloatRNG init(0, 20);
    std::vector<DType> dtypes = {dtype::Float32(), MEGDNN_INC_FLOAT16(dtype::Float16() MEGDNN_COMMA)
                                 dtype::Int32(),   dtype::Int16(),
                                 dtype::Int8(),    dtype::Uint8()};
    for (auto sdtype : dtypes)
        for (auto ddtype : dtypes) {
            TensorLayout src({10, 10}, sdtype), dst({10, 10}, ddtype);
            Checker<TypeCvt> checker(handle_rocm());
            checker.set_rng(0, &init).exec(TensorLayoutArray{src, dst});
        }
}

TEST_F(ROCM, QUANTIZED_TYPECVT) {
    UniformIntRNG int_rng{-66, 66};
    Checker<TypeCvt> checker(handle_rocm());
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng);

    auto run = [&](const DType& src_dtype, const DType& dst_dtype) {
        checker.set_dtype(0, src_dtype)
                .set_dtype(1, dst_dtype)
                .execs({{20, 3, 224, 224}, {20, 3, 224, 224}});
        checker.set_dtype(0, dst_dtype)
                .set_dtype(1, src_dtype)
                .execs({{20, 3, 224, 224}, {20, 3, 224, 224}});
    };

    run(dtype::Float32(), dtype::QuantizedS8(3.0f));
#if !MEGDNN_DISABLE_FLOAT16
    run(dtype::Float16(), dtype::QuantizedS8(3.0f));
#endif
    run(dtype::Int32(), dtype::QuantizedS32(5.0f));
    run(dtype::Int8(), dtype::QuantizedS32(10.0f));

    run(dtype::Float32(), dtype::QuantizedS8(2e-3f));
#if !MEGDNN_DISABLE_FLOAT16
    run(dtype::Float16(), dtype::QuantizedS8(1e-3f));
#endif
    run(dtype::Int32(), dtype::QuantizedS32(1e-3f));
    run(dtype::Int8(), dtype::QuantizedS32(7e-4f));

    run(dtype::QuantizedS8(3.0f), dtype::QuantizedS8(10.0f));
    run(dtype::QuantizedS32(3.0f), dtype::QuantizedS8(10.0f));
    run(dtype::QuantizedS8(3.0f), dtype::QuantizedS32(10.0f));
    run(dtype::QuantizedS32(3.0f), dtype::QuantizedS32(10.0f));

    run(dtype::QuantizedS8(1e-3f), dtype::QuantizedS8(5e-3f));
    run(dtype::QuantizedS32(2e-3f), dtype::QuantizedS8(9e-4f));
    run(dtype::QuantizedS8(9e-4f), dtype::QuantizedS32(7e-4f));
    run(dtype::QuantizedS32(5e-3f), dtype::QuantizedS32(1e-3f));

    run(dtype::Quantized8Asymm(5.0f, (uint8_t)128), dtype::Float32());
#if !MEGDNN_DISABLE_FLOAT16
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)124), dtype::Float16());
#endif
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)30), dtype::Int8());
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)20), dtype::Int32());
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)10), dtype::QuantizedS8(10.5f));
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)18), dtype::QuantizedS32(10.5f));

    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)128), dtype::Float32());
#if !MEGDNN_DISABLE_FLOAT16
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)124), dtype::Float16());
#endif
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)30), dtype::Int8());
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)20), dtype::Int32());
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)10), dtype::QuantizedS8(2e-3f));
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)18), dtype::QuantizedS32(7e-4f));
}

// vim: syntax=cpp.doxygen
