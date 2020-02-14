/**
 * \file dnn/test/cpu/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

namespace megdnn {
namespace test {

TEST_F(CPU, TYPE_CVT) {
    UniformFloatRNG init(0, 20);
    std::vector<DType> dtypes = {
        dtype::Float32(), dtype::Float16(), 
        dtype::Int32(), dtype::Int16(), dtype::Int8(), dtype::Uint8(),
        dtype::Quantized8Asymm(0.01f, (uint8_t)122),
        dtype::Quantized8Asymm(0.174578f, (uint8_t)129),
        dtype::QuantizedS32(0.233f)};
    for (auto sdtype: dtypes) for (auto ddtype: dtypes) {
        TensorLayout src({10, 10}, sdtype), dst({10, 10}, ddtype);
        Checker<TypeCvt> checker(handle());
        checker.set_rng(0, &init).exec(TensorLayoutArray{src, dst});
    }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CPU, BENCHMARK_TYPE_CVT)
{
    size_t N = 1000000;
    std::vector<DType> types{
        dtype::Float32(), dtype::Int8(), dtype::Int16(), dtype::Int32()};
    float memcpy_time;
    {
        std::cout << "memcpy:" << std::endl;
        Benchmarker<Relayout> benchmarker(handle());
        memcpy_time = benchmarker.execs({{N}, {N}});
    }
    for (auto stype: types) for (auto dtype: types)
    {
        std::cout << stype.name() << " to " << dtype.name() << "." << std::endl;
        Benchmarker<TypeCvt> benchmarker(handle());
        benchmarker.set_dtype(0, stype);
        benchmarker.set_dtype(1, dtype);
        float typecvt_time = benchmarker.execs({{N}, {N}});
        ASSERT_LE(typecvt_time, memcpy_time*3);
    }
}

#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
