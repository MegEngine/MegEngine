/**
 * \file dnn/test/common/accuracy_shake_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/accuracy_shake_checker.h"

using namespace megdnn;
using namespace test;

namespace {

template <typename ctype>
::testing::AssertionResult assert_tensor_binary_eq(
        const char* expr0, const char* expr1, const char* /*expr2*/,
        const TensorND& v0, const TensorND& v1, const std::string& algo_name) {
    ctype* it0_orig = v0.ptr<ctype>();
    ctype* it1 = v1.ptr<ctype>();
    ctype* it0 = it0_orig;
    auto nr_elem = v1.layout.total_nr_elems();
    auto nr_elem_single_batch = v0.layout.total_nr_elems();
    for (size_t i = 0; i < nr_elem; ++i) {
        if (i % nr_elem_single_batch == 0) {
            it0 = it0_orig;
        }
        ctype iv0 = *it0, iv1 = *it1;

        if (!good_float(iv0) || !good_float(iv1) ||
            memcmp(it0, it1, sizeof(ctype))) {
            Index index(v1.layout, i);
            return ::testing::AssertionFailure()
                   << "Unequal value\n"
                   << "Value of: " << expr1 << "\n"
                   << "  Actual: " << (iv1 + 0) << "\n"
                   << "Expected: " << expr0 << "\n"
                   << "Which is: " << (iv0 + 0) << "\n"
                   << "At index: " << index.to_string() << "/"
                   << v1.layout.TensorShape::to_string() << "\n"
                   << "   DType: " << v1.layout.dtype.name() << "\n"
                   << "algo: " << algo_name;
        }

        ++it0;
        ++it1;
    }

    return ::testing::AssertionSuccess();
}
}  // namespace

::testing::AssertionResult test::__assert_tensor_binary_eq(
        const char* expr0, const char* expr1, const char* expr2,
        const TensorND& v0, const TensorND& v1,
        const Algorithm::Info::Desc& algo) {
    bool shape_match = v0.layout[0] == 1;
    for (size_t i = 1; i < v0.layout.ndim; ++i) {
        shape_match &= v0.layout[i] == v1.layout[i];
    }
    if (!shape_match) {
        return ::testing::AssertionFailure()
               << "Shape mismatch\n"
               << "Value of: " << expr1 << "\n"
               << "  Actual: " << v1.layout.TensorShape::to_string() << "\n"
               << "Expected: " << expr0 << "\n"
               << "Which is: " << v0.layout.TensorShape::to_string() << "\n"
               << "algo: " << algo.name << "\n";
    }

    if (!v0.layout.is_physical_contiguous() ||
        !v1.layout.is_physical_contiguous()) {
        return ::testing::AssertionFailure()
               << "layout should be physical contiguous\n"
               << "Value of: " << expr1 << "\n"
               << "  Actual: " << v1.layout.is_physical_contiguous() << "\n"
               << "Expected: " << expr0 << "\n"
               << "Which is: " << v0.layout.is_physical_contiguous() << "\n"
               << "algo: " << algo.name << "\n";
    }
    auto dtype = v0.layout.dtype;
    if (dtype != v1.layout.dtype) {
        return ::testing::AssertionFailure()
               << "Data type should match\n"
               << "Value of: " << expr1 << "\n"
               << "  Actual: " << v1.layout.dtype.name() << "\n"
               << "Expected: " << expr0 << "\n"
               << "Which is: " << v0.layout.dtype.name() << "\n"
               << "algo: " << algo.name << "\n";
    }

    switch (dtype.enumv()) {
#define cb(_dt)                                                 \
    case DTypeTrait<_dt>::enumv:                                \
        return assert_tensor_binary_eq<DTypeTrait<_dt>::ctype>( \
                expr0, expr1, expr2, v0, v1, algo.name);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
                default : megdnn_trap();
    }
}

// vim: syntax=cpp.doxygen
