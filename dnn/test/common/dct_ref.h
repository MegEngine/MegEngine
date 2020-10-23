/**
 * \file
 * dnn/test/common/dct_ref.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <math.h>
#include <vector>
#include "megdnn/dtype.h"
#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"
#include "test/common/opr_proxy.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

using Param = DctChannelSelectForward::Param;

struct DctTestcase {
    using TensorValueArray = TensorNDArray;
    TensorValueArray testcase_in;
    TensorValueArray testcase_out;
    std::vector<uint8_t> inp_vec;
    std::vector<int> mask_offset_vec;
    std::vector<int> mask_val_vec;
    std::vector<float> output_vec;
    static std::shared_ptr<DctTestcase> make() {
        return std::make_shared<DctTestcase>();
    }
};

CheckerHelper::TensorsConstriant gen_dct_constriant(
        const size_t n, const size_t ic, const size_t ih, const size_t iw,
        const size_t oc, Param param);

std::shared_ptr<DctTestcase> gen_dct_case(const size_t n, const size_t ic,
                                          const size_t ih, const size_t iw,
                                          const size_t oc, Param param,
                                          DType dst_dtype = dtype::Float32(),
                                          bool correct_result = true);

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
