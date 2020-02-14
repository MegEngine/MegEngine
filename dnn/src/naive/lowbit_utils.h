/**
 * \file dnn/src/naive/lowbit_utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megdnn/basic_types.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

void uint4_to_uint8(const TensorND& in, const TensorND& out);

void uint8_to_uint4(const TensorND& in, const TensorND& out);

void int4_to_int8(const TensorND& in, const TensorND& out);

void int8_to_int4(const TensorND& in , const TensorND& out);

}  // namespace naive
}  // namespace megdnn
