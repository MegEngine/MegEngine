/**
 * \file dnn/test/common/tensor_remap.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include "test/common/rng.h"
#include "test/common/index.h"

namespace megdnn {
namespace test {
namespace tensor_remap {

class MapRNG final : public IIDRNG {
public:
    MapRNG(TensorShape src) : m_cnt(0), m_src(src) {}
    dt_float32 gen_single_val() override;

private:
    size_t m_cnt;
    TensorShape m_src;
};

class NonoverlappingMapRNG final : public IIDRNG {
public:
    NonoverlappingMapRNG(TensorShape src);
    dt_float32 gen_single_val() override;

private:
    size_t m_cnt;
    TensorShape m_src;
    Index m_idx;
};

}  // namespace tensor_remap
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
