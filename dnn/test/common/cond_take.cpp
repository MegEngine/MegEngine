/**
 * \file dnn/test/common/cond_take.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cond_take.h"
#include "./utils.h"
#include "./tensor.h"
#include "./rng.h"

using namespace megdnn;
using namespace test;

using Param = CondTake::Param;

std::vector<CondTakeTestcase> CondTakeTestcase::make() {
    std::vector<CondTakeTestcase> ret;
    for (uint32_t mode = 0; mode < Param::MODE_NR_MEMBER; ++ mode) {
        ret.push_back({
                Param{static_cast<Param::Mode>(mode), 0.1f, 0.1f},
                TensorLayout{{1}, dtype::Int8()},
                TensorLayout{{1}, dtype::Float32()},
                });
        ret.push_back({
                Param{static_cast<Param::Mode>(mode), 0.1f, 0.1f},
                TensorLayout{{2, 3}, dtype::Int8()},
                TensorLayout{{2, 3}, dtype::Float32()},
                });
        ret.push_back({
                Param{static_cast<Param::Mode>(mode), 100},
                TensorLayout{{1024}, dtype::Float32()},
                TensorLayout{{1024}, dtype::Int32()},
                });
    }

    NormalRNG data_rng;
    UniformIntRNG rng_byte(0, 255);
    auto fill_data = [&](TensorND data) {
        auto sz = data.layout.span().dist_byte(),
             szf = sz / sizeof(dt_float32);
        auto pf = static_cast<dt_float32*>(data.raw_ptr);
        data_rng.fill_fast_float32(pf, szf);

        auto prem = reinterpret_cast<uint8_t*>(pf + szf);
        size_t szrem = sz % sizeof(dt_float32);
        for (size_t i = 0; i < szrem; ++ i) {
            prem[i] = rng_byte.gen_single_val();
        }
    };

    for (auto &&i: ret) {
        auto size0 = i.m_data.layout.span().dist_byte(),
             size1 = i.m_mask.layout.span().dist_byte();
        i.m_mem.reset(new uint8_t[size0 + size1]);
        i.m_data.raw_ptr = i.m_mem.get();
        i.m_mask.raw_ptr = i.m_mem.get() + size0;
        fill_data(i.m_data);

        auto mean = i.m_param.val;
        if (i.m_mask.layout.dtype == dtype::Int32()) {
            UniformIntRNG rng(mean - 10, mean + 10);
            rng.gen(i.m_mask);
        } else {
            megdnn_assert(i.m_mask.layout.dtype == dtype::Float32());
            NormalRNG rng(mean);
            rng.gen(i.m_mask);
        }
    }

    return ret;
}

CondTakeTestcase::Result CondTakeTestcase::run(CondTake* opr) {
    auto handle = opr->handle();
    auto data = make_tensor_h2d(handle, m_data),
         mask = make_tensor_h2d(handle, m_mask);

    opr->param() = m_param;

    DynOutMallocPolicyImpl malloc_policy(handle);
    auto workspace_size = opr->get_workspace_in_bytes(data->layout);
    auto workspace_ptr = malloc_policy.alloc_workspace(workspace_size, nullptr);
    auto result =
            opr->exec(*data, *mask, {(dt_byte*)workspace_ptr, workspace_size},
                      &malloc_policy);
    malloc_policy.free_workspace(workspace_ptr, nullptr);
    return {make_tensor_d2h(handle, result[0]),
            make_tensor_d2h(handle, result[1])};
}

// vim: syntax=cpp.doxygen
