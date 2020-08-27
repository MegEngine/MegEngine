/**
 * \file dnn/test/rocm/benchmarker.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "test/rocm/benchmarker.h"

#include <gtest/gtest.h>
#include "test/common/timer.h"

namespace megdnn {
namespace test {

template <typename Opr>
ROCMBenchmarker<Opr>::ROCMBenchmarker(Handle* handle_rocm, Handle* handle_naive)
        : m_handle_naive{handle_naive},
          m_handle_rocm{handle_rocm},
          m_default_rng{new NormalRNG()},
          m_param{Param()},
          m_device_timer{
                  megdnn::rocm::concrete_handle(m_handle_rocm)->stream()} {}

template <typename Opr>
float ROCMBenchmarker<Opr>::exec(const TensorShapeArray& shapes) {
    return exec(make_layouts(shapes));
}

template <typename Opr>
float ROCMBenchmarker<Opr>::exec(TensorLayoutArray layouts) {
    auto opr = m_handle_rocm->create_operator<Opr>();
    opr->param() = m_param;
    auto user_layouts = layouts;
    m_proxy.deduce_layout(opr.get(), layouts);
    for (size_t i = 0; i < layouts.size(); ++i)
        if (user_layouts[i].ndim > 0) {
            auto run = [&]() {
                ASSERT_TRUE(layouts[i].eq_shape(user_layouts[i]))
                        << "User provided shape is "
                        << user_layouts[i].TensorShape::to_string()
                        << "\nExpected shape is "
                        << layouts[i].TensorShape::to_string();
            };
            run();
        }
    auto allocate = [&layouts](Handle* handle) {
        TensorNDArray tensors(layouts.size());
        auto trans_func = [handle](const TensorLayout& layout) {
            auto span = layout.span();
            TensorND res;
            res.raw_ptr = static_cast<uint8_t*>(
                                  megdnn_malloc(handle, span.dist_byte())) +
                          span.low_byte;
            res.layout = layout;
            return res;
        };
        std::transform(layouts.begin(), layouts.end(), tensors.begin(),
                       trans_func);
        return tensors;
    };
    auto tensors_cur = allocate(m_handle_rocm);
    auto tensors_cur_host = allocate(m_handle_naive);
    // init
    for (size_t i = 0; i < tensors_cur_host.size(); ++i) {
        TensorND& tensor = tensors_cur_host[i];
        auto rng = m_rng[i];
        if (!rng)
            rng = m_default_rng.get();
        auto size = tensor.layout.span().high_byte;
        rng->gen(tensor);
        megdnn_memcpy_H2D(m_handle_rocm, tensors_cur[i].raw_ptr, tensor.raw_ptr,
                          size);
    }
    m_device_timer.reset();
    m_device_timer.start();
    m_proxy.exec(opr.get(), tensors_cur);
    m_device_timer.stop();
    auto time_in_ms = m_device_timer.get_time_in_ms();
    if (m_display) {
        std::cout << "Total time is " << time_in_ms << "ms " << std::endl;
    }
    auto free = [](Handle* handle, TensorNDArray& tensors) {
        std::for_each(tensors.begin(), tensors.end(),
                      [handle](const TensorND& tensor) {
                          megdnn_free(handle, tensor.raw_ptr);
                      });
    };
    free(m_handle_rocm, tensors_cur);
    free(m_handle_naive, tensors_cur_host);
    return time_in_ms;
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
