/**
 * \file dnn/test/naive/resize.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/naive/fixture.h"
#include "test/common/checker.h"
#include "test/common/resize.h"
#include "megdnn/oprs/cv.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, RESIZE_NCHW4) {
    Checker<Resize> checker(handle());

    auto args = resize::get_nchw4_args();
    auto convert_true_format = [](const TensorLayout& layout) {
        return layout
                .reshape({layout[0], layout[1] / 4, layout[2], layout[3], 4})
                .dimshuffle({0, 1, 4, 2, 3});
    };

    for (auto&& arg : args) {
        auto extra_impl = [ this, param = arg.param, convert_true_format ](
                const TensorNDArray& tensors) {
            auto resize = handle()->create_operator<Resize>();
            resize->param().imode = param.imode;
            resize->param().format = Resize::Param::Format::NCHW;

            TensorNDArray nchw_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = tensors[i].layout;
                layout = layout.reshape({layout[0], layout[1] * 4, layout[2],
                        layout[3]});
                layout.dtype = dtype::Int8();
                nchw_tensors.emplace_back(malloc(layout.span().dist_byte()),
                        layout);
            }
            TensorNDArray nchw4_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = convert_true_format(nchw_tensors[i].layout);
                nchw4_tensors.emplace_back(tensors[i].raw_ptr,
                                           std::move(layout));
            }

            auto relayout = handle()->create_operator<RelayoutForward>();
            relayout->exec(nchw4_tensors[0], nchw_tensors[0]);

            auto workspace_size = resize->get_workspace_in_bytes(
                    nchw_tensors[0].layout, nchw_tensors[1].layout);
            dt_byte* workspace_ptr =
                    static_cast<dt_byte*>(malloc(workspace_size));
            Workspace workspace{workspace_ptr, workspace_size};

            resize->exec(nchw_tensors[0], nchw_tensors[1], workspace);

            relayout->exec(nchw_tensors[1], nchw4_tensors[1]);

            free(workspace_ptr);
            for (auto &&tensor : nchw_tensors) {
                free(tensor.raw_ptr);
            }
        };
        checker.set_extra_opr_impl(extra_impl);
        checker.set_param(arg.param)
            .set_dtype(0, dtype::QuantizedS8(0.1f))
            .set_dtype(1, dtype::QuantizedS8(0.1f))
            .set_epsilon(1 + 1e-3)
            .execs({arg.src, arg.dst});
    }
}
