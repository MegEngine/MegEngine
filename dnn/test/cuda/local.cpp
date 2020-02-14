/**
 * \file dnn/test/cuda/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/local.h"
#include "test/cuda/local/local.h"
#include <cuda_runtime_api.h>
#include "megcore_cuda.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, LOCAL_FORWARD)
{
    auto args = local::get_args_for_cuda();
    for (auto &&arg: args) {
        Checker<LocalForward> checker(handle_cuda());
        cudaStream_t stream;
        ASSERT_EQ(megcoreSuccess,
                megcoreGetCUDAStream(handle_cuda()->megcore_computing_handle(),
                    &stream));
        pollute_shared_mem(stream);
        checker.set_param(arg.param).exec(TensorShapeArray{
                arg.sshape(), arg.fshape(), arg.dshape()});
    }
}



TEST_F(CUDA, LOCAL_BACKWARD_DATA)
{
    using namespace local;
    //std::vector<TestArg> args;
    //args.emplace_back(param::Convolution{
    //        param::Convolution::Mode::CROSS_CORRELATION,
    //        1, 1, 1, 1},
    //        64, 16, 8, 7, 16, 8, 7, 3, 3);
    auto args = local::get_args_bwd_data_for_cuda();
    for (auto &&arg: args) {
        Checker<LocalBackwardData> checker(handle_cuda());
        cudaStream_t stream;
        ASSERT_EQ(megcoreSuccess,
                megcoreGetCUDAStream(handle_cuda()->megcore_computing_handle(),
                    &stream));
        pollute_shared_mem(stream);
        checker.set_param(arg.param).exec(TensorShapeArray{
                arg.fshape(), arg.dshape(), arg.sshape()});
        }
}

TEST_F(CUDA, LOCAL_BACKWARD_FILTER)
{
    using namespace local;
    //std::vector<TestArg> args;
    //args.emplace_back(param::Convolution{
    //        param::Convolution::Mode::CROSS_CORRELATION,
    //        1, 1, 1, 1},
    //        64, 16, 8, 7, 16, 8, 7, 3, 3);
    auto args = local::get_args_bwd_filter_for_cuda();
    for (auto &&arg: args) {
        Checker<LocalBackwardFilter> checker(handle_cuda());
        cudaStream_t stream;
        ASSERT_EQ(megcoreSuccess,
                megcoreGetCUDAStream(handle_cuda()->megcore_computing_handle(),
                    &stream));
        pollute_shared_mem(stream);
        checker.set_param(arg.param).exec(TensorShapeArray{
                arg.sshape(), arg.dshape(), arg.fshape()});
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
