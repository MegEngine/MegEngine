/**
 * \file src/tensorrt/test/make_trt_net.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/debug.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"

#include <random>

using namespace mgb;
using namespace opr;
using namespace nvinfer1;

template <typename T>
using TensorRTUniquePtr = intl::TensorRTUniquePtr<T>;

namespace mgb{
namespace opr{
namespace intl{

struct SimpleTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x, y;

    HostTensorND host_z1;

    SimpleTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*>
    create_trt_network(bool has_batch_dim);
};

struct SimpleQuantizedTensorRTNetwork {
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> weight_gen{
            1*1.1f, 127*1.1f};
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> range_gen{
            1*1.2f, 127*1.2f};
    std::shared_ptr<HostTensorND> host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x, y;
    SymbolVar quantized_x, quantized_y;

    SimpleQuantizedTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*>
    create_trt_network(bool has_batch_dim);
};

struct ConcatConvTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_x1, host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x0, x1, y;

    HostTensorND host_z1;

    ConcatConvTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*>
    create_trt_network(bool has_batch_dim);
};

}  // namespace intl
}  // namespace opr
}  // namespace mgb


#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
