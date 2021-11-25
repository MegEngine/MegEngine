/**
 * \file src/gopt/test/network.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/test/helper.h"

#include "megbrain/gopt/framework.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

namespace mgb {
class Network {
private:
    HostTensorGenerator<> gen;
    CompNode cn;

public:
    std::shared_ptr<ComputingGraph> graph = ComputingGraph::make();
    Network(CompNode cn_) : cn{cn_} {}
    ~Network() noexcept = default;
    using KernSize = SmallVector<size_t, 2>;
    using Stride = SmallVector<size_t, 2>;
    using Padding = SmallVector<size_t, 2>;
    SymbolVar add_var(const char* name, const TensorShape& shp = {1}) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp), cn).rename(name);
    }
    SymbolVar add_cvar(const char* name, const TensorShape& shp = {1}) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp), cn).rename(name);
    }

    SymbolVar add_conv(
            SymbolVar f, size_t output_channels, KernSize kern_size,
            DType out_dtype = dtype::Float32(), bool has_relu = true,
            Stride stride = {1, 1}, Padding padding = {0, 0});
    SymbolVar add_deconv(
            SymbolVar f, size_t ratio, size_t output_channels, DType out_dtype);
    SymbolVar add_elemwise(
            const SymbolVarArray inps, DType out_dtype = dtype::Float32(),
            opr::Elemwise::Param::Mode mode = opr::Elemwise::Param::Mode::ADD);
    using Window = SmallVector<size_t, 2>;
    SymbolVar add_pooling(
            SymbolVar f, Window window, Stride stride = {1, 1},
            Padding padding = {0, 0},
            opr::Pooling::Param::Mode mode = opr::Pooling::Param::Mode::MAX);
    SymbolVar add_type_cvt(SymbolVar f, DType out_dtype = dtype::Float32());
    SymbolVar add_concat(SymbolVar f, SymbolVar g, int axis = 0);
};

SymbolVar create_block(
        Network& network, SymbolVar f, size_t stride, size_t num_outputs1,
        bool has_proj = false, DType out_dtype = dtype::Float32());

SymbolVar make_resnet18(
        Network& network, size_t batch = 16, DType out_dtype = dtype::Float32());

SymbolVarArray make_det(
        Network& network, size_t batch = 16, DType out_dtype = dtype::Float32());

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
