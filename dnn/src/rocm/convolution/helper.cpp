/**
 * \file dnn/src/rocm/convolution/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./helper.h"
#include "./forward/algo.h"
#include "./backward_data/algo.h"
#include "./backward_filter/algo.h"

using namespace megdnn;
using namespace rocm;
using namespace convolution;

bool convolution::is_miopen_supported(const ForwardSizeArgs& args) {
    //! TODO: We only support NCHW format now. It seems MIOpen do not support
    //! NHWC or NCHW4 now
    if (args.filter_meta.format != param::Convolution::Format::NCHW) {
        return false;
    }
    auto& fm = args.filter_meta;
    //! TODO: It seems MIOpen do not support non xcorr convolution
    return !fm.should_flip;
}

std::string MIOpenCacheKey::to_string_binary() const {
    std::string ret(sizeof(MIOpenCacheKey), '\0');
    auto ptr = reinterpret_cast<MIOpenCacheKey*>(&ret[0]);
    *ptr = *this;
    return ret;
}

template <typename Args, typename ValueType>
void MIOpenCache<Args, ValueType>::set(const Args& args, ValueType val) {
    std::string key = args.to_miopen_algo_cache_key().to_string_binary();
    std::lock_guard<std::mutex> guard{m_mtx};
    m_cache[key] = val;
}

template <typename Args, typename ValueType>
std::pair<bool, ValueType> MIOpenCache<Args, ValueType>::get(const Args& args) {
    std::string key = args.to_miopen_algo_cache_key().to_string_binary();
    std::lock_guard<std::mutex> guard{m_mtx};
    auto search = m_cache.find(key);
    bool find = search != m_cache.end();
    ValueType val = ValueType();
    if (find) {
        val = search->second;
    }
    return std::make_pair(find, val);
}

#define INST(_opr, _miopen_algo)                           \
    template class megdnn::rocm::convolution::MIOpenCache< \
            _opr::AlgoBase::SizeArgs, _miopen_algo>;       \
    template class megdnn::rocm::convolution::MIOpenCache< \
            _opr::AlgoBase::SizeArgs, size_t>;

INST(ConvolutionForwardImpl, miopenConvFwdAlgorithm_t);
INST(ConvolutionBackwardDataImpl, miopenConvBwdDataAlgorithm_t);
INST(ConvolutionBackwardFilterImpl, miopenConvBwdWeightsAlgorithm_t);

WorkspaceBundle convolution::matmul_get_workspace_bundle(
        const ForwardSizeArgs& args) {
    auto dtype = args.src_layout->dtype;
    auto&& fm = args.filter_meta;
    megdnn_assert(fm.group == 1);
    auto N = args.src_layout->shape[0];
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto OH = args.dst_layout->shape[2], OW = args.dst_layout->shape[3];
    SmallVector<size_t> sizes{dtype.size() * args.dst_layout->total_nr_elems(),
                              dtype.size() * IC * FH * FW * OH * OW * N};
    if (args.filter_meta.should_flip) {
        sizes.push_back(dtype.size() * OC * IC * FH * FW);
    }
    return {nullptr, std::move(sizes)};
}

void convolution::flip_filter(const ForwardSizeArgs& args,
                              const Workspace& workspace, void*& raw_ptr) {
    auto&& fm = args.filter_meta;
    megdnn_assert(fm.group == 1 && fm.spatial_ndim == 2);
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto dtype = fm.dtype;
    megdnn_assert(workspace.size >= dtype.size() * OC * IC * FH * FW);

    TensorND src{raw_ptr, {{OC, IC, FH, FW}, dtype}},
            dst{workspace.raw_ptr + (FH * FW - 1) * dtype.size(), src.layout};
    dst.layout.stride[2] = -dst.layout.stride[2];
    dst.layout.stride[3] = -dst.layout.stride[3];
    args.handle->relayout_opr()->exec(src, dst);
    raw_ptr = workspace.raw_ptr;
}

// vim: syntax=cpp.doxygen
