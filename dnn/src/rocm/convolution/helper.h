/**
 * \file dnn/src/rocm/convolution/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "./opr_impl.h"
#include "src/rocm/miopen_wrapper.h"
#include "src/rocm/handle.h"
#include "src/common/utils.h"
#include "src/common/algo_chooser.h"

#include <unordered_map>

namespace megdnn {
namespace rocm {
namespace convolution {

struct MIOpenCacheKey {
    int64_t miopen_handle;
    uint32_t batch, IC, IH, IW, OC, OH, OW, FH, FW, SH, SW, PH, PW, DH, DW,
            group, ocpg, icpg, dtype_enum;
    int exhaustive_search;
    std::string to_string_binary() const;
};

//! FIXME: MIOpenCache to avoid calling find() and GetWorkSpaceSize()
//! redundantly
template <typename Args, typename ValueType>
class MIOpenCache {
    using HashMap = std::unordered_map<std::string, ValueType>;
    HashMap m_cache;
    std::mutex m_mtx;

public:
    MIOpenCache() = default;
    ~MIOpenCache() noexcept = default;
    void set(const Args& args, ValueType val);
    std::pair<bool, ValueType> get(const Args& args);
};

using CanonizedFilterMeta = ConvolutionForward::CanonizedFilterMeta;

//! conv size descriptor in the forward view
struct ForwardSizeArgs {
    HandleImpl* handle;
    const TensorLayout* src_layout;
    CanonizedFilterMeta filter_meta;
    const TensorLayout* dst_layout;
};

//! whether miopen is supported for a filter meta
bool is_miopen_supported(const ForwardSizeArgs& args);

//! get workspace bundle for matmul algo
WorkspaceBundle matmul_get_workspace_bundle(const ForwardSizeArgs& args);

/*!
 * \brief flip conv filter
 *
 * Flip conv filter pointed by \p raw_ptr, store result in workspace, and
 * change \p raw_ptr to workspace.
 * */
void flip_filter(const ForwardSizeArgs& args, const Workspace& workspace,
                 void*& raw_ptr);

struct MIOpenForwardDescs {
    TensorDesc src_desc, filter_desc, dst_desc;
    ConvDesc conv_desc;
    void set(const TensorLayout& src, const CanonizedFilterMeta& filter,
             const TensorLayout& dst, const param::Convolution& param) {
        src_desc.set(src, param.format);
        auto&& group = filter.group;
        auto&& ocpg = filter.ocpg;
        auto&& icpg = filter.icpg;
        auto&& fh = filter.spatial[0];
        auto&& fw = filter.spatial[1];
        TensorLayout filter_layout{{group * ocpg, icpg, fh, fw}, filter.dtype};
        filter_desc.set(filter_layout, param.format);
        dst_desc.set(dst, param.format);
        bool is_depthwise = param.sparse == param::Convolution::Sparse::GROUP &&
                            (icpg == 1) && (ocpg == 1);
        conv_desc.set(param, filter.group, is_depthwise);
    }
};

struct MIOpenBwdDataDescs {
    TensorDesc diff_desc, filter_desc, grad_desc;
    ConvDesc conv_desc;
    void set(const CanonizedFilterMeta& filter, const TensorLayout& diff,
             const TensorLayout& grad, const param::Convolution& param) {
        auto&& group = filter.group;
        auto&& ocpg = filter.ocpg;
        auto&& icpg = filter.icpg;
        auto&& fh = filter.spatial[0];
        auto&& fw = filter.spatial[1];
        TensorLayout filter_layout{{group * ocpg, icpg, fh, fw}, filter.dtype};
        filter_desc.set(filter_layout, param.format);
        diff_desc.set(diff, param.format);
        grad_desc.set(grad, param.format);
        bool is_depthwise = param.sparse == param::Convolution::Sparse::GROUP &&
                            (icpg == 1) && (ocpg == 1);
        conv_desc.set(param, filter.group, is_depthwise);
    }
};

struct MIOpenBwdFilterDescs {
    TensorDesc diff_desc, src_desc, grad_desc;
    ConvDesc conv_desc;
    void set(const TensorLayout& src, const TensorLayout& diff,
             const CanonizedFilterMeta& grad, const param::Convolution& param) {
        src_desc.set(src, param.format);
        diff_desc.set(diff, param.format);
        auto&& group = grad.group;
        auto&& ocpg = grad.ocpg;
        auto&& icpg = grad.icpg;
        auto&& fh = grad.spatial[0];
        auto&& fw = grad.spatial[1];
        TensorLayout grad_layout{{group * ocpg, icpg, fh, fw}, grad.dtype};
        grad_desc.set(grad_layout, param.format);
        bool is_depthwise = param.sparse == param::Convolution::Sparse::GROUP &&
                            (icpg == 1) && (ocpg == 1);
        conv_desc.set(param, grad.group, is_depthwise);
    }
};

//! TODO:miopen does not support non xcorr convolution for now, expecting
//! support in future.
} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: syntax=cpp.doxygen
