/**
 * \file dnn/src/common/indexing_one_hot.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"

#include "src/common/utils.h"

using namespace megdnn;

void IndexingOneHotBase::deduce_layout_fwd(
        const TensorLayout &src, const TensorLayout &index,
        TensorLayout &dst) {
    megdnn_assert(
            m_param.axis < static_cast<int32_t>(src.ndim) && src.ndim >= 2,
            "IndexingOneHot on axis %u, but input has only %zu dims",
            m_param.axis, src.ndim);
    MEGDNN_MARK_USED_VAR(index);
    dst = src;
    dst.shape[m_param.axis] = 1;
    dst.init_contiguous_stride();
}

void IndexingOneHotBase::check_layout_fwd(
        const TensorLayout &src, const TensorLayout &index,
        const TensorLayout &dst) {
    auto errmsg = [&]() -> std::string {
        return megdnn_mangle(ssprintf("bad layout for IndexingOneHot: "
                    "src=%s index=%s dst=%s axis=%d",
                    src.to_string().c_str(), index.to_string().c_str(),
                    dst.to_string().c_str(), m_param.axis));
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_eq_dtype(src, dst);
    megdnn_assert(index.dtype == dtype::Int32(), "%s", errmsg().c_str());
    megdnn_assert(src.is_contiguous() && index.is_contiguous() &&
            dst.is_contiguous(), "%s", errmsg().c_str());

    // check index
    TensorShape idx_shp{src};
    -- idx_shp.ndim;
    megdnn_assert(m_param.axis >= 0, "%s", errmsg().c_str());
    for (auto i = static_cast<uint32_t>(m_param.axis); i < idx_shp.ndim; ++i)
        idx_shp[i] = idx_shp[i + 1];
    megdnn_assert(index.eq_shape(idx_shp), "%s idx_shp=%s", errmsg().c_str(), idx_shp.to_string().c_str());

    // check dst
    megdnn_assert(
            m_param.axis < static_cast<int32_t>(src.ndim) && src.ndim >= 2,
            "%s", errmsg().c_str());
    TensorShape dst_shp{src};
    dst_shp.shape[m_param.axis] = 1;
    megdnn_assert(dst.eq_shape(dst_shp), "%s dst_shp=%s", errmsg().c_str(), dst_shp.to_string().c_str());
}

void IndexingOneHotForward::check_exec(const TensorLayout &src,
        const TensorLayout &index, const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, index, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            src, index, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void IndexingSetOneHotForward::check_exec(const TensorLayout &data,
        const TensorLayout &index, const TensorLayout &sub,
        size_t workspace_in_bytes)
{
    check_layout_fwd(data, index, sub);
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            data, index, sub);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

// vim: syntax=cpp.doxygen
