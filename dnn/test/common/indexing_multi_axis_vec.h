/**
 * \file dnn/test/common/indexing_multi_axis_vec.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "test/common/opr_proxy.h"

namespace megdnn {
namespace test {

struct OprProxyIndexingMultiAxisVecHelper {
    size_t axes[TensorLayout::MAX_NDIM];

    /*!
     * \brief OprProxy for indexing multi-vec family oprs
     *
     * \param init_axes axes that are indexed
     */
    OprProxyIndexingMultiAxisVecHelper(
            std::initializer_list<size_t> init_axes = {}) {
        size_t i = 0;
        for (auto ax : init_axes)
            axes[i++] = ax;
    }

    OprProxyIndexingMultiAxisVecHelper(SmallVector<size_t> init_axes) {
        size_t i = 0;
        for (auto ax : init_axes)
            axes[i++] = ax;
    }

    IndexingMultiAxisVec::IndexDesc make_index_desc(
            const TensorNDArray& tensors) const {
        megdnn_assert(tensors.size() >= 3);
        IndexingMultiAxisVec::IndexDesc ret;
        ret.resize(tensors.size() - 2);
        for (size_t i = 2; i < tensors.size(); ++i) {
            ret[i - 2] = {axes[i - 2], tensors[i]};
        }
        return ret;
    }

    IndexingMultiAxisVec::IndexDescLayoutOnly make_index_layout(
            const TensorLayoutArray& layouts) const {
        megdnn_assert(layouts.size() >= 3);
        IndexingMultiAxisVec::IndexDescLayoutOnly ret;
        ret.resize(layouts.size() - 2);
        for (size_t i = 2; i < layouts.size(); ++i) {
            ret[i - 2] = {axes[i - 2], layouts[i]};
        }
        return ret;
    }
};

template <>
struct OprProxy<IndexingMultiAxisVec>
        : public OprProxyIndexingMultiAxisVecHelper {
    using OprProxyIndexingMultiAxisVecHelper::
            OprProxyIndexingMultiAxisVecHelper;

    void exec(IndexingMultiAxisVec* opr, const TensorNDArray& tensors) const {
        WorkspaceWrapper W(opr->handle(),
                           opr->get_workspace_in_bytes(tensors[1].layout, axes,
                                                       tensors.size() - 2));
        opr->exec(tensors[0], make_index_desc(tensors), tensors[1],
                  W.workspace());
    }

    void deduce_layout(IndexingMultiAxisVec* opr, TensorLayoutArray& layouts) {
        opr->deduce_layout(layouts[0], make_index_layout(layouts), layouts[1]);
    }
};

template <>
struct OprProxy<IndexingIncrMultiAxisVec>
        : public OprProxyIndexingMultiAxisVecHelper {
    using OprProxyIndexingMultiAxisVecHelper::
            OprProxyIndexingMultiAxisVecHelper;

    void exec(IndexingIncrMultiAxisVec* opr,
              const TensorNDArray& tensors) const {
        WorkspaceWrapper W(opr->handle(),
                           opr->get_workspace_in_bytes(tensors[1].layout, axes,
                                                       tensors.size() - 2));
        opr->exec(tensors[0], tensors[1], make_index_desc(tensors),
                  W.workspace());
    }

    void deduce_layout(IndexingIncrMultiAxisVec*, TensorLayoutArray&) {}
};

template <>
struct OprProxy<IndexingSetMultiAxisVec>
        : public OprProxyIndexingMultiAxisVecHelper {
    using OprProxyIndexingMultiAxisVecHelper::
            OprProxyIndexingMultiAxisVecHelper;

    void exec(IndexingSetMultiAxisVec* opr,
              const TensorNDArray& tensors) const {
        WorkspaceWrapper W(opr->handle(),
                           opr->get_workspace_in_bytes(tensors[1].layout, axes,
                                                       tensors.size() - 2));
        opr->exec(tensors[0], tensors[1], make_index_desc(tensors),
                  W.workspace());
    }

    void deduce_layout(IndexingSetMultiAxisVec*, TensorLayoutArray&) {}
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
