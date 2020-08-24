/**
 * \file dnn/test/common/mesh_indexing.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/oprs/general.h"
#include "rng.h"
#include "test/common/indexing_multi_axis_vec.h"
#include "test/common/opr_proxy.h"

namespace megdnn {
namespace test {

#define MESH_INDEXING_LIKE_OPR_PROXY(__opr)                                    \
    template <>                                                                \
    struct OprProxy<__opr> : public OprProxyIndexingMultiAxisVecHelper {       \
        using OprProxyIndexingMultiAxisVecHelper::                             \
                OprProxyIndexingMultiAxisVecHelper;                            \
        void exec(__opr* opr, const TensorNDArray& tensors) const {            \
            WorkspaceWrapper W(opr->handle(), opr->get_workspace_in_bytes(     \
                                                      tensors[1].layout, axes, \
                                                      tensors.size() - 2));    \
            opr->exec(tensors[0], make_index_desc(tensors), tensors[1],        \
                      W.workspace());                                          \
        }                                                                      \
        void deduce_layout(__opr* opr, TensorLayoutArray& layouts) {           \
            MEGDNN_MARK_USED_VAR(opr);                                         \
            MEGDNN_MARK_USED_VAR(layouts);                                     \
            opr->deduce_layout(layouts[0], make_index_layout(layouts),         \
                               layouts[1]);                                    \
        }                                                                      \
    };

#define MESH_MODIFY_LIKE_OPR_PROXY(__opr)                                      \
    template <>                                                                \
    struct OprProxy<__opr> : public OprProxyIndexingMultiAxisVecHelper {       \
        using OprProxyIndexingMultiAxisVecHelper::                             \
                OprProxyIndexingMultiAxisVecHelper;                            \
        void exec(__opr* opr, const TensorNDArray& tensors) const {            \
            WorkspaceWrapper W(opr->handle(), opr->get_workspace_in_bytes(     \
                                                      tensors[1].layout, axes, \
                                                      tensors.size() - 2));    \
            opr->exec(tensors[0], tensors[1], make_index_desc(tensors),        \
                      W.workspace());                                          \
        }                                                                      \
        void deduce_layout(__opr*, TensorLayoutArray&) {}                      \
    };

MESH_INDEXING_LIKE_OPR_PROXY(MeshIndexing);
MESH_INDEXING_LIKE_OPR_PROXY(BatchedMeshIndexing);
MESH_MODIFY_LIKE_OPR_PROXY(IncrMeshIndexing);
MESH_MODIFY_LIKE_OPR_PROXY(BatchedIncrMeshIndexing);
MESH_MODIFY_LIKE_OPR_PROXY(SetMeshIndexing);
MESH_MODIFY_LIKE_OPR_PROXY(BatchedSetMeshIndexing);

#undef MESH_PROXY_COMMON
#undef MESH_INDEXING_LIKE_OPR_PROXY
#undef MESH_MODIFY_LIKE_OPR_PROXY

namespace mesh_indexing {
class NoReplacementIndexRNG final : public RNG {
    size_t& m_size;
    std::mt19937_64 m_rng;

public:
    NoReplacementIndexRNG(size_t& sz, size_t seed) : m_size{sz}, m_rng(seed) {}

    void gen(const TensorND& tensor) override {
        std::vector<int> seq;
        for (size_t i = 0; i < m_size; ++i) {
            seq.push_back(i);
        }
        size_t stride = static_cast<size_t>(tensor.layout.stride[0]);
        size_t size = tensor.layout[0];
        if (tensor.layout.ndim == 1) {
            stride = tensor.layout[0];
            size = 1;
        }
        megdnn_assert(stride <= m_size);

        auto ptr = tensor.ptr<int>();
        for (size_t n = 0; n < size; ++n) {
            std::set<int> used;
            COMPAT_RANDOM(seq.begin(), seq.end());
            for (size_t step = 0; step < stride; ++step) {
                megdnn_assert(used.size() < m_size);
                ptr[n * stride + step] = seq[step];
                used.insert(seq[step]);
            }
        }
    }
};
}  // namespace mesh_indexing

}  // namespace test
}  // namespace megdnn
