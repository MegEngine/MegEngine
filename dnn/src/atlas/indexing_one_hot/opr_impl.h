#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class IndexingOneHotForwardImpl final : public IndexingOneHotForward {
public:
    using IndexingOneHotForward::IndexingOneHotForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in index, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& index,
            const TensorLayout& dst) override {
        size_t size = 0;
        DType index_type = index.dtype;
        size += index_type.size(index.total_nr_elems());
        return size;
    }
};

class IndexingSetOneHotForwardImpl final : public IndexingSetOneHotForward {
public:
    using IndexingSetOneHotForward::IndexingSetOneHotForward;
    void exec(
            _megdnn_tensor_inout src, _megdnn_tensor_in index, _megdnn_tensor_in sub,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& index,
            const TensorLayout& dst) override {
        size_t size = 0;
        DType index_type = index.dtype;
        size += index_type.size(index.total_nr_elems());
        return size;
    }
};
}  // namespace atlas
}  // namespace megdnn