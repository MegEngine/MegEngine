#pragma once

#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace atlas {

class IndexingMultiAxisVecImpl final : public IndexingMultiAxisVec {
public:
    using IndexingMultiAxisVec::IndexingMultiAxisVec;

    size_t get_workspace_in_bytes(size_t) override { return 0; }

    void exec(
            _megdnn_tensor_in src, const IndexDesc& index, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

class IndexingSetMultiAxisVecImpl final : public IndexingSetMultiAxisVec {
public:
    using IndexingSetMultiAxisVec::IndexingSetMultiAxisVec;

    size_t get_workspace_in_bytes(size_t) override { return 0; }

    void exec(
            _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
            _megdnn_workspace workspace) override;
};

class IndexingIncrMultiAxisVecImpl final : public IndexingIncrMultiAxisVec {
public:
    using IndexingIncrMultiAxisVec::IndexingIncrMultiAxisVec;

    size_t get_workspace_in_bytes(size_t) override { return 0; }

    void exec(
            _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
            _megdnn_workspace workspace) override;
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
