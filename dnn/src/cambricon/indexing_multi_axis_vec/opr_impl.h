#pragma once

#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

class IndexingMultiAxisVecImpl final : public IndexingMultiAxisVec {
public:
    using IndexingMultiAxisVec::IndexingMultiAxisVec;

    size_t get_workspace_in_bytes(size_t dst_idx_size) override;

    void exec(
            _megdnn_tensor_in src, const IndexDesc& index, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    WorkspaceBundle get_workspace_bundle(
            const TensorLayout& src_layout, const IndexDesc& index,
            const TensorLayout& dst_layout);
    std::tuple<TensorND, std::vector<TensorND>, std::vector<bool>> prepare_src_and_index(
            const TensorND& src, const IndexDesc& index,
            const WorkspaceBundle& wk_bundle);
    std::pair<TensorND, std::vector<TensorND>> transpose_src_and_indices(
            const TensorND& src, const std::vector<TensorND>& indices,
            const std::vector<bool>& is_indices_defined,
            const WorkspaceBundle& wk_bundle);
};

class IndexingSetMultiAxisVecImpl final : public IndexingSetMultiAxisVec {
public:
    using IndexingSetMultiAxisVec::IndexingSetMultiAxisVec;

    size_t get_workspace_in_bytes(size_t dst_idx_size) override;

    void exec(
            _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
            _megdnn_workspace workspace) override;

    WorkspaceBundle get_workspace_bundle(
            const TensorLayout& src_layout, const TensorLayout& value_layout,
            const IndexDesc& index);
    std::tuple<
            TensorND, std::vector<TensorND>, std::vector<bool>, std::vector<size_t>,
            TensorND>
    prepare_src_index_value(
            const TensorND& src, const IndexDesc& index, const TensorND& value,
            const WorkspaceBundle& wk_bundle);
};

class IndexingIncrMultiAxisVecImpl final : public IndexingIncrMultiAxisVec {
public:
    using IndexingIncrMultiAxisVec::IndexingIncrMultiAxisVec;

    size_t get_workspace_in_bytes(size_t dst_idx_size) override;

    void exec(
            _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
            _megdnn_workspace workspace) override;
    WorkspaceBundle get_workspace_bundle(
            const TensorLayout& src_layout, const TensorLayout& value_layout,
            const IndexDesc& index);
    std::tuple<
            TensorND, std::vector<TensorND>, std::vector<bool>, std::vector<size_t>,
            TensorND>
    prepare_src_index_value(
            const TensorND& src, const IndexDesc& index, const TensorND& value,
            const WorkspaceBundle& wk_bundle);
    void fallback_to_kernel(
            _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
            _megdnn_workspace workspace);
};
}  // namespace cambricon

}  // namespace megdnn

// vim: syntax=cpp.doxygen
