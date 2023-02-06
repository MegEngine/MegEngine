#pragma once
#include "megdnn/handle.h"
#include "megdnn/oprs.h"
#include "src/common/reduce_helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/handle.h"
#include "src/cuda/multi_head_attn/helper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class MultiHeadAttnForwardImpl final : public MultiHeadAttnForward {
public:
    using MultiHeadAttnForward::MultiHeadAttnForward;
#if CUDNN_VERSION >= 8004
    MultiHeadAttnStatus desc_status;
#endif

    void exec(
            _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
            _megdnn_tensor_in wqkv, _megdnn_tensor_out out,
            _megdnn_tensor_out reserveSpace, _megdnn_workspace workspace) override;
    void deduce_layout(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& wqkv, TensorLayout& out,
            TensorLayout& reserveSpace);
    size_t get_reservespace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& wqkv,
            const TensorLayout& out, const TensorLayout& reserveSpace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& wqkv,
            const TensorLayout& out, const TensorLayout& reserveSpace) override;
};

class MultiHeadAttnBackwardImpl final : public MultiHeadAttnBackward {
public:
    using MultiHeadAttnBackward::MultiHeadAttnBackward;
#if CUDNN_VERSION >= 8004
    MultiHeadAttnStatus desc_status;
#endif
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
            _megdnn_tensor_in values, _megdnn_tensor_in wqkv,
            _megdnn_tensor_in reserveSpace, _megdnn_tensor_out dqueries,
            _megdnn_tensor_out dkeys, _megdnn_tensor_out dvalues,
            _megdnn_tensor_out dweights, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& queries,
            const TensorLayout& keys, const TensorLayout& values,
            const TensorLayout& wqkv, const TensorLayout& reserveSpace,
            const TensorLayout& dqueries, const TensorLayout& dkeys,
            const TensorLayout& dvalues, const TensorLayout& dweights) override;
};
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
