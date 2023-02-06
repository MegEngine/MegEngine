#pragma once
#include <memory>
#include "megdnn/oprs.h"
#include "megdnn/oprs/cv.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace naive {

class MultiHeadAttnForwardImpl final : public MultiHeadAttnForward {
public:
    using MultiHeadAttnForward::MultiHeadAttnForward;
    void exec(
            _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
            _megdnn_tensor_in wqkv, _megdnn_tensor_out out,
            _megdnn_tensor_out reserveSpace, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& wqkv,
            const TensorLayout& out, const TensorLayout& reserveSpace) override;
    size_t get_reservespace_in_bytes(
            const TensorLayout& /*queries*/, const TensorLayout& /*keys*/,
            const TensorLayout& /*values*/, const TensorLayout& /*wqkv*/,
            const TensorLayout& /*out*/,
            const TensorLayout& /*reserveSpace*/) override {
        return 0;
    }
};

class MultiHeadAttnBackwardImpl final : public MultiHeadAttnBackward {
public:
    using MultiHeadAttnBackward::MultiHeadAttnBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
            _megdnn_tensor_in values, _megdnn_tensor_in wqkv,
            _megdnn_tensor_in reserveSpace, _megdnn_tensor_out dqueries,
            _megdnn_tensor_out dkeys, _megdnn_tensor_out dvalues,
            _megdnn_tensor_out dweights, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /*diff*/, const TensorLayout& /* queries*/,
            const TensorLayout& /*keyes*/, const TensorLayout& /* values*/,
            const TensorLayout& /*wqkv*/, const TensorLayout& /* reserveSpace*/,
            const TensorLayout& /*dqueries*/, const TensorLayout& /* dkeyes*/,
            const TensorLayout& /*dvalues*/,
            const TensorLayout& /* dweights*/) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
