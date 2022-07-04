#pragma once

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cuda {

class RegionRestrictedConvolutionForwardImpl
        : public RegionRestrictedConvolutionForward {
public:
    using RegionRestrictedConvolutionForward::RegionRestrictedConvolutionForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in rin,
            _megdnn_tensor_in rout, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class RegionRestrictedConvolutionBackwardDataImpl
        : public RegionRestrictedConvolutionBackwardData {
public:
    using RegionRestrictedConvolutionBackwardData::
            RegionRestrictedConvolutionBackwardData;
    void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
            _megdnn_tensor_in rout, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override;
};

class RegionRestrictedConvolutionBackwardFilterImpl
        : public RegionRestrictedConvolutionBackwardFilter {
public:
    using RegionRestrictedConvolutionBackwardFilter::
            RegionRestrictedConvolutionBackwardFilter;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
            _megdnn_tensor_in rout, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
