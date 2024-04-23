#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class GaussianRNGImpl : public GaussianRNG {
public:
    using GaussianRNG::GaussianRNG;
    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

class UniformRNGImpl : public UniformRNG {
public:
    using UniformRNG::UniformRNG;
    void exec(_megdnn_tensor_inout dst, _megdnn_workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }
};

}  // namespace atlas
}  // namespace megdnn