#pragma once

#include "megdnn/oprs.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"

namespace megdnn {
namespace atlas {

class LinspaceImpl : public Linspace {
public:
    using Linspace::Linspace;
    void exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }

private:
};

}  // namespace atlas
}  // namespace megdnn