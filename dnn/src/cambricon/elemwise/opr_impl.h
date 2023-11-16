#pragma once

#include "megdnn/oprs.h"
#include "src/cambricon/handle.h"
#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {

namespace cambricon {

class ElemwiseForwardImpl final : public ElemwiseForwardImplHelper {
public:
    using ElemwiseForwardImplHelper::ElemwiseForwardImplHelper;

    void exec(const TensorNDArray& src, _megdnn_tensor_out dst) override;

    WorkspaceBundle alloc_cnnl_workspace(const TensorNDArray& src, const TensorND& dst);
    void free_cnnl_workspace(const WorkspaceBundle& wk_bundle);
};

}  // namespace cambricon

}  // namespace megdnn