#pragma once

#include "megdnn/oprs.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"
#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {

namespace atlas {

class ElemwiseForwardImpl final : public ElemwiseForwardImplHelper {
public:
    using ElemwiseForwardImplHelper::ElemwiseForwardImplHelper;

    void exec(const TensorNDArray& src, _megdnn_tensor_out dst) override;
};

}  // namespace atlas

}  // namespace megdnn
