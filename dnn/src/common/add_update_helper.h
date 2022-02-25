#include "megdnn/oprs.h"

#include "src/common/elemwise_helper.cuh"

namespace megdnn {

class AddUpdateForwardHelper : public AddUpdateForward {
    using AddUpdateForward::AddUpdateForward;

protected:
    ElemwiseOpParamN<2> make_param(_megdnn_tensor_inout dst, _megdnn_tensor_in delta);
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
