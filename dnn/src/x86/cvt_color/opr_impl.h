#pragma once
#include "megdnn/oprs.h"
#include "src/naive/cvt_color/opr_impl.h"

namespace megdnn {
namespace x86 {

class CvtColorImpl : public naive::CvtColorImpl {
private:
    template <typename T>
    void cvt_color_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Param::Mode mode);

public:
    using naive::CvtColorImpl::CvtColorImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
