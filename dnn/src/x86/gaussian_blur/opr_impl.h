#pragma once
#include <cstring>
#include "megdnn/oprs.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"

namespace megdnn {
namespace x86 {

class GaussianBlurImpl : public GaussianBlur {
private:
    template <typename T>
    void gaussian_blur_exec(
            const TensorND& src_tensor, const TensorND& dst_tensor, const Param& param);
    void gaussian_blur_exec_8u(
            const TensorND& src_tensor, const TensorND& dst_tensor, const Param& param);

    template <typename T>
    void createGaussianKernels(
            megcv::Mat<T>& kx, megcv::Mat<T>& ky, megcv::Size ksize, double sigma_r,
            double sigma_c);

public:
    using GaussianBlur::GaussianBlur;
    using Param = param::GaussianBlur;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    void exec(_megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace)
            override;

};  // class GaussianBlurImpl

}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
