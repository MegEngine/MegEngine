#pragma once
#include "megdnn/oprs.h"

#include <cstring>
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"

namespace megdnn {
namespace fallback {

class GaussianBlurImpl : public GaussianBlur {
private:
    template <typename T>
    void gaussian_blur_exec(const TensorND& src_tensor, const TensorND& dst_tensor);

    void gaussian_blur_exec_8u(const TensorND& src_tensor, const TensorND& dst_tensor);
    template <typename T>
    void createGaussianKernels(
            megcv::Mat<T>& kx, megcv::Mat<T>& ky, megcv::Size ksize, double sigma_x,
            double sigma_y);

public:
    using GaussianBlur::GaussianBlur;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

};  // class GaussianBlurImpl

}  // namespace fallback
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
