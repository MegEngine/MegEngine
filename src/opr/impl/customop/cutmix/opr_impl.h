#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"
using Tensor = custom::Tensor;
#if MGB_CUDA
void launch_cuda_kernel(
        const Tensor& inp1, const Tensor& inp2, const Tensor& cx, const Tensor& cy,
        const Tensor& cut_h, const Tensor& cut_w, Tensor& output, Tensor& bbx1,
        Tensor& bbx2, Tensor& bby1, Tensor& bby2);
#endif
#endif