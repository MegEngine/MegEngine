#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"
using Tensor = custom::Tensor;
#if MGB_CUDA
void launch_dilate_kernel(
        const Tensor& inp, const Tensor& kernel, Tensor& output, Tensor& workspace,
        const int iterations, const std::string& border_type,
        const double border_value);
#endif
#endif