#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"
using Tensor = custom::Tensor;
#if MGB_CUDA
void launch_cropandpad_kernel(
        const Tensor& inp, const Tensor& percent, const Tensor& pad_val, Tensor& output,
        bool align_corners, std::string mode);
#endif
#endif