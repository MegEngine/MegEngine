#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"
#if MGB_CUDA
void launch_fillpoly_kernel(
        const custom::Tensor& inp, const custom::Tensor& points,
        const custom::Tensor& lens, const custom::Tensor& color,
        custom::Tensor& output);
#endif
#endif