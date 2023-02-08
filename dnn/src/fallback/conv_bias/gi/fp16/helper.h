#pragma once
#include "src/fallback/general_intrinsic/gi_float16.h"
#if defined(GI_SUPPORT_F16)

#define ADDF16  GiAddFloat16
#define SUBF16  GiSubtractFloat16
#define MULSF16 GiMultiplyScalerFloat16

#endif
#define CONCAT(a, idx) a##idx
// vim: syntax=cpp.doxygen
