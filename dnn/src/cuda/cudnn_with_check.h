#pragma once

#include <cudnn.h>

#if !(CUDNN_MAJOR >= 5)
#error "CUDNN must be version at least 5."
#endif
