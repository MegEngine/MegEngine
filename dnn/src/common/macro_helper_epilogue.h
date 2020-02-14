/**
 * \file dnn/src/common/macro_helper_epilogue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef MAKE_STR
#error "macro_helper_epilogue.h must be used after macro_helper.h"
#endif

#undef MAKE_STR
#undef MAKE_STR0
#undef CONCAT_STR
#undef CONCAT_STR0
#undef WITH_SIMD_SUFFIX
