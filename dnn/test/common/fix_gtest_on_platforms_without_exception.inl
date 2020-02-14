/**
 * \file dnn/test/common/fix_gtest_on_platforms_without_exception.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/arch.h"

#if !MEGDNN_ENABLE_EXCEPTIONS
#undef EXPECT_THROW
#undef EXPECT_NO_THROW
#undef EXPECT_ANY_THROW
#undef ASSERT_THROW
#undef ASSERT_NO_THROW
#undef ASSERT_ANY_THROW
#define EXPECT_THROW(x, exc)
#define EXPECT_NO_THROW(x) x
#define EXPECT_ANY_THROW(x)
#define ASSERT_THROW(x, exc)
#define ASSERT_NO_THROW(x) x
#define ASSERT_ANY_THROW(x)
#endif
