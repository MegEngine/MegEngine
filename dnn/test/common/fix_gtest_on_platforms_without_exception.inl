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
