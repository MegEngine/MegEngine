#ifdef MAKE_STR
#error "macro_helper.h not used with macro_helper_epilogue.h"
#endif

#define MAKE_STR0(v) #v
#define MAKE_STR(v)  MAKE_STR0(v)

#define CONCAT_STR0(a, b) a##b
#define CONCAT_STR(a, b)  CONCAT_STR0(a, b)

//! add _MEGDNN_SIMD_NAME to given prefix
#define WITH_SIMD_SUFFIX(prefix) CONCAT_STR(prefix##_, MEGDNN_SIMD_NAME)

// vim: syntax=cpp.doxygen
