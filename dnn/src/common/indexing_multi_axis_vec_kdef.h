#include "megdnn/arch.h"

#if MEGDNN_CC_HOST && !defined(__device__)
#define __device__
#define def_device 1
#endif

namespace megdnn {
namespace indexing_multi_axis_vec_kdef {

struct OprFwd {
    template <typename ctype>
    __device__ static void apply(ctype data, ctype& value) {
        value = data;
    }
};

struct OprSet {
    template <typename ctype>
    __device__ static void apply(ctype& data, ctype value) {
        data = value;
    }
};

struct OprIncr {
    template <typename ctype>
    __device__ static void apply(ctype& data, ctype value) {
        data += value;
    }
};

}  // namespace indexing_multi_axis_vec_kdef
}  // namespace megdnn

#if def_device
#undef __device__
#undef def_device
#endif

// vim: syntax=cpp.doxygen
