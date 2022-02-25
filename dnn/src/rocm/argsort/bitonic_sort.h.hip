#pragma once

#include "hip_header.h"
#include <stdint.h>

namespace megdnn {
namespace rocm {

const uint32_t BITONIC_SORT_MAX_LENGTH = 1024;
// cub radix sort seems to be faster with lengths > 1024

/*!
 * \brief bitonic sort for k/v pairs
 *
 * Requires \p length no larger than 4 times of cuda thread num. \p key_inp
 * and \p key_out can be identical, and so are \p value_inp and \p value_out.
 */
template <typename Key, typename Value>
hipError_t bitonic_sort(uint32_t batch, uint32_t length, const Key* key_inp,
                         const Value* value_inp, Key* key_out, Value* value_out,
                         bool ascending, hipStream_t stream);

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

