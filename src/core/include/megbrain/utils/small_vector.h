#pragma once

#pragma once

#include "megbrain/common.h"
#include "megbrain/utils/hash.h"
#include "megdnn/thin/small_vector.h"

namespace mgb {

using ::megdnn::capacity_in_bytes;
using ::megdnn::find;
using ::megdnn::SmallVector;
using ::megdnn::SmallVectorImpl;

/*!
 * \brief explicit hash specification for SmallVectorImpl
 */
template <typename T>
struct HashTrait<SmallVectorImpl<T>> {
    static size_t eval(const SmallVectorImpl<T>& val) {
        size_t rst = hash(val.size());
        for (auto&& i : val)
            rst = hash_pair_combine(rst, ::mgb::hash(i));
        return rst;
    }
};

/*!
 * \brief explicit hash specification for SmallVector
 */
template <typename T, unsigned N>
struct HashTrait<SmallVector<T, N>> {
    static size_t eval(const SmallVector<T, N>& val) {
        return HashTrait<SmallVectorImpl<T>>::eval(val);
    }
};

}  // end namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
