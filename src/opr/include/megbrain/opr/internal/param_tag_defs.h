#pragma once

#include <cstdint>
#include "megbrain/dtype.h"

namespace mgb {
namespace opr {

namespace param_tag {
enum ParamTag : uint32_t {
    ADD_UPDATE = 1,
    DIMSHUFFLE,
    AXIS_ADD_REMOVE,
    HOST2DEVICE_COPY,
    SUBTENSOR_INDEX_DESC,
    LOOP,
    LOOP_INPUT_MAKER,
    SLEEP,
    NNLIB_EMPTY_CONST
};
}

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
