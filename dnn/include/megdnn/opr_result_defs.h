#pragma once

#include <stdint.h>

namespace megdnn {
namespace opr_result {

struct Checksum {
    uint32_t checksum;
    union {
        int32_t iv;
        float fv;
    } last_val;

    bool operator==(const Checksum& rhs) const {
        return checksum == rhs.checksum && last_val.iv == rhs.last_val.iv;
    }

    bool operator!=(const Checksum& rhs) const { return !operator==(rhs); }
};

}  // namespace opr_result
}  // namespace megdnn

// vim: syntax=cpp.doxygen
