#include "megcore.h"
#include "src/common/utils.h"

const char* megcoreGetErrorName(megcoreStatus_t status) {
#define CASE(x) \
    case x:     \
        return (#x)
    switch (status) {
        CASE(megcoreSuccess);
        CASE(megcoreErrorMemoryAllocation);
        CASE(megcoreErrorInvalidArgument);
        CASE(megcoreErrorInvalidDeviceHandle);
        CASE(megcoreErrorInternalError);
        CASE(megcoreErrorInvalidComputingHandle);
        default:
            return "<Unknown MegCore Error>";
    }
#undef CASE
}

// vim: syntax=cpp.doxygen
