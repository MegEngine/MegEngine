#include "megdnn/version.h"
#include "src/common/version_symbol.h"

using namespace megdnn;

Version megdnn::get_version() {
    return {MEGDNN_MAJOR, MEGDNN_MINOR, MEGDNN_PATCH};
}

MEGDNN_VERSION_SYMBOL3(MEGDNN, MEGDNN_MAJOR, MEGDNN_MINOR, MEGDNN_PATCH);

// vim: syntax=cpp.doxygen
