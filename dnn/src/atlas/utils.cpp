#include "src/atlas/utils.h"
#include "megcore_atlas.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace atlas;

void atlas::__throw_acl_error__(aclError err, const char* msg) {
    auto s = ssprintf(
            "acl return %s(%d) occurred; expr: %s", megcore::atlas::get_error_str(err),
            int(err), msg);
    megdnn_throw(s.c_str());
}

// vim: syntax=cpp.doxygen
