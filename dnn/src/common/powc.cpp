#include "megdnn/oprs/general.h"

#include <cmath>
#include "src/common/utils.h"

using namespace megdnn;

void PowC::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    megdnn_assert(
            src.layout.dtype == dst.layout.dtype &&
                    src.layout.dtype.category() == DTypeCategory::FLOAT &&
                    src.layout.eq_shape(dst.layout),
            "invalid layout: %s vs %s", src.layout.to_string().c_str(),
            dst.layout.to_string().c_str());
    int iv, *ivp = nullptr;
    float fv, *fvp = nullptr;
    float p = param().exp;
    int pi = static_cast<int>(std::round(p));
    if (std::abs(static_cast<float>(pi) - p) < std::numeric_limits<float>::epsilon()) {
        iv = pi;
        ivp = &iv;
    } else {
        fv = p;
        fvp = &fv;
    }
    do_exec(src, dst, fvp, ivp);
}

// vim: syntax=cpp.doxygen
