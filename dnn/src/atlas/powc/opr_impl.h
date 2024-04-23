#pragma once

#include "megdnn/oprs.h"
#include "megdnn/oprs/general.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"

namespace megdnn {
namespace atlas {

class PowCImpl : public PowC {
public:
    using PowC::PowC;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override;
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen