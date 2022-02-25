#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace rocm {

class PowCImpl final : public PowC {
public:
    using PowC::PowC;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
