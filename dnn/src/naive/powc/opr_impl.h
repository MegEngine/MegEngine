#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace naive {

class PowCImpl : public PowC {
    template <typename T>
    void do_exec_ct(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i);

public:
    using PowC::PowC;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
