#pragma once

#include "src/naive/powc/opr_impl.h"

namespace megdnn {
namespace fallback {

class PowCImpl final : public naive::PowCImpl {
    template <typename T>
    void do_exec_ct(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i);

public:
    using naive::PowCImpl::PowCImpl;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
