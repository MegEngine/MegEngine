#pragma once

#include "./checker.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace test {
class NonZeroTestcase {
public:
    std::unique_ptr<uint8_t> m_mem;
    NonZero::Param m_param;
    TensorND m_data;
    std::vector<int> correct_answer;

    NonZeroTestcase(NonZero::Param param, const TensorLayout& data)
            : m_param(param), m_data(nullptr, data) {}
    using Result = TensorND;
    using CUDAResult = std::shared_ptr<TensorND>;
    Result run_naive(NonZero* opr);
    CUDAResult run_cuda(NonZero* opr);
    static std::vector<NonZeroTestcase> make();
    static void Assert(
            std::vector<int>& correct_answer, int ndim, NonZeroTestcase::Result result);
};

}  // namespace test
}  // namespace megdnn