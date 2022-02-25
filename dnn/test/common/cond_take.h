#pragma once

#include "./checker.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace test {
class CondTakeTestcase {
    std::unique_ptr<uint8_t> m_mem;
    CondTake::Param m_param;
    TensorND m_data, m_mask;

    CondTakeTestcase(
            CondTake::Param param, const TensorLayout& data, const TensorLayout& mask)
            : m_param{param}, m_data{nullptr, data}, m_mask{nullptr, mask} {}

public:
    //! pair of (data, idx)
    using Result = std::pair<std::shared_ptr<TensorND>, std::shared_ptr<TensorND>>;
    Result run(CondTake* opr);
    static std::vector<CondTakeTestcase> make();
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
