#pragma once
#include "src/arm_common/elemwise/opr_impl.h"
namespace megdnn {
namespace arm_common {
class ElemwiseImpl::AlgoUnary final : public ElemwiseImpl::AlgoBase {
    mutable std::string m_name;

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ssprintf("Elemwise::AlgoUnary");
        }
        return m_name.c_str();
    }

    bool is_available(const KernParam&) const override;
    void exec(const KernParam&) const override;
};

}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
