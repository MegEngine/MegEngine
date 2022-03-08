/**
 * \file dnn/src/fallback/elemwise/gi_impl/unary/algo.h
 */
#pragma once
#include "src/fallback/elemwise/opr_impl.h"
namespace megdnn {
namespace fallback {
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

}  // namespace fallback
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
