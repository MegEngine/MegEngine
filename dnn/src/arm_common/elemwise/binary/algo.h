#pragma once
#include "src/arm_common/elemwise/opr_impl.h"
namespace megdnn {
namespace arm_common {

#define DECL_CB(case)                                                            \
    class ElemwiseImpl::AlgoBinary##case final : public ElemwiseImpl::AlgoBase { \
        mutable std::string m_name;                                              \
        AlgoAttribute attribute() const override {                               \
            return AlgoAttribute::REPRODUCIBLE;                                  \
        }                                                                        \
        const char* name() const override {                                      \
            if (m_name.empty()) {                                                \
                m_name = ssprintf("Elemwise::AlgoBinaryCase" #case);             \
            }                                                                    \
            return m_name.c_str();                                               \
        }                                                                        \
        bool is_available(const KernParam&) const override;                      \
        void exec(const KernParam&) const override;                              \
    };

DECL_CB(VecVec);
DECL_CB(VecScalar);
DECL_CB(VecBcast101);
DECL_CB(VecBcastX0X);
DECL_CB(VecBcast111C);
DECL_CB(VecBcast101xX);
#undef DECL_CB
}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
