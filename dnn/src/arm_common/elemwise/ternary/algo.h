/**
 * \file dnn/src/arm_common/elemwise/ternary/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/arm_common/elemwise/opr_impl.h"
namespace megdnn {
namespace arm_common {

#define DECL_CB(case)                                                                 \
    class ElemwiseImpl::AlgoTernaryFma3##case final : public ElemwiseImpl::AlgoBase { \
        mutable std::string m_name;                                                   \
        AlgoAttribute attribute() const override {                                    \
            return AlgoAttribute::REPRODUCIBLE;                                       \
        }                                                                             \
        const char* name() const override {                                           \
            if (m_name.empty()) {                                                     \
                m_name = ssprintf("Elemwise::AlgoTernaryFma3" #case);                 \
            }                                                                         \
            return m_name.c_str();                                                    \
        }                                                                             \
        bool is_available(const KernParam&) const override;                           \
        void exec(const KernParam&) const override;                                   \
    };

DECL_CB(VecVecVec);
DECL_CB(VecVecScalar);
DECL_CB(Bcast101VecBcast101);
DECL_CB(Bcast111CVecBcast111C);
DECL_CB(Bcast101xXVecBcast101xX);
DECL_CB(VecBcast101Vec);
DECL_CB(VecBcast111CVec);
DECL_CB(VecBcast101xXVec);
DECL_CB(VecScalarVec);
DECL_CB(VecScalarScalar);
#undef DECL_CB
}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
