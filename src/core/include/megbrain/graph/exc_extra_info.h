/**
 * \file src/core/include/megbrain/graph/exc_extra_info.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/exception.h"
#include "megbrain/common.h"

namespace mgb {
namespace cg {

class OperatorNodeBase;

/*!
 * \brief associate an operator with an exception
 */
class OperatorNodeExcExtraInfo final: public MegBrainError::ExtraInfo {
    OperatorNodeBase *m_opr;

    public:
        class ExcMaker;

        OperatorNodeExcExtraInfo(OperatorNodeBase *opr):
            m_opr(opr)
        {}


        /*!
         * \brief record an operator on the exception
         * \return modified \p exc
         */
        static MegBrainError& record(
                OperatorNodeBase *opr, MegBrainError &exc) {
            mgb_assert(opr && !exc.extra_info());
            exc.extra_info(std::make_shared<OperatorNodeExcExtraInfo>(opr));
            return exc;
        }

        /*!
         * \brief get associated operator
         */
        OperatorNodeBase* opr() const {
            return m_opr;
        }
};

/*!
 * \brief helper class to create exception object associated with an operator
 *
 * Typical usecase: mgb_throw(ExcMaker{opr}::make<Exception>
 */
class OperatorNodeExcExtraInfo::ExcMaker {
    OperatorNodeBase * const m_opr;

    public:
        ExcMaker(OperatorNodeBase *opr):
            m_opr{opr}
        {}

        template<class Exc, typename... Args>
        Exc make(Args&&... args) {
            Exc exc{std::forward<Args>(args)...};
            OperatorNodeExcExtraInfo::record(m_opr, exc);
            return exc;
        }

        template<class Exc, typename... Args>
        std::unique_ptr<Exc> make_unique(Args&&... args) {
            auto exc = std::make_unique<Exc>(std::forward<Args>(args)...);
            OperatorNodeExcExtraInfo::record(m_opr, *exc);
            return exc;
        }
};

} // namespace cg
} // namesapce mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

