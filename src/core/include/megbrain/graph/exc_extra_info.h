#pragma once

#include "megbrain/common.h"
#include "megbrain/exception.h"

namespace mgb {
namespace cg {

class OperatorNodeBase;

/*!
 * \brief associate an operator with an exception
 */
class OperatorNodeExcExtraInfo final : public MegBrainError::ExtraInfo {
    OperatorNodeBase* m_opr;

public:
    class ExcMaker;

    OperatorNodeExcExtraInfo(OperatorNodeBase* opr) : m_opr(opr) {}

    /*!
     * \brief record an operator on the exception
     * \return modified \p exc
     */
    static MegBrainError& record(OperatorNodeBase* opr, MegBrainError& exc) {
        mgb_assert(opr && !exc.extra_info());
        exc.extra_info(std::make_shared<OperatorNodeExcExtraInfo>(opr));
        return exc;
    }

    /*!
     * \brief get associated operator
     */
    OperatorNodeBase* opr() const { return m_opr; }
};

/*!
 * \brief helper class to create exception object associated with an operator
 *
 * Typical usecase: mgb_throw(ExcMaker{opr}::make<Exception>
 */
class OperatorNodeExcExtraInfo::ExcMaker {
    OperatorNodeBase* const m_opr;

public:
    ExcMaker(OperatorNodeBase* opr) : m_opr{opr} {}

    template <class Exc, typename... Args>
    Exc make(Args&&... args) {
        Exc exc{std::forward<Args>(args)...};
        OperatorNodeExcExtraInfo::record(m_opr, exc);
        return exc;
    }

    template <class Exc, typename... Args>
    std::unique_ptr<Exc> make_unique(Args&&... args) {
        auto exc = std::make_unique<Exc>(std::forward<Args>(args)...);
        OperatorNodeExcExtraInfo::record(m_opr, *exc);
        return exc;
    }
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
