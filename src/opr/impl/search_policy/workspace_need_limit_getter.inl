#pragma once

#include "megbrain/opr/search_policy/algo_chooser.h"

#include "../internal/megdnn_opr_wrapper.inl"

namespace mgb {
namespace opr {
namespace intl {

#define cb(_Opr)                                           \
    template <>                                            \
    struct AutoAddWorkspaceNeedLimitGetter<megdnn::_Opr> { \
        static constexpr bool val = true;                  \
    };
DNN_FOREACH_FASTRUN_OPR(cb)

#undef cb

}  // namespace intl
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
