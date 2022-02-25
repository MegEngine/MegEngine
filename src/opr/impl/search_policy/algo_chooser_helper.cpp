#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/graph/cg.h"
#include "megbrain/opr/search_policy/algo_chooser.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;
using namespace mixin;
/* ==================== misc impl  ==================== */

AlgoChooserHelper::~AlgoChooserHelper() = default;

void AlgoChooserHelper::set_execution_policy(const ExecutionPolicy& policy) {
    mgb_throw_if(
            m_policy_accessed, InternalError,
            "attempt to modify ExecutionPolicy after it has been accessed");
    m_policy = policy;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
