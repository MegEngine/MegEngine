#pragma once

#include "../../core/impl/graph/cg_impl.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/utils/hash_ct.h"

namespace mgb {
namespace gopt {
namespace {
#define CHECK_OR_RETURN(x) \
    if (!(x)) {            \
        return false;      \
    }

class FindNext {
    using DepType = cg::OperatorNodeProp::DepType;

public:
    FindNext(OptState& opt) {
        opt.graph().iter([&](OperatorNodeBase* opr) {
            for (auto&& i : opr->node_prop().dep_map()) {
                m_readers[i.first->owner_opr()].emplace_back(opr, i.second);
            }
        });
    }
    size_t used_count(OperatorNodeBase* opr) { return m_readers[opr].size(); }

private:
    ThinHashMap<OperatorNodeBase*, SmallVector<std::pair<OperatorNodeBase*, DepType>>>
            m_readers;
};
}  // namespace
}  // namespace gopt

}  // namespace mgb