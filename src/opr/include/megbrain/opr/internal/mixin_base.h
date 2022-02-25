#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace opr {

using OperatorNodeBaseCtorParam = cg::OperatorNodeBase::CtorParamPack;

/*!
 * \brief opr impl mixins, like cg::mixin
 */
namespace mixin {
using cg::OperatorNodeBase;
using cg::mixin::CheckBase;

}  // namespace mixin

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
