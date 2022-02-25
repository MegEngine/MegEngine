#pragma once

#include "megbrain/graph/bases.h"
#include "megbrain/serialization/extern_c_opr.h"

namespace mgb {

/*!
 * \brief config extern c opr dynamic param
 */
MGE_WIN_DECLSPEC_FUC void config_extern_c_opr_dynamic_param(
        std::unique_ptr<cg::AsyncExecutable>& func,
        std::shared_ptr<ExternCOprParam> param);

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
