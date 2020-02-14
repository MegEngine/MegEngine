/**
 * \file src/opr/include/megbrain/opr/internal/mixin_base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace opr {

using OperatorNodeBaseCtorParam = cg::OperatorNodeBase::CtorParamPack;

/*!
 * \brief opr impl mixins, like cg::mixin
 */
namespace mixin  {
    using cg::OperatorNodeBase;
    using cg::mixin::CheckBase;

} // namespace mixin

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

