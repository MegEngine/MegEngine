/**
 * \file src/opr/include/megbrain/opr/dnn/local.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

//! param: src, filter
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD2(LocalForward);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD2(GroupLocalForward);

//! param: filter, diff, src
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(LocalBackwardData);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(GroupLocalBackwardData);

//! param: src, diff, filter
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(LocalBackwardFilter);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(GroupLocalBackwardFilter);


using Local = LocalForward;
using GroupLocal = GroupLocalForward;

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
