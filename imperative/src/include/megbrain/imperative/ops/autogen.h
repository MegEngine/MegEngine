/**
 * \file imperative/src/include/megbrain/imperative/ops/autogen.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"
#include "megdnn/opr_param_defs.h"
#include "megbrain/opr/param_defs.h"

#include "megbrain/utils/hash.h"

namespace mgb::imperative {

// TODO: split into separate files to avoid recompiling all
// impl/ops/*.cpp on each modification of ops.td
#include "./opdef.h.inl"

} // namespace mgb::imperative
