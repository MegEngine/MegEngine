#pragma once

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/utils/to_string.h"
#include "megbrain/opr/param_defs.h"
#include "megdnn/opr_param_defs.h"

#include "megbrain/utils/hash.h"

namespace mgb::imperative {

// TODO: split into separate files to avoid recompiling all
// impl/ops/*.cpp on each modification of ops.td
#include "./opdef.h.inl"

}  // namespace mgb::imperative
