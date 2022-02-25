#pragma once

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative::rng {

using Handle = size_t;

Handle new_handle(CompNode comp_node, uint64_t seed);
size_t delete_handle(Handle handle);
void set_global_rng_seed(uint64_t seed);
uint64_t get_global_rng_seed();
CompNode get_rng_handle_compnode(Handle handle);

}  // namespace mgb::imperative::rng
