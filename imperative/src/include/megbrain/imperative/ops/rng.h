/**
 * \file imperative/src/include/megbrain/imperative/ops/rng.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative::rng {

using Handle = size_t;

Handle new_handle(CompNode comp_node, uint64_t seed);
size_t delete_handle(Handle handle);
void set_global_rng_seed(uint64_t seed);
uint64_t get_global_rng_seed();

} // namespace mgb::imperative::rng
