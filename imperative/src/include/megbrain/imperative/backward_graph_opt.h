/**
 * \file imperative/src/include/megbrain/imperative/backward_graph_opt.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./op_def.h"

namespace mgb::imperative {

struct OptimizedBackwardGraphResult {
    std::shared_ptr<OpDef> precomp;
    std::shared_ptr<OpDef> backward;
    std::vector<bool> save_for_backward;
    std::vector<bool> input_has_grad;

    OptimizedBackwardGraphResult(const BackwardGraphResult& bgraph);
};

} // namespace mgb::imperative
