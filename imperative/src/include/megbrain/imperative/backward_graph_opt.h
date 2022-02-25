#pragma once

#include "./op_def.h"

namespace mgb::imperative {

struct OptimizedBackwardGraphResult {
    Subgraph precomp;
    Subgraph backward;
    SmallVector<bool> save_for_backward;
    SmallVector<bool> input_has_grad;

    OptimizedBackwardGraphResult(const EncodedSubgraph& bgraph);
};

}  // namespace mgb::imperative
