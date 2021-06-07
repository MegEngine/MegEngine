/**
 * \file src/opr/impl/rand.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/rand.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace opr {

    MGB_SEREG_OPR(UniformRNG, 1);
    MGB_SEREG_OPR(GaussianRNG, 1);
    MGB_SEREG_OPR(GammaRNG, 2);
    MGB_SEREG_OPR(PoissonRNG, 1);
    MGB_SEREG_OPR(PermutationRNG, 1);
    MGB_SEREG_OPR(BetaRNG, 2);

} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

