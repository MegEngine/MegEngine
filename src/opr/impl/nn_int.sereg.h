/**
 * \file src/opr/impl/nn_int.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/nn_int.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
template <>
struct OprMaker<opr::ElemwiseMultiType, 0>
        : public OprMakerVariadic<opr::ElemwiseMultiType> {};

}  // namespace serialization

namespace opr {
MGB_SEREG_OPR(ElemwiseMultiType, 0);
MGB_SEREG_OPR(AffineInt, 3);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
