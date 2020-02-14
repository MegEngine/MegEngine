/**
 * \file src/opr/impl/indexing.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/indexing.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/opr/internal/indexing_helper_sereg.h"

MGB_SEREG_GET_SUBTENSOR_OPR(IndexingMultiAxisVec);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(IndexingSetMultiAxisVec);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(IndexingIncrMultiAxisVec);
MGB_SEREG_GET_SUBTENSOR_OPR(MeshIndexing);
MGB_SEREG_GET_SUBTENSOR_OPR(BatchedMeshIndexing);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(IncrMeshIndexing);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(BatchedIncrMeshIndexing);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(SetMeshIndexing);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(BatchedSetMeshIndexing);

namespace mgb {
namespace opr {
    MGB_SEREG_OPR(IndexingOneHot, 2);
    MGB_SEREG_OPR(IndexingRemap, 2);
    MGB_SEREG_OPR(IndexingRemapBackward, 3);
    MGB_SEREG_OPR(IndexingSetOneHot, 3);
} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

