#include "megbrain/opr/indexing.h"
#include "megbrain/opr/internal/indexing_helper_sereg.h"
#include "megbrain/serialization/sereg.h"

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
MGB_SEREG_OPR(Diag, 1);
MGB_SEREG_OPR(DiagBackward, 2);
MGB_SEREG_OPR(IndexingOneHot, 2);
MGB_SEREG_OPR(IndexingRemap, 2);
MGB_SEREG_OPR(IndexingRemapBackward, 3);
MGB_SEREG_OPR(IndexingSetOneHot, 3);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
