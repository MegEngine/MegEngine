#include "hcc_detail/hcc_defs_prologue.h"

#include "hip_header.h"
#include "src/common/indexing_multi_axis_vec_kdef.h"
#define KERN_APPLY_OPR_OPR  ::megdnn::indexing_multi_axis_vec_kdef::OprFwd
#include "./kern_apply_opr_impl.hipinl"

// vim: ft=cuda syntax=cpp.doxygen

