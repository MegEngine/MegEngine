#pragma once

#include "../helper.h"

namespace mlir::tblgen {

bool gen_op_def_pybind11(raw_ostream& os, llvm::RecordKeeper& keeper);

}  // namespace mlir::tblgen
