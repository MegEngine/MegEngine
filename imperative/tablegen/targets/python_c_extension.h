#pragma once

#include "../helper.h"

namespace mlir::tblgen {

bool gen_op_def_python_c_extension(raw_ostream& os, llvm::RecordKeeper& keeper);

}  // namespace mlir::tblgen
