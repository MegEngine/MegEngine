#pragma once

#include "../helper.h"

namespace mlir::tblgen {

bool gen_op_def_c_header(raw_ostream& os, llvm::RecordKeeper& keeper);

bool gen_op_def_c_body(raw_ostream& os, llvm::RecordKeeper& keeper);

}  // namespace mlir::tblgen
