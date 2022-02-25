#pragma once

#include "../helper.h"

namespace mlir::tblgen {

bool gen_enum_param_list_macro(raw_ostream& os, llvm::RecordKeeper& keeper);

}  // namespace mlir::tblgen
