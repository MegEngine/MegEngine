/**
 * \file imperative/tablegen/targets/cpp_class.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "../helper.h"

namespace mlir::tblgen {

bool gen_op_def_c_header(raw_ostream &os, llvm::RecordKeeper &keeper);

bool gen_op_def_c_body(raw_ostream &os, llvm::RecordKeeper &keeper);

} // namespace mlir::tblgen
