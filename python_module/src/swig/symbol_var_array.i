/*
 * $File: symbol_var_array.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

/*
 * In megbrain, SymbolVarArray is SmallVector<SymbolVar>.
 *
 * I do no want to convert between std::vector<> and mgb::SmallVector in the
 * C++ wrappers; neither do I want to write a SmallVector<> interface file as
 * good as swig's std::vector<> implementation.
 *
 * So the goal becomes making swig generate python wrapper for std::vector<>,
 * but call SymbolVarArray in the generated C++ file.
 *
 * A logical solution is to derive SymbolVarArray from std::vector<> only in
 * the .i file so swig can use the correct name; however the generated python
 * class becomes uniterable. So our hack here is to specialize std::vector to
 * use SymbolVarArray in the generated C++ file.
 *
 * This file must be included before instantiation of std::vector<SymbolVar>.
 */
%{
#include <vector>
#include "megbrain/graph/symbol_var.h"
using SymbolVar = mgb::cg::SymbolVar;
using SymbolVarArray = mgb::cg::SymbolVarArray;
namespace std {
template<typename alloc>
class vector<SymbolVar, alloc> : public SymbolVarArray {
public:
    using SymbolVarArray::SymbolVarArray;
    using allocator_type = alloc;

    allocator_type get_allocator() const {
        mgb_throw(mgb::MegBrainError, "get_allocator() should not be called");
        return {};
    }
};
}
%}

// vim: ft=swig
