#pragma once

#include "megbrain/graph.h"

#include <cmath>

using namespace mgb;

namespace mgb {
namespace test {
namespace loop {

/*!
 *\brief calc sin(x) = sum((-1)^k * x^(1+2k) / (1+2k)!, k >= 0)
 */
SymbolVar sin_by_taylor(SymbolVar x);

/*!
 *\brief calc exp(x) = sum(x^k / k!, k >= 0)
 */
SymbolVar exp_by_taylor(SymbolVar x);

}  // namespace loop
}  // namespace test
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
