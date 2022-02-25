#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include <vector>

#include "./common.h"

namespace mgb {
namespace jit {

/*! polynomial of degree N:
 *  C_0 + C_1 * x + C_2 * x^2 + ... + C_N * x^N
 *  where coeff = [C_N, ..., C_2, C_1, C_0]
 */
mlir::Value polynomial(
        ValueBuilderHelper& helper, mlir::Value x, std::vector<mlir::Value>& coeff);

//! numerical approximation of arctangent
mlir::Value atan2_approx(ValueBuilderHelper& helper, mlir::Value y, mlir::Value x);

//! numerical approximation of gauss error function
mlir::Value erf_approx(ValueBuilderHelper& helper, mlir::Value x);

//! numerical approximation of the inverse of normal distribution function
mlir::Value ndtri_approx(ValueBuilderHelper& helper, mlir::Value x);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
