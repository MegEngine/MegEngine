#pragma once

#include "megbrain/gopt/framework.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"

namespace mgb {
namespace gopt {

class TensorRTReplacePass final : public Pass {
    class Impl;

public:
    const char* name() const override;
    void apply(OptState& opt) const override;
};

}  // namespace gopt

namespace tensorrt {

void transform_dest_vars_inplace(
        mgb::cg::VarNodeArray& dest_vars, cg::GraphCommonOptimizeOptions& options);
}

}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
