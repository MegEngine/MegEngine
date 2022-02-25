#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megbrain/jit/compiler.h"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Module.h>

namespace mgb {
namespace jit {

/*!
 * \brief MLIR compiler
 */
class MLIRCompiler final : public Compiler {
    std::unique_ptr<Executable> do_compile(
            const InternalGraph& graph, const JITExecutor::Args& args) override;

public:
    MLIRCompiler(CompNode::DeviceType device_type = CompNode::DeviceType::CPU);
    Property property() const override {
        using F = Property::Flag;
        return Property{F::BIND_NDIM | F::BIND_SHAPE, JITFeatureBits::DIMSHUFFLE, 64};
    }

    size_t get_nr_workspace_outputs(JITExecutor* opr) const override;

    void init_workspace_size_infer(JITExecutor* opr) override;

private:
    void run_lowering_pass(mlir::OwningModuleRef& module, CompNode cn);

    CompNode::DeviceType m_device_type;
    static thread_local mlir::MLIRContext sm_ctx;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
