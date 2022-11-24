#pragma once

#include "megbrain_build_config.h"
#if MGB_OPENCL

#include "megbrain/jit/compiler.h"

namespace mgb {
namespace jit {

/*!
 * \brief Executable class for OPENCL
 */
class OpenCLExecutable final : public Executable {
public:
    OpenCLExecutable(std::string source, std::string name, bool is_debug);
    ~OpenCLExecutable() = default;

    /*!
     * \brief execute
     * A Executable instance can be executed by one or more fusion_opr
     */
    void execute(JITExecutor* fusion_opr) override final;

private:
    const std::string m_source;
    const std::string m_name;
    bool m_is_debug;
};

/*!
 * \brief OpenCL tiny compiler, now only handle elemwise opr and just call DNN CL runtime
 */
class OpenCLTinyCompiler final : public Compiler {
    std::unique_ptr<Executable> do_compile(
            const InternalGraph& graph, const JITExecutor::Args& args) override;
    bool m_is_debug;

public:
    OpenCLTinyCompiler(CompNode::DeviceType device_type = CompNode::DeviceType::OPENCL);
    Property property() const override {
        using F = Property::Flag;
        return Property{F::BIND_NDIM | F::BIND_SHAPE, JITFeatureBits::NONE, 64};
    }

    size_t get_nr_workspace_outputs(JITExecutor* opr) const override;

    void init_workspace_size_infer(JITExecutor* opr) override;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_OPENCL
