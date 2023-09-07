#include "./mlir/compiler.h"
#include "./halide/compiler_cuda.h"
#include "./nvrtc/compiler_cuda.h"

#include "megbrain/jit/compiler.h"
#include "megbrain/utils/hash.h"

#if MGB_JIT

using namespace mgb;
using namespace jit;

namespace {
class CompilerHolder final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

public:
    std::mutex mtx;
    ThinHashMap<CompNode::DeviceType, std::unique_ptr<Compiler>> dev2compiler;
};
MGB_TYPEINFO_OBJ_IMPL(CompilerHolder);

}  // anonymous namespace

class Compiler::EmptyCompiler final : public Compiler {
public:
    Property property() const {
        return {Property::Flag::NONE, JITFeatureBits::NONE, 100};
    }

    size_t get_nr_workspace_outputs(JITExecutor*) const { return 0; }

    void init_workspace_size_infer(JITExecutor*) {}

    std::unique_ptr<Executable> do_compile(
            const InternalGraph&, const JITExecutor::Args&) {
        mgb_throw(InternalError, "EmptyCompiler should not be used");
    }
};

bool Compiler::is_supported_device(CompNode::DeviceType device) {
    switch (device) {
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            return true;
#endif
        case CompNode::DeviceType::CPU:
            return true;
        default:
            return false;
    }
}

Compiler* Compiler::get(ComputingGraph& graph, CompNode comp_node) {
    static EmptyCompiler empty_compiler;
    if (comp_node == CompNode::default_cpu()) {
        // oprs in the internal graph are on default cpu; this case handles
        // nested JITExecutor
        return &empty_compiler;
    }

    CompilerHolder* holder;
    {
        static std::mutex mtx;
        MGB_LOCK_GUARD(mtx);
        holder = graph.options().user_data.get_user_data_or_create<CompilerHolder>();
    }
    MGB_LOCK_GUARD(holder->mtx);
    auto&& compiler = holder->dev2compiler[comp_node.device_type()];
    std::string backend = gopt::JITFusionPass::get_jit_backend_str();
    mgb_assert(
            !backend.empty(),
            "code issue happened, need call config_jit_backends before get compiler");
    //! please keep logic with JITFusionPass::Impl::config_jit_backends
    mgb_log_debug("Compiler: JIT backend: %s", backend.c_str());
    if (!compiler) {
        switch (comp_node.device_type()) {
#if MGB_CUDA
            case CompNode::DeviceType::CUDA:
#if MGB_JIT_HALIDE
                if (!strcmp(backend.c_str(), "HALIDE")) {
                    compiler = std::make_unique<HalideCudaCompiler>();
                    break;
                }
#endif
#if MGB_JIT_MLIR
                if (!strcmp(backend.c_str(), "MLIR")) {
                    compiler =
                            std::make_unique<MLIRCompiler>(CompNode::DeviceType::CUDA);
                    break;
                }
#endif
                if (!strcmp(backend.c_str(), "NVRTC")) {
                    compiler = std::make_unique<CudaCompiler>();
                    break;
                }
                mgb_throw(
                        InternalError,
                        "No compiler support for cuda, may caused by build not enable "
                        "MLIR/HALIDE module or error config jit backend env");
                break;
#endif
            case CompNode::DeviceType::CPU:
#if MGB_JIT_MLIR
                if (!strcmp(backend.c_str(), "MLIR")) {
                    compiler =
                            std::make_unique<MLIRCompiler>(CompNode::DeviceType::CPU);
                    break;
                }
#endif
                mgb_throw(
                        InternalError,
                        "No compiler support for cpu, may caused by build not enable "
                        "MLIR module or error config jit backend env");
                break;
            default:
                mgb_throw(
                        InternalError,
                        "unsupported JIT config: "
                        "comp_node=%s backend_setting=%s",
                        comp_node.to_string().c_str(), backend.c_str());
        }
    }

    return compiler.get();
}

Executable* Compiler::compile(JITExecutor* opr) {
    MGB_LOCK_GUARD(m_mtx);
    auto&& args = opr->args();
    auto&& args_cache = m_expr_cache[&(opr->internal_graph())];
    auto q = args_cache.get(args);
    if (q.first) {
        *q.second = do_compile(opr->internal_graph(), opr->args());
    }
    return q.second->get();
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
