/**
 * \file src/jit/impl/halide/compiler_cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./halide_header.h"

#if MGB_JIT_HALIDE && MGB_CUDA

#include "./halide_executable.h"
#include "megbrain/jit/compiler.h"

#include <cuda.h>

namespace mgb {
namespace jit {

class HalideCudaTargetTrait final : public HalideExecutable::TargetTrait {
public:
    FeatureSet features(CompNode comp_node) const override;
    FunctionHandle compile_and_load(CompNode comp_node, Halide::Target target,
                                    const HalideExecutable& hl_exec) override;
    void* get_user_context(CompNode comp_node) override;

private:
    struct UserData;
    struct HalideUserContext {
        CUcontext ctx;
        CUstream strm;
    };
    //! property for a single device
    struct DeviceProp {
        HalideUserContext ctx;
        int max_threads_per_block = -1;
    };
    CompNode::UnorderedMap<DeviceProp> m_cn2prop;
    std::mutex m_mtx;

    DeviceProp& get_dev_prop(CompNode comp_node);

    Halide::Pipeline gen_halide_pipeline_schedule(
            const ast_hl::AstNodePtr& dst_output,
            const DeviceProp& device_prop);
};

/*!
 * \brief Halide CUDA compiler
 */
class HalideCudaCompiler final : public Compiler {
    std::shared_ptr<HalideCudaTargetTrait> m_trait{
            std::make_shared<HalideCudaTargetTrait>()};

    std::unique_ptr<Executable> do_compile(
            const InternalGraph& graph, const JITExecutor::Args& args) override;

public:
    Property property() const override {
        using F = Property::Flag;
        return Property{F::BIND_NDIM | F::BIND_SHAPE | F::NEED_INPUT_CONTIG,
                        JITFeatureBits::REDUCE, 64};
    }

    size_t get_nr_workspace_outputs(JITExecutor*) const override { return 0; }

    void init_workspace_size_infer(JITExecutor*) override {}

    //! get object file name for cuda runtime override library
    static const std::string& cuda_runtime_lib();
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT_HALIDE && MGB_CUDA

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
