/**
 * \file src/jit/impl/nvrtc/compiler_cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#if MGB_JIT && MGB_CUDA

#include <cuda.h>
#include <nvrtc.h>
#include "megbrain/jit/compiler.h"

#define MGB_NVRTC_CHECK(expr)                                            \
    do {                                                                 \
        nvrtcResult __nvrtc_result = (expr);                             \
        if (!mgb_likely(__nvrtc_result == NVRTC_SUCCESS)) {              \
            ::mgb::jit::_on_nvrtc_error(#expr, __nvrtc_result, __FILE__, \
                                        __func__, __LINE__);             \
        }                                                                \
    } while (0)

namespace mgb {
namespace jit {

[[noreturn]] void _on_nvrtc_error(const char* expr, nvrtcResult nvrtc_res,
                                  const char* file, const char* func, int line);

/*!
 * \brief Executable class for CUDA
 */
class CudaExecutable final : public Executable {
public:
    CudaExecutable(std::string source, std::string name);
    ~CudaExecutable();

    /*!
     * \brief execute
     * A Executable instance can be executed by one or more fusion_opr
     */
    void execute(JITExecutor* fusion_opr) override final;

private:
    //! cache for a func on a specific device
    struct FuncCache {
        struct Func {
            int block_size{-1};
            CUmodule module{nullptr};
            CUfunction func{nullptr};
        };

        std::mutex mtx;
        std::string ptx;
        CompNode::UnorderedMap<Func> cn2func;

        void compile(const std::string& cache_category, int major, int minor,
                     const CudaExecutable* cuda_exe);
        void exec(const JITExecutor* fusion_opr,
                  const CudaExecutable* cuda_exe);
    };

    const std::string m_source;
    const std::string m_name;
    std::mutex m_mtx;
    //! (cuda_major, cuda_minor) => func
    ThinHashMap<std::pair<uint32_t, uint32_t>, FuncCache> m_func_cache;
};

/*!
 * \brief CUDA compiler using NVRTC
 */
class CudaCompiler final : public Compiler {
    std::unique_ptr<Executable> do_compile(
            const InternalGraph& graph, const JITExecutor::Args& args) override;

public:
    /*!
     *  \brief should limit the input size of JIT fusion because the largest
     *  parameter size of a kernel function is 4096 bytes.
     *
     *  parameter size = (1 + nr_inps) * 8 + 8 + nr_inps *
     *  sizeof(ParamElemVisitor) + 4 + 4
     */
    static constexpr size_t MAX_CUDA_NR_INPUT = 38;

    Property property() const override {
        using F = Property::Flag;
        return Property{F::NEED_INPUT_COLLAPSE | F::BIND_NDIM,
                        JITFeatureBits::NONE, 64};
    }

    size_t get_nr_workspace_outputs(JITExecutor* opr) const override;

    void init_workspace_size_infer(JITExecutor* opr) override;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_CUDA

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
