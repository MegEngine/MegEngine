/**
 * \file src/tensorrt/include/megbrain/tensorrt/tensorrt_runtime_opr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node_env.h"
#include "megbrain/graph.h"
#include "megbrain/serialization/opr_registry.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"

// some classes in NvInfer has no virtual dtor; so we ignore this warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <NvInfer.h>

namespace mgb {
namespace opr {


/*!
 * \brief an operator that evaluates a nvinfer::ICudaEngine object
 *
 * Input shapes and max batch size can not be changed.
 */
MGB_DEFINE_OPR_CLASS(TensorRTRuntimeOpr,
                           mgb::cg::SingleCNOutshapePureByInshapeOprBase) // {
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;

    void add_input_layout_constraint() override;
    void init_output_dtype() override;
    void scn_do_execute() override;

public:
    template <typename T>
    using TensorRTDeleter = intl::TensorRTDeleter<T>;
    template <typename T>
    using TensorRTUniquePtr = intl::TensorRTUniquePtr<T>;
    using GpuAllocator = TensorRTOpr::GpuAllocator;

    TensorRTRuntimeOpr(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                       std::shared_ptr<GpuAllocator> gpu_allocator,
                       const VarNodeArray& inputs,
                       const OperatorNodeConfig& config);

    //! get underlying TensorRT ICudaEngine object
    const std::shared_ptr<nvinfer1::ICudaEngine>& trt_cuda_engine() const {
        return m_engine;
    }

    //! serialization load/dump
    struct LoadDumpImpl;

    static SymbolVarArray make(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                               std::shared_ptr<GpuAllocator> gpu_allocator,
                               const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

    //! create an operator from a serialized ICudaEngine
    static SymbolVarArray make(const void* buf, size_t buf_size,
                               const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

    static std::shared_ptr<nvinfer1::ICudaEngine> to_shared_ptr_engine(
            nvinfer1::ICudaEngine* engine) {
        return {engine, TensorRTDeleter<nvinfer1::ICudaEngine>()};
    }

    //! get the underlying TensorRT IGpuAllocator used by the network
    const std::shared_ptr<GpuAllocator> trt_gpu_allocator() const {
        return m_gpu_allocator;
    }

private:
    // note: gpu allocator must be released after other trt objects
    std::shared_ptr<TensorRTOpr::GpuAllocator> m_gpu_allocator;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    intl::TensorRTManager m_manager;
    // if m_engine's dims with batch
    bool m_trt_engine_has_batch;
};  // namespace mgb

struct TensorRTRuntimeOpr::LoadDumpImpl {
    using Opr = opr::TensorRTRuntimeOpr;

    static void dump(serialization::OprDumpContext& ctx,
                     const cg::OperatorNodeBase& opr);

    static cg::OperatorNodeBase* load(serialization::OprLoadContext& ctx,
                                      const cg::VarNodeArray& inputs,
                                      const OperatorNodeConfig& config);
};

}  // namespace opr
}  // namespace mgb

#pragma GCC diagnostic pop

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
