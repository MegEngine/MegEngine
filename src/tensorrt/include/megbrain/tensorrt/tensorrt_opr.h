/**
 * \file src/tensorrt/include/megbrain/tensorrt/tensorrt_opr.h
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

#if MGB_ENABLE_TENSOR_RT

// some classes in NvInfer has no virtual dtor; so we ignore this warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvInfer.h>

#define NV_TENSOR_RT_VERSION                                  \
    ((NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + \
     NV_TENSORRT_PATCH)  // major, minor, patch

namespace mgb {
namespace opr {

namespace intl {
enum class TensorRTGraphFeatureBits : uint32_t {
    NCHW_FLOAT = 0,
    NCHW4_QINT8 = 1,
};

template <typename T>
struct TensorRTDeleter {
    void operator()(T* p) {
        if (p != nullptr)
            p->destroy();
    }
};

template <typename T>
using TensorRTUniquePtr = std::unique_ptr<T, TensorRTDeleter<T>>;

class TensorRTManager {
    std::vector<void*> m_trt_iobuf;
    TensorRTUniquePtr<nvinfer1::IExecutionContext> m_context;
    void* m_device_workspace_memory_ptr;
    bool m_has_profiler;

public:
    void exec(cg::SingleCNOperatorNodeBase* opr, CompNode comp_node_check,
              nvinfer1::ICudaEngine* engine, size_t batch = 1);

    void clear_trt_context() { m_context.reset(); }

    //! number of items in the I/O buffer; used for testing
    size_t iobuf_size() const { return m_trt_iobuf.size(); }
};

static inline size_t workspace_size(nvinfer1::ICudaEngine* engine) {
    return engine->getDeviceMemorySize();
}
}  // namespace intl


/*!
 * \brief an operator that evaluates a nvinfer::INetworkDefinition object
 *
 * This operator allows input shapes to be changed.
 */
MGB_DEFINE_OPR_CLASS(TensorRTOpr,
                           mgb::cg::SingleCNOutshapePureByInshapeOprBase) // {
    void init_output_dtype() override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    
    void add_input_layout_constraint() override;
    
    void scn_do_execute() override;
    
    void set_input_by_tensor_shape(nvinfer1::ITensor* const input,
                               const TensorShape& tensor_shape) const;

public:
    template <typename T>
    using TensorRTDeleter = intl::TensorRTDeleter<T>;
    template <typename T>
    using TensorRTUniquePtr = intl::TensorRTUniquePtr<T>;
    using TensorRTGraphFeatureBits = intl::TensorRTGraphFeatureBits;

    //! TensorRT logger impl
    class Logger;
    //! TensorRT IGpuAllocator impl
    class GpuAllocator;

    //! sharing a network across builders is not recommended.
    //! use shared_ptr instead of unique_ptr for builder
    TensorRTOpr(std::shared_ptr<nvinfer1::IBuilder> builder,
                std::shared_ptr<nvinfer1::INetworkDefinition> network,
                TensorRTGraphFeatureBits feature_bits,
                std::shared_ptr<GpuAllocator> gpu_allocator,
                const VarNodeArray& inputs,
                std::shared_ptr<nvinfer1::ICudaEngine> engine,
                const OperatorNodeConfig& config);

    //! get underlying TensorRT IBuilder object
    const std::shared_ptr<nvinfer1::IBuilder>& trt_builder() const {
        return m_builder;
    }

    //! get underlying TensorRT INetworkDefinition object
    const std::shared_ptr<nvinfer1::INetworkDefinition>& trt_network_def()
            const {
        return m_network;
    }

    const std::shared_ptr<nvinfer1::ICudaEngine>& trt_cuda_engine() const {
        return m_engine;
    }

    //! get the underlying TensorRT IGpuAllocator used by the network
    const std::shared_ptr<GpuAllocator> trt_gpu_allocator() const {
        return m_gpu_allocator;
    }

    TensorRTGraphFeatureBits trt_graph_feature_bits() const {
        return m_feature_bits;
    }

    static SymbolVarArray make(
            std::shared_ptr<nvinfer1::IBuilder> builder,
            std::shared_ptr<nvinfer1::INetworkDefinition> network,
            TensorRTGraphFeatureBits feature_bits,
            std::shared_ptr<GpuAllocator> gpu_allocator,
            const SymbolVarArray& src,
            std::shared_ptr<nvinfer1::ICudaEngine> engine =
                    {nullptr, TensorRTDeleter<nvinfer1::ICudaEngine>()},
            const OperatorNodeConfig& config = {});

    static std::shared_ptr<nvinfer1::INetworkDefinition> to_shared_ptr_network(
            nvinfer1::INetworkDefinition* network) {
        return {network, TensorRTDeleter<nvinfer1::INetworkDefinition>()};
    }

    static std::shared_ptr<nvinfer1::IBuilder> to_shared_ptr_builder(
            nvinfer1::IBuilder* builder) {
        return {builder, TensorRTDeleter<nvinfer1::IBuilder>()};
    }

    //! convert TensorRT Dims to mgb TensorShape
    static TensorShape dims2shape(const nvinfer1::Dims& dims, size_t batch = 0);

    //! get underlying TensorRTManager; for debug
    const intl::TensorRTManager& trt_manager() const { return m_manager; }

    //! build cuda engine from cache
    void build_engine_from_cache();

    //! serialize engine to cache
    void serialize_engine_to_cache() const;

private:
    // note: gpu allocator must be released after other trt objects
    std::shared_ptr<GpuAllocator> m_gpu_allocator;
    std::shared_ptr<nvinfer1::INetworkDefinition> m_network;
    std::shared_ptr<nvinfer1::IBuilder> m_builder;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
#if NV_TENSOR_RT_VERSION >= 6001
    TensorRTUniquePtr<nvinfer1::IBuilderConfig> m_builder_config;
#endif
    intl::TensorRTManager m_manager;
    TensorRTGraphFeatureBits m_feature_bits;
};

class TensorRTOpr::Logger final : public nvinfer1::ILogger, NonCopyableObj {
    Logger();

public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override;
    static Logger& instance();
};

class TensorRTOpr::GpuAllocator final : public nvinfer1::IGpuAllocator {
    CompNode m_cn;
    std::mutex m_ptr2size_mtx;
    ThinHashMap<void*, size_t> m_ptr2size;

public:
    explicit GpuAllocator(CompNode cn);
    ~GpuAllocator() noexcept;

    void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) override;
    void free(void* memory) override;

    CompNode comp_node() const { return m_cn; }
};


}  // namespace opr
}  // namespace mgb

#pragma GCC diagnostic pop

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
