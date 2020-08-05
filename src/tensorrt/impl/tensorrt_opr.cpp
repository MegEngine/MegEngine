/**
 * \file src/tensorrt/impl/tensorrt_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/tensorrt/tensorrt_engine_cache.h"
#include "megbrain/common.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/version_symbol.h"
#include "megbrain/utils/timer.h"

#include <cinttypes>

#if MGB_ENABLE_TENSOR_RT

using namespace mgb;
using namespace opr;
using TensorRTManager = intl::TensorRTManager;

namespace {

#if MGB_ENABLE_JSON
class TensorRTProfiler : public nvinfer1::IProfiler {
public:
    typedef std::pair<std::string, float> Record;
    std::vector<Record> profile;

    void reportLayerTime(const char* layerName, float ms) override;
    void print_layer_times();
    std::shared_ptr<json::Value> to_json();
};

void TensorRTProfiler::reportLayerTime(const char* layerName, float ms) {
    profile.push_back(std::make_pair(layerName, ms));
}

void TensorRTProfiler::print_layer_times() {
    float total_time = 0;
    for (size_t i = 0; i < profile.size(); ++i) {
        printf("%s %4.3fms\n", profile[i].first.c_str(), profile[i].second);
        total_time += profile[i].second;
    }
    printf("Total time: %4.3fms\n", total_time);
}

std::shared_ptr<json::Value> TensorRTProfiler::to_json() {
    using namespace json;
    auto prof_arr = Array::make();
    for (auto&& rec : profile) {
        auto&& item = Array::make();
        item->add(String::make(rec.first));
        item->add(Number::make(rec.second));
        prof_arr->add(item);
    }
    return prof_arr;
}
#endif  // MGB_ENABLE_JSON


}  // anonymous namespace

/* ========================== Logger ========================== */

void TensorRTOpr::Logger::log(nvinfer1::ILogger::Severity severity,
                              const char* msg) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            mgb_log("TRT_INTERNAL_ERROR: %s", msg);
            return;
        case Severity::kERROR:
            mgb_log("TRT_ERROR: %s", msg);
            return;
        case Severity::kWARNING:
            mgb_log("TRT_WARNING: %s", msg);
            return;
        case Severity::kINFO:
            mgb_log_debug("TRT_INFO: %s", msg);
            return;
#if NV_TENSOR_RT_VERSION >= 6001
        case Severity::kVERBOSE:
            mgb_log_debug("TRT_VERBOSE: %s", msg);
            return;
#endif
        default:
            mgb_log_debug("TRT_UNKNOWN: %s", msg);
            return;
    }
}

TensorRTOpr::Logger::Logger() {
    int expect = NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 +
                 NV_TENSORRT_PATCH,
        got = getInferLibVersion();
    mgb_log("loaded TensorRT version: %d", got);
    mgb_assert(expect <= got,
               "TensorRT library is older than mgb compiled version: got=%d "
               "compiled_with=%d",
               got, expect);
    if (expect != got) {
        mgb_log_warn(
                "MegBrain is compiled with TensorRT %d but get %d at runtime",
                expect, got);
    }
}

TensorRTOpr::Logger& TensorRTOpr::Logger::instance() {
    static Logger logger;
    return logger;
}

/* ========================== GpuAllocator ========================== */

TensorRTOpr::GpuAllocator::GpuAllocator(CompNode cn) : m_cn{cn} {
    mgb_assert(cn.device_type() == CompNode::DeviceType::CUDA,
               "can not use GPU allocator on comp node %s",
               cn.to_string().c_str());
}

TensorRTOpr::GpuAllocator::~GpuAllocator() noexcept {
    MGB_LOCK_GUARD(m_ptr2size_mtx);
    if (!m_ptr2size.empty()) {
        std::string msg{"there are unreleased TRT mem buffers:\n"};
        for (auto&& i : m_ptr2size) {
            msg.append(ssprintf("  %p: %zu\n", i.first, i.second));
        }
        mgb_log_error("%sabort now", msg.c_str());
        mgb_trap();
    }
}

void* TensorRTOpr::GpuAllocator::allocate(uint64_t size, uint64_t alignment,
                                          uint32_t flags) {
    static bool enable_log = getenv("MGB_LOG_TRT_MEM_ALLOC");
    mgb_assert(!flags && !(alignment & (alignment - 1)),
               "flags=%u alignment=%" PRIu64, flags, alignment);
    auto ret = m_cn.alloc_device(size);
    mgb_assert(!(reinterpret_cast<uintptr_t>(ret) & (alignment - 1)),
               "ptr=%p alignment=%" PRIu64, ret, alignment);
    if (enable_log) {
        mgb_log("trt mem alloc on %s: size=%" PRIu64 " align=%" PRIu64
                " ptr=%p",
                m_cn.to_string().c_str(), size, alignment, ret);
    }
    {
        MGB_LOCK_GUARD(m_ptr2size_mtx);
        m_ptr2size[ret] = size;
    }
    return ret;
}

void TensorRTOpr::GpuAllocator::free(void* memory) {
    {
        auto iter = m_ptr2size.find(memory);
        mgb_assert(iter != m_ptr2size.end(), "ptr %p not found", memory);
        m_ptr2size.erase(iter);
    }
    m_cn.free_device(memory);
}

/* ========================== TensorRTManager ========================== */
void TensorRTManager::exec(cg::SingleCNOperatorNodeBase* opr,
                           CompNode comp_node_check,
                           nvinfer1::ICudaEngine* engine,
                           size_t batch) {

    auto comp_node = opr->comp_node();
    // ICudaEngine is bound to the currently active device
    comp_node.activate();

    if (comp_node_check.valid()) {
        mgb_assert(comp_node_check == comp_node,
                   "gpu allocator is on %s, but execution is on %s",
                   comp_node_check.to_string().c_str(),
                   comp_node.to_string().c_str());
    }
#if MGB_ENABLE_JSON
    auto pf_holder_pair =
            opr->owner_graph()
                    ->options()
                    .user_data.get_user_data<opr_profile::OprProfileHolder>();
    if (m_has_profiler && !pf_holder_pair.second) {
        m_context.reset();
        m_has_profiler = false;
    }
#endif
    auto workspace_ptr = opr->output().back()->dev_tensor().raw_ptr();
    bool should_reinit_device_memory =
            !m_context || m_device_workspace_memory_ptr != workspace_ptr;
    if (!m_context) {
        m_context = {engine->createExecutionContextWithoutDeviceMemory(), {}};
        m_has_profiler = false;
    }
    m_trt_iobuf.resize(opr->input().size() + opr->output().size() - 1);
    bool is_trt_opr = false;
    if (opr->same_type<TensorRTOpr>()) {
        is_trt_opr = true;
        auto network = opr->cast_final_safe<TensorRTOpr>().trt_network_def();
        int nr_input = network->getNbInputs();
        for (int i = 0; i < nr_input; ++i) {
            int binding_idx =
                    engine->getBindingIndex(network->getInput(i)->getName());
            m_trt_iobuf[binding_idx] = opr->input(i)->dev_tensor().raw_ptr();
        }
        int nr_output = network->getNbOutputs();
        for (int i = 0; i < nr_output; ++i) {
            int binding_idx =
                    engine->getBindingIndex(network->getOutput(i)->getName());
            m_trt_iobuf[binding_idx] = opr->output(i)->dev_tensor().raw_ptr();
        }
    } else {
        for (size_t i = 0; i < opr->input().size(); ++i) {
            m_trt_iobuf[i] = opr->input(i)->dev_tensor().raw_ptr();
        }
        for (size_t i = 0; i < opr->output().size() - 1; ++i) {
            m_trt_iobuf[opr->input().size() + i] =
                    opr->output(i)->dev_tensor().raw_ptr();
        }
    }
    MGB_MARK_USED_VAR(is_trt_opr);
    if (should_reinit_device_memory) {
        mgb_assert(opr->output().back()->shape()[0] ==
                           intl::workspace_size(engine) &&
                   !(reinterpret_cast<uintptr_t>(workspace_ptr) % 256));
        m_context->setDeviceMemory(workspace_ptr);
        m_device_workspace_memory_ptr = workspace_ptr;
    }
    auto&& env = mgb::CompNodeEnv::from_comp_node(comp_node);

    bool exec_success = false;

#if MGB_ENABLE_JSON
    if (!pf_holder_pair.second) {
        mgb_assert(!m_has_profiler,
                   "Invalid state of TensorRTRuntimeOpr: should not have "
                   "profiler.");
#if NV_TENSOR_RT_VERSION >= 6001
        if (is_trt_opr)
            exec_success = m_context->enqueueV2(m_trt_iobuf.data(),
                                                env.cuda_env().stream, nullptr);
        else
            exec_success = m_context->enqueue(batch, m_trt_iobuf.data(),
                                              env.cuda_env().stream, nullptr);
#else
        exec_success = m_context->enqueue(batch, m_trt_iobuf.data(),
                                          env.cuda_env().stream, nullptr);
#endif
        mgb_assert(exec_success, "TensorRTOpr failed in execution.");
    } else {
        TensorRTProfiler trt_profiler;
        m_context->setProfiler(&trt_profiler);
        m_has_profiler = true;
        // TensorRT documentation stated that IExecutionContext->execute
        // "Synchronously execute inference on a batch", and it does not take a
        // cudaStream_t, we expect it do a device synchronize. But it seems like
        // what it really does is execute and sync on its own stream instead of
        // synchronize entire device, execute then synchronize again. So we have
        // to synchronize before execution to make profiling accurate.
        comp_node.sync();
#if NV_TENSOR_RT_VERSION >= 6001
        if (is_trt_opr)
            exec_success = m_context->executeV2(m_trt_iobuf.data());
        else
            exec_success = m_context->execute(batch, m_trt_iobuf.data());
#else
        exec_success = m_context->execute(batch, m_trt_iobuf.data());
#endif
        mgb_assert(exec_success, "trt execution failed: opr=%s", opr->cname());
        pf_holder_pair.first[0]->id2object_map[opr] = trt_profiler.to_json();
        printf("TRT profile info of opr %s:\n", opr->name().c_str());
        trt_profiler.print_layer_times();
    }
#else
#if NV_TENSOR_RT_VERSION >= 6001
    if (is_trt_opr)
        exec_success = m_context->enqueueV2(m_trt_iobuf.data(),
                                            env.cuda_env().stream, nullptr);
    else
        exec_success = m_context->enqueue(batch, m_trt_iobuf.data(),
                                          env.cuda_env().stream, nullptr);
#else
    exec_success = m_context->enqueue(batch, m_trt_iobuf.data(),
                                      env.cuda_env().stream, nullptr);
#endif
    mgb_assert(exec_success, "trt execution failed: opr=%s", opr->cname());
#endif
}

/* ========================== TensorRTOpr ========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TensorRTOpr);
TensorRTOpr::TensorRTOpr(std::shared_ptr<nvinfer1::IBuilder> builder,
                         std::shared_ptr<nvinfer1::INetworkDefinition> network,
                         TensorRTGraphFeatureBits feature_bits,
                         std::shared_ptr<GpuAllocator> gpu_allocator,
                         const VarNodeArray& inputs,
                         std::shared_ptr<nvinfer1::ICudaEngine> engine,
                         const OperatorNodeConfig& config)
        : Super(inputs.at(0)->owner_graph(), config, "tensor_rt",
                {inputs.at(0)}),
          m_gpu_allocator{std::move(gpu_allocator)},
          m_network{std::move(network)},
          m_builder{std::move(builder)},
          m_engine{std::move(engine)},
          m_feature_bits{feature_bits} {
    mgb_assert(
            inputs[0]->comp_node().device_type() == CompNode::DeviceType::CUDA,
            "TensorRTOpr can only be used on cuda comp nodes; got %s",
            inputs[0]->comp_node().to_string().c_str());
    mgb_assert(inputs.size() == static_cast<size_t>(m_network->getNbInputs()),
               "inputs size not equal: expect=%zu got=%d", inputs.size(),
               m_network->getNbInputs());
    for (auto i : inputs) {
        add_input({i});
    }
    if (m_network->getNbOutputs() == 1)
        add_output(None);
    else {
        for (int i = 0; i < m_network->getNbOutputs(); ++i)
            add_output(ssprintf("o%d", i));
    }
    cg::add_workspace_output(this);

    add_equivalence_component<mgb::ScalarHash<void*>>(m_network.get());
    mgb_assert(m_builder != nullptr);
#if NV_TENSOR_RT_VERSION >= 6001
    m_builder_config = {m_builder->createBuilderConfig(),
                        TensorRTDeleter<nvinfer1::IBuilderConfig>()};
    m_builder_config->setMaxWorkspaceSize(1 << 30);
    if (m_feature_bits == TensorRTGraphFeatureBits::NCHW4_QINT8) {
        mgb_assert(m_builder->platformHasFastInt8(),
                   "Cuda platform does not support fast native int8");
        m_builder_config->setInt8Calibrator(nullptr);
        nvinfer1::BuilderFlags flags;
        flags = 1 << static_cast<int>(nvinfer1::BuilderFlag::kINT8);
        m_builder_config->setFlags(flags);
    }
#else
    m_builder->setMaxWorkspaceSize(1 << 30);
    if (m_feature_bits == TensorRTGraphFeatureBits::NCHW4_QINT8) {
        // check has fast int8
        m_builder->setInt8Mode(true);
        m_builder->setInt8Calibrator(nullptr);
        m_builder->setStrictTypeConstraints(false);
    }
#endif
    if (!m_gpu_allocator) {
        m_gpu_allocator =
                std::make_shared<GpuAllocator>(inputs[0]->comp_node());
    }
    m_builder->setGpuAllocator(m_gpu_allocator.get());
}

SymbolVarArray TensorRTOpr::make(
        std::shared_ptr<nvinfer1::IBuilder> builder,
        std::shared_ptr<nvinfer1::INetworkDefinition> network,
        TensorRTGraphFeatureBits feature_bits,
        std::shared_ptr<GpuAllocator> gpu_allocator, const SymbolVarArray& src,
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto tensor_rt_opr = std::make_unique<TensorRTOpr>(
            std::move(builder), std::move(network), feature_bits,
            std::move(gpu_allocator), var_node_array, std::move(engine),
            config);
    auto ret = cg::to_symbol_var_array(
            src[0].node()
                    ->owner_graph()
                    ->insert_opr(std::move(tensor_rt_opr))
                    ->output());
    ret.pop_back();  // remove workspace
    return ret;
}

TensorShape TensorRTOpr::dims2shape(const nvinfer1::Dims& dims, size_t batch) {
    TensorShape ret;
    ret.ndim = dims.nbDims;
    if (batch > 0)
        ++ret.ndim;
    mgb_assert(ret.ndim <= TensorShape::MAX_NDIM,
               "TensorShape ndim > MAX_NDIM");
    if (batch > 0) {
        ret[0] = batch;
        for (size_t i = 1; i < ret.ndim; ++i) {
            ret[i] = dims.d[i-1];
        }
    } else {
        for (size_t i = 0; i < ret.ndim; ++i) {
            ret[i] = dims.d[i];
        }
    }
    return ret;
}

void TensorRTOpr::set_input_by_tensor_shape(
        nvinfer1::ITensor* const input, const TensorShape& tensor_shape) const {
    nvinfer1::Dims dims = input->getDimensions();
#if NV_TENSOR_RT_VERSION >= 6001
    auto tensor_format = input->getAllowedFormats();
    if (tensor_format &
        (1 << static_cast<int>(nvinfer1::TensorFormat::kCHW4))) {
        mgb_assert(dims.nbDims == 4 && tensor_shape.ndim == 5 &&
                           tensor_shape[4] == 4,
                   "input tensor format need to be NCHW4(got: %s)",
                   tensor_shape.to_string().c_str());
        for (int i = 0; i < dims.nbDims; i++) {
            dims.d[i] = tensor_shape.shape[i];
        }
        dims.d[1] *= 4;
    } else {
        mgb_assert(tensor_format &
                   (1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR)));
        mgb_assert(static_cast<int>(tensor_shape.ndim) == dims.nbDims,
                   "input dim is not qual to which in trt network created");
        for (size_t i = 0; i < tensor_shape.ndim; i++) {
            dims.d[i] = tensor_shape.shape[i];
        }
    }
#else
    mgb_assert(static_cast<int>(tensor_shape.ndim) == dims.nbDims,
               "input dim is not qual to which in trt network created");
    for (size_t i = 0; i < tensor_shape.ndim; i++) {
        dims.d[i] = tensor_shape.shape[i];
    }
#endif
    input->setDimensions(dims);
}

void TensorRTOpr::init_output_dtype() {
    auto get_mgb_dtype_from_itensor = [](nvinfer1::ITensor* tensor) -> DType {
        switch (tensor->getType()) {
            case nvinfer1::DataType::kFLOAT:
                return dtype::Float32();
            case nvinfer1::DataType::kHALF:
                return dtype::Float16();
            case nvinfer1::DataType::kINT8: {
#if NV_TENSOR_RT_VERSION >= 5020
#if NV_TENSOR_RT_VERSION >= 5120
                auto range_max = tensor->getDynamicRangeMax(),
                     range_min = tensor->getDynamicRangeMin();
                auto range = std::max(range_max, range_min);
#else
                auto range = tensor->getDynamicRange();
#endif
                mgb_assert(range >= 0,
                           "trt dynamic range should be non-negative");
                static constexpr int8_t i_max =
                        std::numeric_limits<int8_t>().max();
                float scale =
                        static_cast<float>(range) / static_cast<float>(i_max);
                return dtype::QuantizedS8{scale};
#else
                return dtype::Int8();
#endif
            }
            case nvinfer1::DataType::kINT32:
                return dtype::Int32();
            default:
                mgb_throw(InternalError,
                          "trt DataType should be kFLOAT/kHALF/kINT8/kINT32.");
        }
    };
    for (int i = 0; i < m_network->getNbOutputs(); ++i) {
        output(i)->dtype(get_mgb_dtype_from_itensor(m_network->getOutput(i)));
    }
}

void TensorRTOpr::get_output_var_shape(const TensorShapeArray& inp_shape,
                                       TensorShapeArray& out_shape) const {
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        set_input_by_tensor_shape(m_network->getInput(i), inp_shape[i]);
    }

    for (int i = 0; i < m_network->getNbOutputs(); ++i) {
#if NV_TENSOR_RT_VERSION >= 6001
        auto output = m_network->getOutput(i);
        out_shape[i] = dims2shape(output->getDimensions());
        auto tensor_format = output->getAllowedFormats();
        // fix tensor shape from tensor format
        if (tensor_format &
            (1 << static_cast<int>(nvinfer1::TensorFormat::kCHW4))) {
            mgb_assert(out_shape[i].ndim == 4);
            out_shape[i].ndim++;
            out_shape[i].shape[1] /= 4;
            out_shape[i].shape[4] = 4;
        }
#else
        out_shape[i] = dims2shape(m_network->getOutput(i)->getDimensions());
#endif
    }

    // Because input shape is NCHW, so the batch size should always be 1.
    m_builder->setMaxBatchSize(1);

    auto self = const_cast<TensorRTOpr*>(this);
    if (m_engine == nullptr && TensorRTEngineCache::enable_engine_cache()) {
        self->build_engine_from_cache();
    }

    bool engine_valid = true;
    if (m_engine == nullptr) {
        engine_valid = false;
    } else {
        int nr_input = m_network->getNbInputs();
        mgb_assert(static_cast<size_t>(nr_input) == input().size(),
                   "input size changed");
        for (int i = 0; i < nr_input; ++i) {
            int binding_idx = m_engine->getBindingIndex(
                    m_network->getInput(i)->getName());
            auto cuda_engine_shp =
                    dims2shape(m_engine->getBindingDimensions(binding_idx));
#if NV_TENSOR_RT_VERSION >= 6001
            auto tensor_format = m_engine->getBindingFormat(binding_idx);
            // fix tensor shape from tensor format
            if (tensor_format == nvinfer1::TensorFormat::kCHW4) {
                mgb_assert(cuda_engine_shp.ndim == 4);
                cuda_engine_shp.ndim++;
                cuda_engine_shp[1] /= 4;
                cuda_engine_shp[4] = 4;
            }
#endif
            if (!cuda_engine_shp.eq_shape(inp_shape[i])) {
                engine_valid = false;
                break;
            }
        }
    }

    if (!engine_valid) {
        comp_node().activate();
        // If a context created by a cuda engine, the context must be destroyed
        // before the corresponding cuda engine. Otherwise, a segmentfault will
        // occur.
        self->m_manager.clear_trt_context();
        RealTimer timer;
#if NV_TENSOR_RT_VERSION >= 6001
        self->m_engine = {
                m_builder->buildEngineWithConfig(*m_network, *m_builder_config),
                TensorRTDeleter<nvinfer1::ICudaEngine>()};
#else
        self->m_engine = {m_builder->buildCudaEngine(*m_network),
                          TensorRTDeleter<nvinfer1::ICudaEngine>()};
#endif
        mgb_assert(m_engine != nullptr, "build engine failed");
        mgb_log_warn("TensorRTOpr(name:%s) engine build time %.2f ms", cname(),
                     timer.get_msecs());

        if (TensorRTEngineCache::enable_engine_cache()) {
            serialize_engine_to_cache();
        }
    }

    out_shape.back() = {intl::workspace_size(m_engine.get())};
}

void TensorRTOpr::add_input_layout_constraint() {
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void TensorRTOpr::scn_do_execute() {
    m_manager.exec(this, m_gpu_allocator->comp_node(), m_engine.get());
}

void TensorRTOpr::build_engine_from_cache() {
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(TensorRTOpr::Logger::instance()), {}};
    runtime->setGpuAllocator(m_gpu_allocator.get());
    auto ret = TensorRTEngineCache::inst().get(
            TensorRTEngineCache::make_key_from_trt_opr(this));
    if (!ret.valid())
        return;
    comp_node().activate();
    auto engine = runtime->deserializeCudaEngine(
            reinterpret_cast<const void*>(ret->ptr), ret->size, nullptr);
    mgb_assert(engine, "failed to deserialize ICudaEngine");
    m_engine = {engine, TensorRTDeleter<nvinfer1::ICudaEngine>()};
}

void TensorRTOpr::serialize_engine_to_cache() const {
    TensorRTUniquePtr<nvinfer1::IHostMemory> buf{trt_cuda_engine()->serialize(),
                                                 {}};
    mgb_assert(buf, "failed to serialize ICudaEngine");
    TensorRTEngineCache::inst().put(
            TensorRTEngineCache::make_key_from_trt_opr(this),
            {buf->data(), buf->size()});
}

MGB_VERSION_SYMBOL3(TENSORRT, NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                    NV_TENSORRT_PATCH);

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
