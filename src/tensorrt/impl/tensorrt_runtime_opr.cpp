/**
 * \file src/tensorrt/impl/tensorrt_runtime_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/tensorrt/tensorrt_runtime_opr.h"
#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/common.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/version_symbol.h"
#include "megdnn/basic_types.h"

#include <cinttypes>

#if MGB_ENABLE_TENSOR_RT

using namespace mgb;
using namespace opr;
using TensorRTManager = intl::TensorRTManager;

namespace {

DType get_dtype_from_trt(nvinfer1::DataType trt_dtype) {
    switch (trt_dtype) {
        case nvinfer1::DataType::kFLOAT:
            return dtype::Float32();
        case nvinfer1::DataType::kHALF:
#if !MEGDNN_DISABLE_FLOAT16
            return dtype::Float16();
#else
            mgb_throw(MegBrainError, "Float16 support is disabled.");
#endif
        // We cannot get scale of an Tensor from tensorrt Engine, so the scale
        // here is not correct. When researchers build TensorRT engine, they
        // should make sure the scale of quantized int8 tensors in MegBrain
        // matches with dynamic ranges of TensorRT tensors
        case nvinfer1::DataType::kINT8:
            return dtype::QuantizedS8(1.f);
        case nvinfer1::DataType::kINT32:
            return dtype::Int32();
        default:
            mgb_assert("DataType of trt engine is unknown.");
    }
    return DType();
}

}  // anonymous namespace


/* ========================== TensorRTRuntimeOpr ========================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TensorRTRuntimeOpr);
TensorRTRuntimeOpr::TensorRTRuntimeOpr(
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        std::shared_ptr<GpuAllocator> gpu_allocator, const VarNodeArray& inputs,
        const OperatorNodeConfig& config)
        : Super(inputs.at(0)->owner_graph(), config, "tensor_rt",
                {inputs.at(0)}),
          m_gpu_allocator{std::move(gpu_allocator)},
          m_engine{std::move(engine)},
          m_trt_engine_has_batch{false} {
    mgb_assert(
            inputs[0]->comp_node().device_type() == CompNode::DeviceType::CUDA,
            "TensorRTRuntimeOpr can only be used on cuda comp nodes; got %s",
            inputs[0]->comp_node().to_string().c_str());
    size_t nr_input = 0;
    bool is_input = true;
    for (int i = 0; i < m_engine->getNbBindings(); ++i) {
        // nbDims == 3, means CHW, without batch
        if (m_engine->getBindingDimensions(i).nbDims != 3)
            m_trt_engine_has_batch = true;

        if (m_engine->bindingIsInput(nr_input)) {
            mgb_assert(is_input, "mixed input/output bindings");
            ++nr_input;
        } else {
            is_input = false;
        }
    }
    size_t nr_output = m_engine->getNbBindings() - nr_input;
    mgb_assert(nr_input == inputs.size(),
               "inputs size not equal: expect=%zu got=%zu", nr_input,
               inputs.size());
    for (auto i : inputs) {
        add_input({i});
    }
    if (nr_output == 1) {
        add_output(None);
    } else {
        for (size_t i = 0; i < nr_output; ++i)
            add_output(ssprintf("o%zu", i));
    }
    cg::add_workspace_output(this);
    add_equivalence_component<mgb::ScalarHash<void*>>(m_engine.get());
}

void TensorRTRuntimeOpr::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    auto batch = inp_shape.at(0)[0];
    auto get_mgb_shape = [this, batch](int binding_idx) -> TensorShape {
        auto dims = m_engine->getBindingDimensions(binding_idx);
#if NV_TENSOR_RT_VERSION >= 6001
        auto format = m_engine->getBindingFormat(binding_idx);
        // converting dims to nchw4 format
        if (format == nvinfer1::TensorFormat::kCHW4) {
            mgb_assert(dims.nbDims == 3 || dims.nbDims == 4,
                       "Tensor with NCHW4 format should have dimensions of "
                       "3/4.(got: %d)",
                       dims.nbDims);
            int chan_pos = 0;
            if (dims.nbDims == 4) {
                chan_pos = 1;
            }
            dims.nbDims = dims.nbDims + 1;
            dims.d[chan_pos] = dims.d[chan_pos] / 4;
            dims.d[dims.nbDims - 1] = 4;
        }
#endif
        return m_trt_engine_has_batch ? TensorRTOpr::dims2shape(dims)
                                      : TensorRTOpr::dims2shape(dims, batch);
    };
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        mgb_assert(batch == inp_shape[i][0], "input batchsize not equal");
        TensorShape shp = get_mgb_shape(i);
        mgb_assert(shp.eq_shape(inp_shape[i]),
                   "input shape mismatch: expect=%s got=%s",
                   shp.to_string().c_str(), inp_shape[i].to_string().c_str());
    }
    for (size_t i = 0; i < out_shape.size() - 1; ++i) {
        out_shape[i] = get_mgb_shape(i + input().size());
    }
    out_shape.back() = {intl::workspace_size(m_engine.get())};
}

void TensorRTRuntimeOpr::add_input_layout_constraint() {
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void TensorRTRuntimeOpr::scn_do_execute() {
    auto batch = this->input(0)->shape()[0];
    if (m_trt_engine_has_batch)
        m_manager.exec(this,
                m_gpu_allocator ? m_gpu_allocator->comp_node() : CompNode{},
                m_engine.get());
    else
        m_manager.exec(this,
                m_gpu_allocator ? m_gpu_allocator->comp_node() : CompNode{},
                m_engine.get(), batch);
}

void TensorRTRuntimeOpr::init_output_dtype() {
    DType dt_trt, dt_input;
    int idx = 0;
    for (auto inp : input()) {
        dt_trt = get_dtype_from_trt(m_engine->getBindingDataType(idx));
        dt_input = inp->dtype();
        mgb_assert(dt_trt.valid() && dt_input.valid() &&
                           dt_trt.enumv() == dt_input.enumv(),
                   "Input %d Dtype is not expected in trt engine: expected %s, "
                   "got %s",
                   idx, dt_trt.name(), dt_input.name());
        idx++;
    }

    for (size_t i = 0; i < output().size(); ++i) {
        dt_trt = get_dtype_from_trt(m_engine->getBindingDataType(idx));
        mgb_assert(dt_trt.valid(),
                   "output dtype checking failed: invalid dtype returned.");
        if (dt_trt.enumv() == DTypeEnum::QuantizedS8) {
            mgb_assert(output(i)->dtype().valid(),
                       "user should specify scale of output tensor of "
                       "TensorRTRuntimeOpr.");
        }
        if (!output(i)->dtype().valid())
            output(i)->dtype(dt_trt);
        idx++;
    }
}

SymbolVarArray TensorRTRuntimeOpr::make(
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        std::shared_ptr<GpuAllocator> gpu_allocator, const SymbolVarArray& src,
        const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto tensor_rt_opr = std::make_unique<TensorRTRuntimeOpr>(
            std::move(engine), std::move(gpu_allocator), var_node_array,
            config);
    auto ret = cg::to_symbol_var_array(
            src[0].node()
                    ->owner_graph()
                    ->insert_opr(std::move(tensor_rt_opr))
                    ->output());
    ret.pop_back();  // remove workspace
    return ret;
}

SymbolVarArray TensorRTRuntimeOpr::make(const void* buf, size_t buf_size,
                                        const SymbolVarArray& src,
                                        const OperatorNodeConfig& config) {
    mgb_throw_if(
            !CompNode::get_device_count(CompNode::DeviceType::CUDA),
            SystemError,
            "can not create TensorRTRuntimeOpr when CUDA is not available");
    mgb_assert(!src.empty(), "no inputs provided");
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(TensorRTOpr::Logger::instance()), {}};
    auto gpu_allocator =
            std::make_shared<GpuAllocator>(src[0].node()->comp_node());
    runtime->setGpuAllocator(gpu_allocator.get());
    auto engine = runtime->deserializeCudaEngine(buf, buf_size, nullptr);
    mgb_assert(engine, "failed to deserialize ICudaEngine");
    return make(to_shared_ptr_engine(engine), gpu_allocator, src, config);
}

void TensorRTRuntimeOpr::LoadDumpImpl::dump(serialization::OprDumpContext& ctx,
                                            const cg::OperatorNodeBase& opr) {
    TensorRTUniquePtr<nvinfer1::IHostMemory> buf{
            opr.cast_final_safe<Opr>().trt_cuda_engine()->serialize(), {}};
    mgb_assert(buf, "failed to serialize ICudaEngine");
    ctx.dump_buf_with_len(buf->data(), buf->size());
}

cg::OperatorNodeBase* TensorRTRuntimeOpr::LoadDumpImpl::load(
        serialization::OprLoadContext& ctx, const cg::VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    inputs.at(0)->comp_node().activate();
    auto buf = ctx.load_shared_buf_with_len();
    return Opr::make(buf.data(), buf.size(), cg::to_symbol_var_array(inputs),
                     config)
            .at(0)
            .node()
            ->owner_opr();
}


#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
