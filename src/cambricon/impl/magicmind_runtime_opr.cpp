/**
 * \file src/cambricon/impl/magicmind_runtime_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/cambricon/magicmind_runtime_opr.h"
#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"

#if MGB_CAMBRICON

using namespace mgb;
using namespace opr;
using namespace magicmind;

namespace {
Dims mgb_shape_to_mm_dims(TensorShape mgb_shp) {
    size_t ndim = mgb_shp.ndim;
    std::vector<int64_t> dimensions(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        dimensions[i] = mgb_shp[i];
    }
    return Dims{dimensions};
}
TensorShape mm_dims_to_mgb_shape(const Dims& dims) {
    TensorShape ret;
    ret.ndim = dims.GetDimsNum();
    auto&& dimensions = dims.GetDims();
    for (size_t i = 0; i < ret.ndim; ++i) {
        ret[i] = dimensions[i];
    }
    return ret;
}
DType mm_dtype_to_mgb_dtype(DataType data_type) {
    switch (data_type) {
        case DataType::FLOAT16:
#if !MEGDNN_DISABLE_FLOAT16
            return dtype::Float16();
#else
            mgb_throw(MegBrainError, "Float16 support is disabled at compile time.");
#endif
        case DataType::FLOAT32:
            return dtype::Float32();
        case DataType::INT8:
            return dtype::QuantizedS8(1.f);
        case DataType::INT16:
            return dtype::Int16();
        case DataType::INT32:
            return dtype::Int32();
        case DataType::UINT8:
            return dtype::Uint8();
        //! TODO: check scale
        case DataType::QINT8:
            return dtype::QuantizedS8(1.f);
        case DataType::INT4:
            return dtype::QuantizedS4(1.f);
        case DataType::UINT4:
            return dtype::Quantized4Asymm(1.f, static_cast<uint8_t>(8));
        default:
            mgb_throw(
                    MegBrainError, "DataType %u is not supported by MegEngine.",
                    static_cast<uint32_t>(data_type));
    }
}
DataType mgb_dtype_to_mm_dtype(DType data_type) {
    switch (data_type.enumv()) {
#if !MEGDNN_DISABLE_FLOAT16
        case DTypeEnum::Float16:
            return DataType::FLOAT16;
#endif
        case DTypeEnum::Float32:
            return DataType::FLOAT32;
        case DTypeEnum::QuantizedS8:
            return DataType::QINT8;
        case DTypeEnum::Int8:
            return DataType::INT8;
        case DTypeEnum::Int32:
            return DataType::INT32;
        case DTypeEnum::Uint8:
            return DataType::UINT8;
        case DTypeEnum::QuantizedS4:
            return DataType::INT4;
        case DTypeEnum::Quantized4Asymm:
            return DataType::UINT4;
        default:
            mgb_throw(
                    MegBrainError,
                    "megengine data type %s is not supported by magicmind.",
                    data_type.name());
    }
}
};  // namespace

/* =========== MagicMindRuntimeOpr::CambriconAllocator =========== */
class MagicMindRuntimeOpr::CambriconAllocator final : public IAllocator {
    CompNode m_cn;
    std::mutex m_ptr2size_mtx;
    ThinHashMap<void*, size_t> m_ptr2size;

public:
    explicit CambriconAllocator(CompNode cn);
    ~CambriconAllocator() noexcept;

    void* AllocateRaw(size_t size, size_t alignment) override;
    void DeallocateRaw(void* ptr) override;

    CompNode comp_node() const { return m_cn; }
};

MagicMindRuntimeOpr::CambriconAllocator::CambriconAllocator(CompNode cn) : m_cn{cn} {
    mgb_assert(
            cn.device_type() == CompNode::DeviceType::CAMBRICON,
            "invalid comp node %s for CambriconAllocator", cn.to_string().c_str());
}

MagicMindRuntimeOpr::CambriconAllocator::~CambriconAllocator() noexcept {
    MGB_LOCK_GUARD(m_ptr2size_mtx);
    if (!m_ptr2size.empty()) {
        std::string msg{"there are unreleased magicmind mem buffers:\n"};
        for (auto&& i : m_ptr2size) {
            msg.append(ssprintf("  %p: %zu\n", i.first, i.second));
        }
        mgb_log_error("%sabort now", msg.c_str());
        mgb_trap();
    }
}

void* MagicMindRuntimeOpr::CambriconAllocator::AllocateRaw(
        size_t size, size_t alignment) {
    static bool enable_log = getenv("MGE_LOG_MAGICMIND_MEM_ALLOC");
    mgb_assert(!(alignment & (alignment - 1)), "invalid alignment(%zu)", alignment);
    auto ret = m_cn.alloc_device(size);
    mgb_assert(
            !(reinterpret_cast<uintptr_t>(ret) & (alignment - 1)),
            "alignment not required(ptr:%p,alignment:%zu)", ret, alignment);
    if (enable_log) {
        mgb_log("magicmind mem alloc on %s: size=%zu, align=%zu, ptr=%p",
                m_cn.to_string().c_str(), size, alignment, ret);
    }
    {
        MGB_LOCK_GUARD(m_ptr2size_mtx);
        m_ptr2size[ret] = size;
    }
    return ret;
}

void MagicMindRuntimeOpr::CambriconAllocator::DeallocateRaw(void* ptr) {
    {
        auto iter = m_ptr2size.find(ptr);
        mgb_assert(iter != m_ptr2size.end(), "ptr %p not found", ptr);
        m_ptr2size.erase(iter);
    }
    m_cn.free_device(ptr);
}

/* ====================== MagicMindRuntimeOpr ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MagicMindRuntimeOpr);
MagicMindRuntimeOpr::MagicMindRuntimeOpr(
        IModelPtr model, CambriconAllocatorPtr allocator, const VarNodeArray& inputs,
        const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "magic_runtime", inputs),
          m_allocator{std::move(allocator)},
          m_engine{nullptr},
          m_context{nullptr},
          m_model{std::move(model)} {
    mgb_assert(
            inputs[0]->comp_node().device_type() == CompNode::DeviceType::CAMBRICON,
            "MagicMindRuntimeOpr can only be used on cambricon comp node; "
            "got %s",
            inputs[0]->comp_node().to_string().c_str());
    size_t nr_inputs = m_model->GetInputNum();
    mgb_assert(
            nr_inputs == inputs.size(), "input number mismatch(got:%zu,expected:%zu)",
            inputs.size(), nr_inputs);
    for (auto i : inputs) {
        add_input({i});
    }
    size_t nr_outputs = m_model->GetOutputNum();
    for (size_t i = 0; i < nr_outputs; ++i) {
        add_output(m_model->GetOutputName(i));
    }
    IModel::EngineConfig engine_config;
    engine_config.device_type = "MLU";
    engine_config.allocator = m_allocator.get();
    auto&& cnrt_env = CompNodeEnv::from_comp_node(m_allocator->comp_node()).cnrt_env();
    cnrt_env.activate();
    m_engine = {
            m_model->CreateIEngine(engine_config),
            magicmind_intl::MagicMindDeleter<IEngine>()};
    mgb_assert(
            m_engine != nullptr,
            "create IEngine failed, corresponding MagicMindRuntimeOpr(%s)", cname());
    cg::add_workspace_output(this);
    add_equivalence_component<mgb::ScalarHash<void*>>(m_model.get());
};

void MagicMindRuntimeOpr::scn_do_execute() {
    mgb_assert(m_engine != nullptr);
    mgb_assert(m_context != nullptr);
    auto&& cnrt_env = CompNodeEnv::from_comp_node(input(0)->comp_node()).cnrt_env();
    cnrt_env.activate();
    std::vector<IRTTensor*> inputs, outputs;
    MM_CHECK(CreateInputTensors(m_context.get(), &inputs));
    MM_CHECK(CreateOutputTensors(m_context.get(), &outputs));
    size_t nr_inputs = input().size();
    mgb_assert(nr_inputs == inputs.size());
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto&& iname = m_model->GetInputName(i);
        auto tensor = FindIRTTensorByName(inputs, iname);
        mgb_assert(
                tensor != nullptr, "failed to find input tensor(name:%s)",
                iname.c_str());
        MM_CHECK(tensor->SetDimensions(mgb_shape_to_mm_dims(input(i)->shape())));
        MM_CHECK(tensor->SetData(input(i)->dev_tensor().raw_ptr()));
    }
    size_t nr_outputs = output().size();
    mgb_assert(nr_outputs == outputs.size() + 1);
    for (size_t i = 0; i < nr_outputs - 1; ++i) {
        auto&& oname = m_model->GetOutputName(i);
        auto tensor = FindIRTTensorByName(outputs, oname);
        mgb_assert(
                tensor != nullptr, "failed to find output tensor(name:%s)",
                oname.c_str());
        MM_CHECK(tensor->SetDimensions(mgb_shape_to_mm_dims(output(i)->shape())));
        MM_CHECK(tensor->SetData(output(i)->dev_tensor().raw_ptr()));
    }
    auto size = output().back()->dev_tensor().layout().span().dist_byte();
    MM_CHECK(m_context->SetWorkspace(output().back()->dev_tensor().raw_ptr(), size));
    MM_CHECK(m_context->Enqueue(inputs, outputs, cnrt_env.queue));
    for (auto&& i : inputs) {
        i->Destroy();
    }
    for (auto&& o : outputs) {
        o->Destroy();
    }
}

void MagicMindRuntimeOpr::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    mgb_assert(m_engine != nullptr);
    mgb_assert(input().size() == inp_shape.size());
    auto&& cnrt_env = CompNodeEnv::from_comp_node(input(0)->comp_node()).cnrt_env();
    cnrt_env.activate();
    if (m_context == nullptr) {
        m_context = {
                m_engine->CreateIContext(),
                magicmind_intl::MagicMindDeleter<IContext>()};
        mgb_assert(
                m_context != nullptr,
                "failed to create IContext, corresponding MagicMindRuntimeOpr(%s)",
                cname());
    }
    std::vector<IRTTensor*> inputs, outputs;
    MM_CHECK(CreateInputTensors(m_context.get(), &inputs));
    MM_CHECK(CreateOutputTensors(m_context.get(), &outputs));
    size_t nr_inputs = input().size();
    mgb_assert(nr_inputs == inputs.size());
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto&& iname = m_model->GetInputName(i);
        auto tensor = FindIRTTensorByName(inputs, iname);
        mgb_assert(
                tensor != nullptr, "failed to find input tensor(name:%s)",
                iname.c_str());
        MM_CHECK(tensor->SetDimensions(mgb_shape_to_mm_dims(input(i)->shape())));
    }
    if (Status::OK() == m_context->InferOutputShape(inputs, outputs)) {
        size_t nr_outputs = output().size();
        mgb_assert(nr_outputs == outputs.size() + 1);
        for (size_t i = 0; i < nr_outputs - 1; ++i) {
            auto&& oname = m_model->GetOutputName(i);
            auto tensor = FindIRTTensorByName(outputs, oname);
            mgb_assert(
                    tensor != nullptr, "failed to find output tensor(name:%s)",
                    oname.c_str());
            auto&& dims = tensor->GetDimensions();
            out_shape[i] = mm_dims_to_mgb_shape(dims);
        }
        std::vector<Dims> shape(inp_shape.size());
        for (size_t i = 0; i < nr_inputs; ++i) {
            shape[i] = mgb_shape_to_mm_dims(input(i)->shape());
        }
        size_t wk_size = 0;
        MM_CHECK(m_engine->QueryContextMaxWorkspaceSize(shape, &wk_size));
        out_shape.back() = {wk_size};
    } else {
        mgb_assert(
                false, "static shape infer for MagicMindRuntimeOpr(%s) failed",
                cname());
    }
    return;
    for (auto&& i : inputs) {
        i->Destroy();
    }
    for (auto&& o : outputs) {
        o->Destroy();
    }
}

void MagicMindRuntimeOpr::add_input_layout_constraint() {
    //! default contiguous
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void MagicMindRuntimeOpr::init_output_dtype() {
    std::vector<DataType> inp_dtypes = m_model->GetInputDataTypes();
    mgb_assert(
            inp_dtypes.size() == input().size(),
            "input size mismatch(got:%zu,expected:%zu)", inp_dtypes.size(),
            input().size());
    size_t nr_inputs = input().size();
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto dt_mm = mm_dtype_to_mgb_dtype(inp_dtypes[i]);
        auto dt_inp = input(i)->dtype();
        MGB_MARK_USED_VAR(dt_mm);
        MGB_MARK_USED_VAR(dt_inp);
        mgb_assert(
                dt_mm.valid() && dt_inp.valid() && dt_mm.enumv() == dt_inp.enumv(),
                "input %zu's data type mismatch with that in "
                "IModel: expected %s, got %s",
                i, dt_mm.name(), dt_inp.name());
    }
    std::vector<DataType> out_dtypes = m_model->GetOutputDataTypes();
    mgb_assert(
            out_dtypes.size() + 1 == output().size(),
            "output size mismatch(got:%zu,expected:%zu)", out_dtypes.size(),
            output().size());
    size_t nr_outputs = out_dtypes.size();
    for (size_t i = 0; i < nr_outputs; ++i) {
        auto dt_mm = mm_dtype_to_mgb_dtype(out_dtypes[i]);
        mgb_assert(
                dt_mm.valid(), "output dtype checking failed: invalid dtype returned.");
        if (dt_mm.enumv() == DTypeEnum::QuantizedS8) {
            mgb_assert(
                    output(i)->dtype().valid(),
                    "user should specify scale of output tensor of "
                    "MagicMindRuntimeOpr.");
        }
        if (!output(i)->dtype().valid())
            output(i)->dtype(dt_mm);
    }
}

SymbolVarArray MagicMindRuntimeOpr::make(
        IModelPtr model, CambriconAllocatorPtr allocator, const SymbolVarArray& src,
        const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto magicmind_runtime_opr = std::make_unique<MagicMindRuntimeOpr>(
            std::move(model), std::move(allocator), var_node_array, config);
    auto ret = cg::to_symbol_var_array(
            src[0].node()
                    ->owner_graph()
                    ->insert_opr(std::move(magicmind_runtime_opr))
                    ->output());
    ret.pop_back();  // remove workspace
    return ret;
}

SymbolVarArray MagicMindRuntimeOpr::make(
        const void* buf, size_t size, const SymbolVarArray& src,
        const OperatorNodeConfig& config) {
    mgb_throw_if(
            !CompNode::get_device_count(CompNode::DeviceType::CAMBRICON), SystemError,
            "can not create MagicMindRuntimeOpr when MagicMind is not "
            "available");
    auto cambricon_allocator =
            std::make_shared<CambriconAllocator>(src[0].node()->comp_node());
    IModelPtr model = make_model_ptr(CreateIModel());
    model->DeserializeFromMemory(const_cast<void*>(buf), size);
    return make(std::move(model), std::move(cambricon_allocator), src, config);
}

#endif  // MGB_CAMBRICON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
