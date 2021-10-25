/**
 * \file src/opr/impl/mc20_runtime_op.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/mc20_runtime_op.h"
#include "megbrain/common.h"
#include "megbrain/graph/event.h"
#include "megdnn/dtype.h"

#include <memory>

#if MGB_MC20

using namespace mgb;
using namespace opr;

namespace {
TensorShape mc20_shape_to_mgb_shape(AX_NPU_SDK_EX_TENSOR_META_T tensor_meta) {
    TensorShape ret;
    ret.ndim = tensor_meta.nShapeNDim;
    for (size_t i = 0; i < ret.ndim; ++i) {
        ret[i] = tensor_meta.pShape[i];
    }
    return ret;
}
DType mc20_dtype_to_mgb_dtype(AX_NPU_SDK_EX_ADV_TENSOR_DTYPE data_type) {
    switch (data_type) {
        case AX_NPU_TDT_UINT8:
            return dtype::Uint8();
        case AX_NPU_TDT_FLOAT32:
            return dtype::Float32();
        case AX_NPU_TDT_INT16:
            return dtype::Int16();
        case AX_NPU_TDT_INT32:
            return dtype::Int32();
        default:
            mgb_throw(
                    MegBrainError, "MC20DataType %d is not supported by MegBrain.",
                    static_cast<int>(data_type));
    }
}

};  // namespace

constexpr AX_NPU_SDK_EX_HANDLE_T MC20RuntimeOpr::INVALID_MODEL_HANDLE;

/* ====================== MC20RuntimeOpr ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MC20RuntimeOpr);
MC20RuntimeOpr::MC20RuntimeOpr(
        SharedBuffer buf, AX_NPU_SDK_EX_HANDLE_T model_handle,
        const VarNodeArray& inputs, const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "mc20_runtime", inputs),
          m_buffer{std::move(buf)},
          m_model_handle(model_handle) {
    mgb_assert(
            inputs[0]->comp_node().device_type() == CompNode::DeviceType::MC20,
            "MC20RuntimeOpr can only be used on mc20 comp node; "
            "got %s",
            inputs[0]->comp_node().to_string().c_str());

    for (auto i : inputs) {
        add_input({i});
    }
    if (m_model_handle == INVALID_MODEL_HANDLE) {
        MGB_MC20_CHECK(AX_NPU_SDK_EX_Create_handle(
                &m_model_handle, m_buffer.data(), m_buffer.size()));
        m_is_model_holder = true;
    }

    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    size_t nr_outputs = io_info->nOutputSize;
    bool has_workspace = false;
    if (nr_outputs == 1) {
        const auto& tensor_meta = *(io_info->pOutputs[0].pTensorMeta);
        add_output(std::string(reinterpret_cast<char*>(tensor_meta.pName)));
        if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
            mgb_assert(tensor_meta.nInnerSize > 0);
            has_workspace = true;
        }

    } else {
        for (size_t i = 0; i < nr_outputs; ++i) {
            const auto& tensor_meta = *(io_info->pOutputs[i].pTensorMeta);
            add_output(std::string(reinterpret_cast<char*>(tensor_meta.pName)));
            if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
                mgb_assert(tensor_meta.nInnerSize > 0);
                has_workspace = true;
            }
        }
    }
    mgb_assert(has_workspace, "Currently only support model with cpu tail");

    //! \warning There is no interface in MC20 to get the batch size of
    //! model.MC20 supports multi-batch by changing the input of n-batch to n
    //! 1-batch input.
    mgb_assert(
            io_info->nInputSize % inputs.size() == 0,
            "The number of inputs in the neu model should be multiple of "
            "the number of inputs in megbrain, but got %zu(neu model) vs "
            "%zu(mgb model)",
            io_info->nInputSize, inputs.size());
    m_model_batch = reinterpret_cast<size_t>(io_info->nInputSize / inputs.size());

    add_equivalence_component<mgb::ScalarHash<const void*>>(m_buffer.data());
    cg::add_workspace_output(this);
};

MC20RuntimeOpr::~MC20RuntimeOpr() {
    if (m_is_model_holder) {
        MGB_MC20_CHECK(AX_NPU_SDK_EX_Destroy_handle(m_model_handle));
    }
}

void MC20RuntimeOpr::execute_mc20() {
    auto&& mc20_env = CompNodeEnv::from_comp_node(input(0)->comp_node()).mc20_env();
    mc20_env.activate();

    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    AX_NPU_SDK_EX_IO_T npu_io;
    memset(&npu_io, 0, sizeof(npu_io));
    size_t batch_size = input(0)->dev_tensor().layout().shape[0];
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx += m_model_batch) {
        //! prepare input
        npu_io.nInputSize = io_info->nInputSize;
        auto inputs = std::make_unique<AX_NPU_SDK_EX_BUF_T[]>(npu_io.nInputSize);
        npu_io.pInputs = inputs.get();
        for (size_t i = 0; i < npu_io.nInputSize; i++) {
            // get input addr info
            size_t inp_idx = reinterpret_cast<size_t>(i / m_model_batch);
            AX_VOID* p_virtual_addr = input(inp_idx)->dev_tensor().raw_ptr();
            AX_U64 phy_addr = MC20MemoryManager::Instance().get_phyaddr(p_virtual_addr);
            auto nr_bytes_per_batch =
                    input(inp_idx)->layout().span().dist_byte() / batch_size;
            // add batch offset
            p_virtual_addr = reinterpret_cast<AX_VOID*>(
                    reinterpret_cast<AX_U64>(p_virtual_addr) +
                    nr_bytes_per_batch * (batch_idx + i % m_model_batch));
            phy_addr += nr_bytes_per_batch * (batch_idx + i % m_model_batch);

            MGB_MC20_CHECK(AX_NPU_SDK_EX_ADV_Make_io_buffer(
                    phy_addr, p_virtual_addr, nr_bytes_per_batch, phy_addr,
                    p_virtual_addr, nr_bytes_per_batch, &npu_io.pInputs[i]));
        }

        //! prepare output
        npu_io.nOutputSize = io_info->nOutputSize;
        auto outputs = std::make_unique<AX_NPU_SDK_EX_BUF_T[]>(npu_io.nOutputSize);
        npu_io.pOutputs = outputs.get();
        AX_U32 offset = 0;
        AX_VOID* inner_virtual_addr_start = nullptr;
        AX_U64 inner_phy_addr_start = 0;
        // get innder addr form workspace
        inner_virtual_addr_start = output(npu_io.nOutputSize)->dev_tensor().raw_ptr();
        inner_phy_addr_start =
                MC20MemoryManager::Instance().get_phyaddr(inner_virtual_addr_start);
        for (size_t i = 0; i < npu_io.nOutputSize; i++) {
            // get output addr info
            AX_VOID* p_virtual_addr = output(i)->dev_tensor().raw_ptr();
            AX_U64 phy_addr = 0;
            auto nr_bytes_per_batch =
                    output(i)->layout().span().dist_byte() / batch_size;
            // add batch offset
            p_virtual_addr = reinterpret_cast<AX_VOID*>(
                    reinterpret_cast<AX_U64>(p_virtual_addr) +
                    nr_bytes_per_batch * batch_idx);
            phy_addr += nr_bytes_per_batch * batch_idx;

            const auto& tensor_meta = *(io_info->pOutputs[i].pTensorMeta);
            if (tensor_meta.eMemoryType == AX_NPU_MT_PHYSICAL) {
                MGB_MC20_CHECK(AX_NPU_SDK_EX_ADV_Make_io_buffer(
                        phy_addr, p_virtual_addr, nr_bytes_per_batch, phy_addr,
                        p_virtual_addr, nr_bytes_per_batch, &npu_io.pOutputs[i]));
            } else if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
                auto p_inner_virtual_addr = reinterpret_cast<AX_VOID*>(
                        reinterpret_cast<AX_U64>(inner_virtual_addr_start) + offset);
                auto innerphy_addr = inner_phy_addr_start + offset;
                MGB_MC20_CHECK(AX_NPU_SDK_EX_ADV_Make_io_buffer(
                        phy_addr, p_virtual_addr, nr_bytes_per_batch, innerphy_addr,
                        p_inner_virtual_addr, tensor_meta.nInnerSize,
                        &npu_io.pOutputs[i]));

                offset += tensor_meta.nInnerSize;
            }
        }

        MGB_MC20_CHECK(AX_NPU_SDK_EX_Run_task_sync(m_model_handle, &npu_io));
    }
}

void MC20RuntimeOpr::init_output_comp_node() {
    //! set output to cpu compnode if has cpu tail
    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    CompNode input_cn;
    for (auto&& i : input()) {
        if (!input_cn.valid()) {
            input_cn = i->comp_node();
        } else {
            mgb_assert(
                    input_cn.mem_node() == i->comp_node().mem_node(),
                    "opr %s{%s} requires all input to be on the same memory "
                    "node expect=%s cur_var=%s cur_cn=%s",
                    this->cname(), this->dyn_typeinfo()->name,
                    input_cn.to_string().c_str(), i->cname(),
                    i->comp_node().to_string().c_str());
        }
    }
    for (size_t i = 0; i < io_info->nOutputSize; i++) {
        //! compnode of the var should be default_cpu as the output will be
        //! proxy to user
        output(i)->comp_node(CompNode::default_cpu());
    }
    //! the last output is workspace, which should be the same as input
    output(io_info->nOutputSize)->comp_node(input_cn);
}

MC20RuntimeOpr::NodeProp* MC20RuntimeOpr::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return ret;
}

void MC20RuntimeOpr::do_execute(ExecEnv& env) {
    CompNode cn = output(0)->comp_node();
    auto runner = [this, cn]() {
        this->owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(this, cn);
        cn.activate();
        execute_mc20();
        this->owner_graph()->event().signal_inplace<cg::event::AfterKernel>(this, cn);
    };
    env.dispatch_on_comp_node(cn, runner);

    // Send BeforeKernel/AfterKernel event on every different comp_node
    ThinHashSet<mgb::CompNode> st = cg::get_opr_comp_node_set(this);
    for (auto cn : st) {
        auto send_event = [this, cn]() {
            this->owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(
                    this, cn);
            this->owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
                    this, cn);
        };
        env.dispatch_on_comp_node(cn, send_event);
    }
}

void MC20RuntimeOpr::on_output_comp_node_stream_changed() {
    mgb_throw(SystemError, "comp node of output should not change");
}

void MC20RuntimeOpr::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);
    size_t nr_inputs = io_info->nInputSize;

    for (size_t i = 0; i < nr_inputs; ++i) {
        const auto& tensor_meta = *(io_info->pInputs[i].pTensorMeta);
        auto model_shape = mc20_shape_to_mgb_shape(tensor_meta);
        size_t inp_idx = reinterpret_cast<size_t>(i / m_model_batch);
        // enable mutibatch
        mgb_assert(
                inp_shape[inp_idx][0] % model_shape[0] == 0 &&
                        (inp_shape[inp_idx][0] / model_shape[0]) % m_model_batch == 0,
                "input %zu batch is %zu, while model's input batch is %zu", i,
                inp_shape[inp_idx][0], model_shape[0]);
        model_shape[0] = inp_shape[inp_idx][0];
        mgb_assert(
                model_shape.eq_shape(inp_shape[inp_idx]),
                "shape mismatch of input %zu, expected: %s got: %s", i,
                model_shape.to_string().c_str(),
                inp_shape[inp_idx].to_string().c_str());
    }
    size_t input_batch = (io_info->pInputs[0].pTensorMeta)->pShape[0];
    //! \warning mc20 sdk implement multi-batch by breaking an n-batch input up
    //! into n 1-batch inputs
    mgb_assert(input_batch == 1, "input batch: %d, net's input batch: 1", input_batch);
    AX_U32 workspace_size = 0;
    for (size_t i = 0; i < io_info->nOutputSize; ++i) {
        const auto& tensor_meta = *(io_info->pOutputs[i].pTensorMeta);
        out_shape[i] = mc20_shape_to_mgb_shape(tensor_meta);
        // enable mutibatch
        out_shape[i][0] =
                out_shape[i][0] * inp_shape[0][0] / input_batch / m_model_batch;
        if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
            workspace_size += tensor_meta.nInnerSize;
        }
    }
    out_shape.back() = {workspace_size};
}

void MC20RuntimeOpr::add_input_layout_constraint() {
    //! default contiguous
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void MC20RuntimeOpr::init_output_dtype() {
    DType dt_mc20, dt_input;
    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);
    for (size_t i = 0; i < io_info->nInputSize; ++i) {
        dt_mc20 = mc20_dtype_to_mgb_dtype(io_info->pInputs[i].eDType);
        size_t inp_idx = reinterpret_cast<size_t>(i / m_model_batch);
        dt_input = input(inp_idx)->dtype();
        mgb_assert(
                dt_mc20.valid() && dt_input.valid() &&
                        dt_mc20.enumv() == dt_input.enumv(),
                "dtype mismatch of input %zu: expected %s, "
                "got %s",
                i, dt_mc20.name(), dt_input.name());
    }

    for (size_t i = 0; i < io_info->nOutputSize; ++i) {
        dt_mc20 = mc20_dtype_to_mgb_dtype(io_info->pOutputs[i].eDType);
        mgb_assert(
                dt_mc20.valid(),
                "output dtype checking failed: invalid dtype returned.");
        if (!output(i)->dtype().valid())
            output(i)->dtype(dt_mc20);
    }
}

SymbolVarArray MC20RuntimeOpr::make(
        SharedBuffer buf, const SymbolVarArray& src, const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto mc20_runtime_opr = std::make_unique<MC20RuntimeOpr>(
            std::move(buf), INVALID_MODEL_HANDLE, var_node_array, config);
    auto ret = cg::to_symbol_var_array(src[0].node()
                                               ->owner_graph()
                                               ->insert_opr(std::move(mc20_runtime_opr))
                                               ->output());
    ret.pop_back();  // remove workspace
    return ret;
}

SymbolVarArray MC20RuntimeOpr::make(
        const void* buf, size_t size, const SymbolVarArray& src,
        const OperatorNodeConfig& config) {
    mgb_throw_if(
            !CompNode::get_device_count(CompNode::DeviceType::MC20), SystemError,
            "can not create MC20RuntimeOpr when mc20 is not "
            "available");
    std::shared_ptr<uint8_t> shptr{new uint8_t[size], [](uint8_t* p) { delete[] p; }};
    memcpy(shptr.get(), buf, size);
    SharedBuffer buffer{std::move(shptr), size};
    return make(std::move(buffer), src, config);
}

SymbolVarArray MC20RuntimeOpr::make(
        SharedBuffer buf, AX_NPU_SDK_EX_HANDLE_T model_handle,
        const SymbolVarArray& src, const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto mc20_runtime_opr = std::make_unique<MC20RuntimeOpr>(
            std::move(buf), model_handle, var_node_array, config);
    auto ret = cg::to_symbol_var_array(src[0].node()
                                               ->owner_graph()
                                               ->insert_opr(std::move(mc20_runtime_opr))
                                               ->output());
    ret.pop_back();  // remove workspace
    return ret;
}

#endif  // MGB_MC20

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
