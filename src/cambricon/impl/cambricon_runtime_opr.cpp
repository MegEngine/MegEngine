/**
 * \file src/cambricon/impl/cambricon_runtime_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/cambricon/cambricon_runtime_opr.h"
#include "megbrain/common.h"

#if MGB_CAMBRICON

using namespace mgb;
using namespace opr;

namespace {
SmallVector<int> mgb_shape_to_cnrt_shape(TensorShape mgb_shp) {
    int ndim = mgb_shp.ndim;
    SmallVector<int> cnrt_shp(ndim);
    for (int i = 0; i < ndim; ++i) {
        cnrt_shp[i] = mgb_shp[i];
    }
    return cnrt_shp;
}
TensorShape cnrt_shape_to_mgb_shape(int* dim_values, int dim_num) {
    TensorShape ret;
    ret.ndim = dim_num;
    for (int i = 0; i < dim_num; ++i) {
        ret[i] = dim_values[i];
    }
    return ret;
}
DType cnrt_dtype_to_mgb_dtype(cnrtDataType_t data_type) {
    switch (data_type) {
        case CNRT_FLOAT16:
#if !MEGDNN_DISABLE_FLOAT16
            return dtype::Float16();
#else
            mgb_throw(MegBrainError,
                      "Float16 support is disabled at compile time.");
#endif
        case CNRT_FLOAT32:
            return dtype::Float32();
        case CNRT_INT8:
            return dtype::QuantizedS8(1.f);
        case CNRT_INT16:
            return dtype::Int16();
        case CNRT_INT32:
            return dtype::Int32();
        case CNRT_UINT8:
            return dtype::Uint8();
        //! TODO: check scale
        case CNRT_QUANT8:
            return dtype::QuantizedS8(1.f);
        default:
            mgb_throw(MegBrainError,
                      "cnrtDataType %x is not supported by MegBrain.",
                      data_type);
    }
}
cnrtDataType_t mgb_dtype_to_cnrt_dtype(DType data_type) {
    switch (data_type.enumv()) {
#if !MEGDNN_DISABLE_FLOAT16
        case DTypeEnum::Float16:
            return CNRT_FLOAT16;
#endif
        case DTypeEnum::Float32:
            return CNRT_FLOAT32;
        case DTypeEnum::QuantizedS8:
            return CNRT_QUANT8;
        case DTypeEnum::Int32:
            return CNRT_INT32;
        default:
            mgb_throw(MegBrainError,
                      "megbrain data type %s is not supported by cnrt.",
                      data_type.name());
    }
}
};  // namespace

/* ====================== CambriconRuntimeOpr ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CambriconRuntimeOpr);
CambriconRuntimeOpr::CambriconRuntimeOpr(SharedBuffer buf, std::string symbol,
                                         const VarNodeArray& inputs,
                                         bool tensor_dim_mutable,
                                         const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "cambricon_runtime", inputs),
          m_buffer{std::move(buf)},
          m_symbol{std::move(symbol)},
          m_model{nullptr},
          m_function{nullptr},
          m_context{nullptr},
          m_tensor_dim_mutable{tensor_dim_mutable} {
    mgb_assert(inputs[0]->comp_node().device_type() ==
                       CompNode::DeviceType::CAMBRICON,
               "CambriconRuntimeOpr can only be used on cambricon comp node; "
               "got %s",
               inputs[0]->comp_node().to_string().c_str());

    for (auto i : inputs) {
        add_input({i});
    }
    if (m_model == nullptr) {
        m_model = {new cnrtModel_t(), cnrt_intl::ModelUnloader()};
        MGB_CNRT_CHECK(cnrtLoadModelFromMem(
                m_model.get(),
                reinterpret_cast<char*>(const_cast<void*>(m_buffer.data()))));
    }
    if (m_function == nullptr) {
        m_function = {new cnrtFunction_t(), cnrt_intl::FunctionDeleter()};
        MGB_CNRT_CHECK(cnrtCreateFunction(m_function.get()));
        MGB_CNRT_CHECK(cnrtExtractFunction(m_function.get(), *m_model,
                                           m_symbol.c_str()));
    }
    int nr_inputs = 0;
    int nr_outputs = 0;
    int64_t* inputs_size = nullptr;
    int64_t* outputs_size = nullptr;
    MGB_CNRT_CHECK(cnrtGetInputDataSize(&inputs_size, &nr_inputs, *m_function));
    mgb_assert(static_cast<size_t>(nr_inputs) == inputs.size(),
               "inputs size mismatch: expect=%d, got=%zu", nr_inputs,
               inputs.size());
    MGB_CNRT_CHECK(
            cnrtGetOutputDataSize(&outputs_size, &nr_outputs, *m_function));
    if (nr_outputs == 1) {
        add_output(None);
    } else {
        for (int i = 0; i < nr_outputs; ++i) {
            add_output(ssprintf("o%d", i));
        }
    }
    add_equivalence_component<mgb::ScalarHash<const void*>>(m_buffer.data());
};

void CambriconRuntimeOpr::scn_do_execute() {
    mgb_assert(m_function != nullptr);
    auto&& cnrt_env =
            CompNodeEnv::from_comp_node(input(0)->comp_node()).cnrt_env();
    cnrt_env.activate();
    if (m_context == nullptr) {
        m_context = {new cnrtRuntimeContext_t(),
                     cnrt_intl::RuntimeContextDeleter()};
        MGB_CNRT_CHECK(cnrtCreateRuntimeContext(m_context.get(), *m_function,
                                                nullptr));
        int dev_id = cnrt_env.device;
        MGB_CNRT_CHECK(cnrtSetRuntimeContextDeviceId(*m_context, dev_id));
        MGB_CNRT_CHECK(cnrtInitRuntimeContext(*m_context, nullptr));
    }
    size_t nr_inputs = input().size(), nr_outputs = output().size();
    SmallVector<void*> params(nr_inputs + nr_outputs);
    SmallVector<cnrtParamDesc_t> param_descs(nr_inputs + nr_outputs);
    for (size_t i = 0; i < nr_inputs; ++i) {
        params[i] = input(i)->dev_tensor().raw_ptr();
        MGB_CNRT_CHECK(cnrtCreateParamDesc(&param_descs[i]));
        MGB_CNRT_CHECK(cnrtSetDataTypeToParamDesc(
                param_descs[i], mgb_dtype_to_cnrt_dtype(input(i)->dtype())));
        auto dims = mgb_shape_to_cnrt_shape(input(i)->shape());
        MGB_CNRT_CHECK(cnrtSetShapeToParamDesc(param_descs[i], dims.data(),
                                               static_cast<int>(dims.size())));
    }
    for (size_t i = 0; i < nr_outputs; ++i) {
        params[nr_inputs + i] = output(i)->dev_tensor().raw_ptr();
        MGB_CNRT_CHECK(cnrtCreateParamDesc(&param_descs[nr_inputs + i]));
        MGB_CNRT_CHECK(cnrtSetDataTypeToParamDesc(
                param_descs[nr_inputs + i],
                mgb_dtype_to_cnrt_dtype(output(i)->dtype())));
        auto dims = mgb_shape_to_cnrt_shape(output(i)->shape());
        MGB_CNRT_CHECK(cnrtSetShapeToParamDesc(param_descs[nr_inputs + i],
                                               dims.data(),
                                               static_cast<int>(dims.size())));
    }
    MGB_CNRT_CHECK(cnrtInvokeRuntimeContext_V2(*m_context, param_descs.data(),
                                               params.data(), cnrt_env.queue,
                                               nullptr));
    for (auto& param : param_descs) {
        MGB_CNRT_CHECK(cnrtDestroyParamDesc(param));
    }
}

void CambriconRuntimeOpr::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    mgb_assert(m_function != nullptr);
    mgb_assert(input().size() == inp_shape.size());
    if (m_tensor_dim_mutable) {
        cnrtParamDescArray_t input_descs, output_descs;
        int inp_param_num = input().size();
        int out_param_num = output().size();
        MGB_CNRT_CHECK(cnrtCreateParamDescArray(&input_descs, inp_param_num));
        MGB_CNRT_CHECK(cnrtCreateParamDescArray(&output_descs, out_param_num));
        for (int i = 0; i < inp_param_num; ++i) {
            MGB_CNRT_CHECK(cnrtSetDataTypeToParamDesc(
                    input_descs[i],
                    mgb_dtype_to_cnrt_dtype(input(i)->dtype())));
            auto dims = mgb_shape_to_cnrt_shape(inp_shape[i]);
            MGB_CNRT_CHECK(
                    cnrtSetShapeToParamDesc(input_descs[i], dims.data(),
                                            static_cast<int>(dims.size())));
        }
        MGB_CNRT_CHECK(cnrtInferFunctionOutputShape(*m_function, inp_param_num,
                                                    input_descs, out_param_num,
                                                    output_descs));
        for (int i = 0; i < out_param_num; ++i) {
            int* dims = nullptr;
            int dim_num = 0;
            MGB_CNRT_CHECK(cnrtGetShapeFromParamDesc(output_descs[i], &dims,
                                                     &dim_num));
            out_shape[i] = cnrt_shape_to_mgb_shape(dims, dim_num);
        }
        MGB_CNRT_CHECK(cnrtDestroyParamDescArray(input_descs, inp_param_num));
        MGB_CNRT_CHECK(cnrtDestroyParamDescArray(output_descs, out_param_num));
    } else {
        //! check input shape match
        for (size_t i = 0; i < inp_shape.size(); ++i) {
            int* dim_values = nullptr;
            int dim_num = 0;
            MGB_CNRT_CHECK(cnrtGetInputDataShape(
                    &dim_values, &dim_num, static_cast<int>(i), *m_function));
            auto shp_in_func = cnrt_shape_to_mgb_shape(dim_values, dim_num);
            auto inpshp = inp_shape[i];
            MGB_MARK_USED_VAR(shp_in_func);
            mgb_assert(
                    inpshp.eq_shape(shp_in_func),
                    "input shape(%s) mismatch with that(%s) in cnrtFunction_t.",
                    inpshp.to_string().c_str(),
                    shp_in_func.to_string().c_str());
        }
        //! remarks: cnrt does not provide interface to let user manage
        //! workspace
        MGB_MARK_USED_VAR(mgb_dtype_to_cnrt_dtype);
        for (size_t i = 0; i < out_shape.size(); ++i) {
            int* dim_values = nullptr;
            int dim_num = 0;
            MGB_CNRT_CHECK(cnrtGetOutputDataShape(
                    &dim_values, &dim_num, static_cast<int>(i), *m_function));
            out_shape[i] = cnrt_shape_to_mgb_shape(dim_values, dim_num);
        }
    }
}

void CambriconRuntimeOpr::add_input_layout_constraint() {
    //! default contiguous
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void CambriconRuntimeOpr::init_output_dtype() {
    cnrtDataType_t* inp_dtype_array = nullptr;
    int inp_num;
    MGB_CNRT_CHECK(
            cnrtGetInputDataType(&inp_dtype_array, &inp_num, *m_function));
    for (size_t i = 0; i < input().size(); ++i) {
        auto dt_cnrt = cnrt_dtype_to_mgb_dtype(inp_dtype_array[i]);
        auto dt_inp = input(i)->dtype();
        MGB_MARK_USED_VAR(dt_cnrt);
        MGB_MARK_USED_VAR(dt_inp);
        mgb_assert(dt_cnrt.valid() && dt_inp.valid() &&
                           dt_cnrt.enumv() == dt_inp.enumv(),
                   "Input %zu's data type mismatch with that in "
                   "cnrtFunction_t: expected %s, got %s",
                   i, dt_cnrt.name(), dt_inp.name());
    }
    cnrtDataType_t* out_dtype_array = nullptr;
    int out_num;
    MGB_CNRT_CHECK(
            cnrtGetOutputDataType(&out_dtype_array, &out_num, *m_function));
    for (size_t i = 0; i < output().size(); ++i) {
        auto dt_cnrt = cnrt_dtype_to_mgb_dtype(out_dtype_array[i]);
        mgb_assert(dt_cnrt.valid(),
                   "output dtype checking failed: invalid dtype returned.");
        if (dt_cnrt.enumv() == DTypeEnum::QuantizedS8) {
            mgb_assert(output(i)->dtype().valid(),
                       "user should specify scale of output tensor of "
                       "CambriconRuntimeOpr.");
        }
        if (!output(i)->dtype().valid())
            output(i)->dtype(dt_cnrt);
    }
}

SymbolVarArray CambriconRuntimeOpr::make(SharedBuffer buf, std::string symbol,
                                         const SymbolVarArray& src,
                                         bool tensor_dim_mutable,
                                         const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto cambricon_runtime_opr = std::make_unique<CambriconRuntimeOpr>(
            std::move(buf), std::move(symbol), var_node_array,
            tensor_dim_mutable, config);
    auto ret = cg::to_symbol_var_array(
            src[0].node()
                    ->owner_graph()
                    ->insert_opr(std::move(cambricon_runtime_opr))
                    ->output());
    return ret;
}

SymbolVarArray CambriconRuntimeOpr::make(const void* buf, size_t size,
                                         std::string symbol,
                                         const SymbolVarArray& src,
                                         bool tensor_dim_mutable,
                                         const OperatorNodeConfig& config) {
    mgb_throw_if(!CompNode::get_device_count(CompNode::DeviceType::CAMBRICON),
                 SystemError,
                 "can not create CambriconRuntimeOpr when Cambricon is not "
                 "available");
    std::shared_ptr<uint8_t> shptr{new uint8_t[size],
                                   [](uint8_t* p) { delete[] p; }};
    memcpy(shptr.get(), buf, size);
    SharedBuffer buffer{std::move(shptr), size};
    return make(std::move(buffer), std::move(symbol), src, tensor_dim_mutable,
                config);
}

#endif  // MGB_CAMBRICON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
