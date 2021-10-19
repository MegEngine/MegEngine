/**
 * \file src/custom/include/megbrain/custom/op.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <unordered_set>
#include "param.h"
#include "tensor.h"

#define PREVENT_COPY_AND_ASSIGN(Cls)     \
    Cls(const Cls&) = delete;            \
    Cls(const Cls&&) = delete;           \
    Cls& operator=(const Cls&) = delete; \
    Cls& operator=(const Cls&&) = delete

#define CUSTOM_OP_MAJOR 0
#define CUSTOM_OP_MINOR 1
#define CUSTOM_OP_PATCH 0

#define CUSTOM_OP_VERSION \
    CUSTOM_OP_MAJOR * 10000 + CUSTOM_OP_MINOR * 100 + CUSTOM_OP_PATCH

namespace custom {

using RunTimeId = uint64_t;

class ArgInfo {
    CUSTOM_PIMPL_CLS_DECL(ArgInfo);
    MGE_WIN_DECLSPEC_FUC ArgInfo(
            const std::string& name, const std::string& desc,
            const std::unordered_set<std::string>& dtypes, const int& ndim,
            const std::string& mem_stgy);

    MGE_WIN_DECLSPEC_FUC const std::string& name(void) const;
    MGE_WIN_DECLSPEC_FUC const std::string& desc(void) const;
    MGE_WIN_DECLSPEC_FUC const std::unordered_set<std::string>& dtypes(void) const;
    MGE_WIN_DECLSPEC_FUC int ndim(void) const;
    MGE_WIN_DECLSPEC_FUC const std::string& mem_strategy(void) const;

    MGE_WIN_DECLSPEC_FUC std::string str() const;
};

class CustomOp {
    std::unique_ptr<void, void_deleter> m_impl;

public:
    MGE_WIN_DECLSPEC_FUC CustomOp(const std::string& op_type, uint32_t version);
    PREVENT_COPY_AND_ASSIGN(CustomOp);

    using DeviceInferFuncPtr =
            void (*)(const std::vector<Device>&, const Param&, std::vector<Device>&);
    using ShapeInferFuncPtr =
            void (*)(const std::vector<Shape>&, const Param&, std::vector<Shape>&);
    using DTypeInferFuncPtr =
            void (*)(const std::vector<DType>&, const Param&, std::vector<DType>&);
    using FormatInferFuncPtr =
            void (*)(const std::vector<Format>&, const Param&, std::vector<Format>&);
    using PreprocessFuncPtr =
            void (*)(const std::vector<Tensor>&, const Param&, std::vector<Tensor>&);
    using PostprocessFuncPtr =
            void (*)(const std::vector<Tensor>&, const Param&, std::vector<Tensor>&);
    using ComputeFuncPtr =
            void (*)(const std::vector<Tensor>&, const Param&, std::vector<Tensor>&);

    // write for forward
    MGE_WIN_DECLSPEC_FUC CustomOp& set_device_infer(DeviceInferFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_shape_infer(ShapeInferFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_dtype_infer(DTypeInferFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_format_infer(FormatInferFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_preprocess(PreprocessFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_preprocess(
            const std::string& device, PreprocessFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_postprocess(PostprocessFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_postprocess(
            const std::string& device, PostprocessFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_compute(ComputeFuncPtr func);
    MGE_WIN_DECLSPEC_FUC CustomOp& set_compute(
            const std::string& device, ComputeFuncPtr func);

    MGE_WIN_DECLSPEC_FUC CustomOp& set_description(const std::string& op_desc);
    MGE_WIN_DECLSPEC_FUC CustomOp& add_input(
            const std::string& name, const std::string& desc,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    MGE_WIN_DECLSPEC_FUC CustomOp& add_output(
            const std::string& name, const std::string& desc,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    MGE_WIN_DECLSPEC_FUC CustomOp& add_input(
            const std::string& name,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    MGE_WIN_DECLSPEC_FUC CustomOp& add_output(
            const std::string& name,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    MGE_WIN_DECLSPEC_FUC CustomOp& add_inputs(const size_t& input_num);
    MGE_WIN_DECLSPEC_FUC CustomOp& add_outputs(const size_t& output_num);
    MGE_WIN_DECLSPEC_FUC CustomOp& add_param(
            const std::string& name, const ParamVal& default_val);
    MGE_WIN_DECLSPEC_FUC CustomOp& add_param(
            const std::string& name, const std::string& desc,
            const ParamVal& default_val);

    // read
    MGE_WIN_DECLSPEC_FUC std::string op_type(void) const;
    MGE_WIN_DECLSPEC_FUC std::string op_desc(void) const;
    MGE_WIN_DECLSPEC_FUC RunTimeId runtime_id(void) const;
    MGE_WIN_DECLSPEC_FUC size_t input_num(void) const;
    MGE_WIN_DECLSPEC_FUC size_t output_num(void) const;
    MGE_WIN_DECLSPEC_FUC std::string str(void) const;

    MGE_WIN_DECLSPEC_FUC const ParamInfo& param_info(void) const;
    MGE_WIN_DECLSPEC_FUC ArgInfo input_info(size_t idx) const;
    MGE_WIN_DECLSPEC_FUC ArgInfo output_info(size_t idx) const;
    MGE_WIN_DECLSPEC_FUC const std::vector<ArgInfo>& inputs_info(void) const;
    MGE_WIN_DECLSPEC_FUC const std::vector<ArgInfo>& outputs_info(void) const;

    // use
    MGE_WIN_DECLSPEC_FUC std::vector<Device> infer_output_device(
            const std::vector<Device>&, const Param&) const;
    MGE_WIN_DECLSPEC_FUC std::vector<Shape> infer_output_shape(
            const std::vector<Shape>&, const Param&) const;
    MGE_WIN_DECLSPEC_FUC std::vector<DType> infer_output_dtype(
            const std::vector<DType>&, const Param&) const;
    MGE_WIN_DECLSPEC_FUC std::vector<Format> infer_output_format(
            const std::vector<Format>&, const Param&) const;
    MGE_WIN_DECLSPEC_FUC void compute(
            const std::vector<Tensor>&, const Param&, std::vector<Tensor>&) const;
};

}  // namespace custom
