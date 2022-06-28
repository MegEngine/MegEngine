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

class MGE_WIN_DECLSPEC_FUC ArgInfo {
    CUSTOM_PIMPL_CLS_DECL(ArgInfo);
    ArgInfo(const std::string& name, const std::string& desc,
            const std::unordered_set<std::string>& dtypes, const int& ndim,
            const std::string& mem_stgy);

    const std::string& name(void) const;
    const std::string& desc(void) const;
    const std::unordered_set<std::string>& dtypes(void) const;
    int ndim(void) const;
    const std::string& mem_strategy(void) const;

    std::string str() const;
};

class CudaRuntimeArgs;

class MGE_WIN_DECLSPEC_FUC RuntimeArgs {
    Device m_device;

public:
    RuntimeArgs() = default;
    RuntimeArgs(Device device) : m_device(device){};

    const Device& device() const { return m_device; }
};

class MGE_WIN_DECLSPEC_FUC CustomOp {
    std::unique_ptr<void, void_deleter> m_impl;

public:
    CustomOp(const std::string& op_type, uint32_t version);
    PREVENT_COPY_AND_ASSIGN(CustomOp);

    using DeviceInferFuncPtr =
            void (*)(const std::vector<Device>&, const Param&, std::vector<Device>&);
    using ShapeInferFuncPtr =
            void (*)(const std::vector<Shape>&, const Param&, std::vector<Shape>&);
    using DTypeInferFuncPtr =
            void (*)(const std::vector<DType>&, const Param&, std::vector<DType>&);
    using FormatInferFuncPtr =
            void (*)(const std::vector<Format>&, const Param&, std::vector<Format>&);
    using ProcessFuncPtr = void (*)(
            const std::vector<Tensor>&, const Param&, std::vector<Tensor>&,
            const RuntimeArgs&);
    using ProcessFuncPtrWithoutRuntimeArgs =
            void (*)(const std::vector<Tensor>&, const Param&, std::vector<Tensor>&);

    // write for forward
    CustomOp& set_device_infer(DeviceInferFuncPtr func);
    CustomOp& set_shape_infer(ShapeInferFuncPtr func);
    CustomOp& set_dtype_infer(DTypeInferFuncPtr func);
    CustomOp& set_format_infer(FormatInferFuncPtr func);
    //! set process function with RuntimeArgs e.g. cuda
    CustomOp& set_preprocess(ProcessFuncPtr func);
    CustomOp& set_preprocess(const std::string& device, ProcessFuncPtr func);
    CustomOp& set_postprocess(ProcessFuncPtr func);
    CustomOp& set_postprocess(const std::string& device, ProcessFuncPtr func);
    CustomOp& set_compute(ProcessFuncPtr func);
    CustomOp& set_compute(const std::string& device, ProcessFuncPtr func);

    //! set process function without RuntimeArgs e.g. cpu
    CustomOp& set_preprocess(ProcessFuncPtrWithoutRuntimeArgs func);
    CustomOp& set_preprocess(
            const std::string& device, ProcessFuncPtrWithoutRuntimeArgs func);
    CustomOp& set_postprocess(ProcessFuncPtrWithoutRuntimeArgs func);
    CustomOp& set_postprocess(
            const std::string& device, ProcessFuncPtrWithoutRuntimeArgs func);
    CustomOp& set_compute(ProcessFuncPtrWithoutRuntimeArgs func);
    CustomOp& set_compute(
            const std::string& device, ProcessFuncPtrWithoutRuntimeArgs func);

    CustomOp& set_description(const std::string& op_desc);
    CustomOp& add_input(
            const std::string& name, const std::string& desc,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    CustomOp& add_output(
            const std::string& name, const std::string& desc,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    CustomOp& add_input(
            const std::string& name,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    CustomOp& add_output(
            const std::string& name,
            const std::initializer_list<std::string>& legal_dtypes = {"float32"},
            int dims = -1, const std::string& mem_stgy = "default");
    CustomOp& add_inputs(const size_t& input_num);
    CustomOp& add_outputs(const size_t& output_num);
    CustomOp& add_param(const std::string& name, const ParamVal& default_val);
    CustomOp& add_param(
            const std::string& name, const std::string& desc,
            const ParamVal& default_val);

    // read
    std::string op_type(void) const;
    std::string op_desc(void) const;
    RunTimeId runtime_id(void) const;
    size_t input_num(void) const;
    size_t output_num(void) const;
    std::string str(void) const;

    const ParamInfo& param_info(void) const;
    ArgInfo input_info(size_t idx) const;
    ArgInfo output_info(size_t idx) const;
    const std::vector<ArgInfo>& inputs_info(void) const;
    const std::vector<ArgInfo>& outputs_info(void) const;

    // use
    std::vector<Device> infer_output_device(
            const std::vector<Device>&, const Param&) const;
    std::vector<Shape> infer_output_shape(
            const std::vector<Shape>&, const Param&) const;
    std::vector<DType> infer_output_dtype(
            const std::vector<DType>&, const Param&) const;
    std::vector<Format> infer_output_format(
            const std::vector<Format>&, const Param&) const;
    void compute(const std::vector<Tensor>&, const Param&, std::vector<Tensor>&) const;
};

}  // namespace custom
