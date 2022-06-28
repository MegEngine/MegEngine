#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include <sstream>
#include <unordered_set>
#include "megbrain/custom/op.h"
#include "megbrain/custom/utils.h"
#include "megbrain/utils/thin/function.h"

using namespace mgb;

namespace custom {

class ArgInfoImpl {
    std::string m_name;
    std::string m_desc;
    std::unordered_set<std::string> m_dtypes;
    int m_ndim;  // use int rather than size_t for representing m_dims = -1
    std::string m_mem_stgy;

    friend class ArgInfo;
};

CUSTOM_PIMPL_CLS_DEFINE(ArgInfo)

ArgInfo::ArgInfo(
        const std::string& name, const std::string& desc,
        const std::unordered_set<std::string>& dtypes, const int& ndim,
        const std::string& mem_stgy)
        : m_impl(new ArgInfoImpl(), impl_deleter<ArgInfoImpl>) {
    for (auto&& dtype : dtypes) {
        mgb_assert(
                DType::is_legal(dtype), "unsupported tensor data type: %s",
                dtype.c_str());
    }
    mgb_assert(mem_stgy == "default", "only default mem strategy is supported now!");
    TypedRef(ArgInfoImpl, m_impl.get()).m_name = name;
    TypedRef(ArgInfoImpl, m_impl.get()).m_desc = desc;
    TypedRef(ArgInfoImpl, m_impl.get()).m_dtypes = dtypes;
    TypedRef(ArgInfoImpl, m_impl.get()).m_ndim = ndim;
    TypedRef(ArgInfoImpl, m_impl.get()).m_mem_stgy = mem_stgy;
}

const std::string& ArgInfo::name(void) const {
    return TypedRef(ArgInfoImpl, m_impl.get()).m_name;
}

const std::string& ArgInfo::desc(void) const {
    return TypedRef(ArgInfoImpl, m_impl.get()).m_desc;
}

const std::unordered_set<std::string>& ArgInfo::dtypes(void) const {
    return TypedRef(ArgInfoImpl, m_impl.get()).m_dtypes;
}

int ArgInfo::ndim(void) const {
    return TypedRef(ArgInfoImpl, m_impl.get()).m_ndim;
}

const std::string& ArgInfo::mem_strategy(void) const {
    return TypedRef(ArgInfoImpl, m_impl.get()).m_mem_stgy;
}

std::string ArgInfo::str() const {
    std::stringstream ss;
    ss << "name: " << TypedRef(ArgInfoImpl, m_impl.get()).m_name << "\n"
       << "desc: " << TypedRef(ArgInfoImpl, m_impl.get()).m_desc << "\nlegal_dtypes: {";

    for (auto& val : TypedRef(ArgInfoImpl, m_impl.get()).m_dtypes) {
        ss << val << ", ";
    }
    if (TypedRef(ArgInfoImpl, m_impl.get()).m_dtypes.size() != 0) {
        ss.seekp(ss.tellp() - std::streampos(2));
    }

    ss << "}\ndims: " << TypedRef(ArgInfoImpl, m_impl.get()).m_ndim << "\n"
       << "memory_strategy: " << TypedRef(ArgInfoImpl, m_impl.get()).m_mem_stgy;
    return ss.str();
}

#define assert_inputs_size_right(inputs_vec)                                         \
    mgb_assert(                                                                      \
            inputs_vec.size() == input_num(), "op %s need %lu inputs but given %lu", \
            op_type().c_str(), static_cast<unsigned long>(input_num()),              \
            static_cast<unsigned long>(inputs_vec.size()))

#define assert_outputs_size_right(outputs_vec)                         \
    mgb_assert(                                                        \
            outputs_vec.size() == output_num(),                        \
            "op %s have %lu outputs but given %lu", op_type().c_str(), \
            static_cast<unsigned long>(output_num()),                  \
            static_cast<unsigned long>(outputs_vec.size()))

#define assert_arg_shape_dim_right(real_shape, arg_info)                              \
    mgb_assert(                                                                       \
            (arg_info).ndim() == -1 || static_cast<int>((real_shape).ndim()) ==       \
                                               static_cast<int>((arg_info).ndim()),   \
            "%s's args: %s dim match error, need %d but given %d", op_type().c_str(), \
            (arg_info).name().c_str(), static_cast<int>((arg_info).ndim()),           \
            static_cast<int>((real_shape).ndim()))

class CustomOpImpl {
    static constexpr uint32_t CURRENT_VERSION = CUSTOM_OP_VERSION;
    const uint32_t m_version;

    const std::string m_op_type;
    std::string m_op_desc;
    std::vector<ArgInfo> m_input_infos;
    std::vector<ArgInfo> m_output_infos;
    ParamInfo m_param_infos;

    using DeviceInfer = thin_function<void(
            const std::vector<Device>&, const Param&, std::vector<Device>&)>;
    using ShapeInfer = thin_function<void(
            const std::vector<Shape>&, const Param&, std::vector<Shape>&)>;
    using DTypeInfer = thin_function<void(
            const std::vector<DType>&, const Param&, std::vector<DType>&)>;
    using FormatInfer = thin_function<void(
            const std::vector<Format>&, const Param&, std::vector<Format>&)>;
    using Process = thin_function<void(
            const std::vector<Tensor>&, const Param&, std::vector<Tensor>&,
            const RuntimeArgs&)>;

    DeviceInfer infer_output_device_func;
    ShapeInfer infer_output_shape_func;
    DTypeInfer infer_output_dtype_func;
    FormatInfer infer_output_format_func;

    std::unordered_map<std::string, Process> compute_funcs;
    std::unordered_map<std::string, Process> preprocess_funcs;
    std::unordered_map<std::string, Process> postprocess_funcs;

public:
    CustomOpImpl(const std::string&, uint32_t version);
    PREVENT_COPY_AND_ASSIGN(CustomOpImpl);
    friend CustomOp;
};

CustomOpImpl::CustomOpImpl(const std::string& op_type, uint32_t version)
        : m_version(version), m_op_type(op_type) {
    if (m_version != CURRENT_VERSION) {
        mgb_log_warn(
                "the version of loaded custom op %s is %u, but custom op version "
                "of the system is %u\n",
                op_type.c_str(), m_version, CURRENT_VERSION);
    }

    infer_output_device_func = [](const std::vector<Device>& inputs, const Param&,
                                  std::vector<Device>& outputs) -> void {
        static UnImpleWarnLog log_once("output_device_infer", "device", "x86");
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = inputs.size() > 0 ? inputs[0] : Device("x86");
        }
    };

    infer_output_shape_func = [](const std::vector<Shape>& inputs, const Param&,
                                 std::vector<Shape>& outputs) -> void {
        static UnImpleWarnLog log_once("output_shape_infer", "shape", "{1}");
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = inputs.size() > 0 ? inputs[0] : Shape({1});
        }
    };

    infer_output_dtype_func = [](const std::vector<DType>& inputs, const Param&,
                                 std::vector<DType>& outputs) -> void {
        static UnImpleWarnLog log_once("output_dtype_infer", "dtype", "float32");
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = inputs.size() > 0 ? inputs[0] : DType("float32");
        }
    };

    infer_output_format_func = [](const std::vector<Format>& inputs, const Param&,
                                  std::vector<Format>& outputs) -> void {
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = inputs.size() > 0 ? inputs[0] : Format("default");
        }
    };

    for (const auto& device : Device::legal_devices()) {
        compute_funcs[device] = [](const std::vector<Tensor>&, const Param&,
                                   std::vector<Tensor>& outputs,
                                   const RuntimeArgs&) -> void {
            auto device = outputs[0].device();
            mgb_assert(
                    false,
                    "There is no forward function for your op on device `%s`. "
                    "Please implement this function and register it.",
                    device.str().c_str());
        };
        preprocess_funcs[device] = [](const std::vector<Tensor>&, const Param&,
                                      std::vector<Tensor>&,
                                      const RuntimeArgs&) -> void { return; };
        postprocess_funcs[device] = [](const std::vector<Tensor>&, const Param&,
                                       std::vector<Tensor>&,
                                       const RuntimeArgs&) -> void { return; };
    }
    m_param_infos.set_tag(op_type);
}

CustomOp::CustomOp(const std::string& op_type, uint32_t version)
        : m_impl(new CustomOpImpl(op_type, version), impl_deleter<CustomOpImpl>) {}

#define OpImplRef(raw_ptr) reinterpret_cast<CustomOpImpl*>(raw_ptr)

CustomOp& CustomOp::set_device_infer(DeviceInferFuncPtr func) {
    OpImplRef(m_impl.get())->infer_output_device_func = func;
    return *this;
}

CustomOp& CustomOp::set_shape_infer(ShapeInferFuncPtr func) {
    OpImplRef(m_impl.get())->infer_output_shape_func = func;
    return *this;
}

CustomOp& CustomOp::set_dtype_infer(DTypeInferFuncPtr func) {
    OpImplRef(m_impl.get())->infer_output_dtype_func = func;
    return *this;
}

CustomOp& CustomOp::set_format_infer(FormatInferFuncPtr func) {
    OpImplRef(m_impl.get())->infer_output_format_func = func;
    return *this;
}

CustomOp& CustomOp::set_preprocess(ProcessFuncPtrWithoutRuntimeArgs func) {
    set_preprocess("x86", func);
    return *this;
}

CustomOp& CustomOp::set_preprocess(
        const std::string& device, ProcessFuncPtrWithoutRuntimeArgs func) {
    auto wrap_func = [func](const std::vector<Tensor>& input, const Param& param,
                            std::vector<Tensor>& output, const RuntimeArgs&) -> void {
        return func(input, param, output);
    };

    OpImplRef(m_impl.get())->preprocess_funcs[device] = wrap_func;
    return *this;
}

CustomOp& CustomOp::set_preprocess(ProcessFuncPtr func) {
    set_preprocess("x86", func);
    return *this;
}

CustomOp& CustomOp::set_preprocess(const std::string& device, ProcessFuncPtr func) {
    OpImplRef(m_impl.get())->preprocess_funcs[device] = func;
    return *this;
}

CustomOp& CustomOp::set_postprocess(ProcessFuncPtrWithoutRuntimeArgs func) {
    set_postprocess("x86", func);
    return *this;
}

CustomOp& CustomOp::set_postprocess(
        const std::string& device, ProcessFuncPtrWithoutRuntimeArgs func) {
    auto wrap_func = [func](const std::vector<Tensor>& input, const Param& param,
                            std::vector<Tensor>& output,
                            const RuntimeArgs&) -> void { func(input, param, output); };

    OpImplRef(m_impl.get())->postprocess_funcs[device] = wrap_func;
    return *this;
}

CustomOp& CustomOp::set_postprocess(ProcessFuncPtr func) {
    set_postprocess("x86", func);
    return *this;
}

CustomOp& CustomOp::set_postprocess(const std::string& device, ProcessFuncPtr func) {
    OpImplRef(m_impl.get())->postprocess_funcs[device] = func;
    return *this;
}

CustomOp& CustomOp::set_compute(ProcessFuncPtrWithoutRuntimeArgs func) {
    set_compute("x86", func);
    return *this;
}

CustomOp& CustomOp::set_compute(
        const std::string& device, ProcessFuncPtrWithoutRuntimeArgs func) {
    auto wrap_func = [func](const std::vector<Tensor>& input, const Param& param,
                            std::vector<Tensor>& output,
                            const RuntimeArgs&) -> void { func(input, param, output); };

    OpImplRef(m_impl.get())->compute_funcs[device] = wrap_func;
    return *this;
}

CustomOp& CustomOp::set_compute(ProcessFuncPtr func) {
    set_compute("x86", func);
    return *this;
}

CustomOp& CustomOp::set_compute(const std::string& device, ProcessFuncPtr func) {
    OpImplRef(m_impl.get())->compute_funcs[device] = func;
    return *this;
}

CustomOp& CustomOp::set_description(const std::string& op_desc) {
    OpImplRef(m_impl.get())->m_op_desc = op_desc;
    return *this;
}

CustomOp& CustomOp::add_input(
        const std::string& name, const std::string& desc,
        const std::initializer_list<std::string>& legal_dtypes, int dims,
        const std::string& mem_stgy) {
    auto& ref = OpImplRef(m_impl.get())->m_input_infos;
    for (const auto& input : ref) {
        mgb_assert(input.name() != name, "input %s has been registered", name.c_str());
    }
    ref.emplace_back(name, desc, legal_dtypes, dims, mem_stgy);
    return *this;
}

CustomOp& CustomOp::add_output(
        const std::string& name, const std::string& desc,
        const std::initializer_list<std::string>& legal_dtypes, int dims,
        const std::string& mem_stgy) {
    auto& ref = OpImplRef(m_impl.get())->m_output_infos;
    for (const auto& output : ref) {
        mgb_assert(
                output.name() != name, "output %s has been registered", name.c_str());
    }
    ref.emplace_back(name, desc, legal_dtypes, dims, mem_stgy);
    return *this;
}

CustomOp& CustomOp::add_input(
        const std::string& name, const std::initializer_list<std::string>& legal_dtypes,
        int dims, const std::string& mem_stgy) {
    add_input(name, name, legal_dtypes, dims, mem_stgy);
    return *this;
}

CustomOp& CustomOp::add_output(
        const std::string& name, const std::initializer_list<std::string>& legal_dtypes,
        int dims, const std::string& mem_stgy) {
    add_output(name, name, legal_dtypes, dims, mem_stgy);
    return *this;
}

CustomOp& CustomOp::add_inputs(const size_t& num) {
    size_t cur_inp_num = input_num();
    for (size_t i = cur_inp_num; i < cur_inp_num + num; i++) {
        add_input(op_type() + "_Input_" + std::to_string(i));
    }
    return *this;
}

CustomOp& CustomOp::add_outputs(const size_t& num) {
    size_t cur_oup_num = output_num();
    for (size_t i = cur_oup_num; i < cur_oup_num + num; i++) {
        add_output(op_type() + "_Output_" + std::to_string(i));
    }
    return *this;
}

CustomOp& CustomOp::add_param(const std::string& name, const ParamVal& default_val) {
    add_param(name, name, default_val);
    return *this;
}

CustomOp& CustomOp::add_param(
        const std::string& name, const std::string& desc, const ParamVal& default_val) {
    auto& meta = OpImplRef(m_impl.get())->m_param_infos.meta();
    for (const auto& schema : meta) {
        mgb_assert(
                name != schema.name(), "param %s has been registered\n", name.c_str());
    }
    ParamSchema sch = ParamSchema(name, default_val, desc);
    meta.emplace_back(sch);
    return *this;
}

std::string CustomOp::op_type(void) const {
    return OpImplRef(m_impl.get())->m_op_type;
}

std::string CustomOp::op_desc(void) const {
    return OpImplRef(m_impl.get())->m_op_desc;
}

RunTimeId CustomOp::runtime_id(void) const {
    return (RunTimeId)(this);
}

size_t CustomOp::input_num(void) const {
    return OpImplRef(m_impl.get())->m_input_infos.size();
}

size_t CustomOp::output_num(void) const {
    return OpImplRef(m_impl.get())->m_output_infos.size();
}

std::string CustomOp::str(void) const {
    std::stringstream ss;
    ss << "op name: " << op_type() << "\nop desc: " << op_desc() << "\n\ninputs:\n";
    for (const auto& input : inputs_info()) {
        ss << input.str();
        ss << "\n--------------------\n";
    }
    ss << "\noutputs:\n";
    for (const auto& output : outputs_info()) {
        ss << output.str();
        ss << "\n--------------------\n";
    }
    ss << "\nparams:\n";
    for (const auto& param : param_info().meta()) {
        ss << param.str();
        ss << "\n--------------------\n";
    }
    return ss.str();
}

const ParamInfo& CustomOp::param_info(void) const {
    return OpImplRef(m_impl.get())->m_param_infos;
}

ArgInfo CustomOp::input_info(size_t idx) const {
    return OpImplRef(m_impl.get())->m_input_infos[idx];
}

ArgInfo CustomOp::output_info(size_t idx) const {
    return OpImplRef(m_impl.get())->m_output_infos[idx];
}

const std::vector<ArgInfo>& CustomOp::inputs_info(void) const {
    return OpImplRef(m_impl.get())->m_input_infos;
}

const std::vector<ArgInfo>& CustomOp::outputs_info(void) const {
    return OpImplRef(m_impl.get())->m_output_infos;
}

std::vector<Device> CustomOp::infer_output_device(
        const std::vector<Device>& inputs, const Param& param) const {
    assert_inputs_size_right(inputs);

    std::vector<Device> outputs(output_num());
    OpImplRef(m_impl.get())->infer_output_device_func(inputs, param, outputs);

    assert_outputs_size_right(outputs);
    return outputs;
}

std::vector<Shape> CustomOp::infer_output_shape(
        const std::vector<Shape>& inputs, const Param& param) const {
    assert_inputs_size_right(inputs);
    for (size_t i = 0; i < inputs_info().size(); i++) {
        assert_arg_shape_dim_right(inputs[i], input_info(i));
    }

    std::vector<Shape> outputs(output_num());
    OpImplRef(m_impl.get())->infer_output_shape_func(inputs, param, outputs);
    for (size_t i = 0; i < outputs_info().size(); i++) {
        assert_arg_shape_dim_right(outputs[i], output_info(i));
    }

    assert_outputs_size_right(outputs);
    return outputs;
}

std::vector<DType> CustomOp::infer_output_dtype(
        const std::vector<DType>& inputs, const Param& param) const {
    assert_inputs_size_right(inputs);

    for (size_t i = 0; i < inputs_info().size(); i++) {
        std::unordered_set<std::string> legal_input_dtypes_i = input_info(i).dtypes();
        mgb_assert(
                legal_input_dtypes_i.find(inputs[i].str()) !=
                        legal_input_dtypes_i.end(),
                "dtypes of input: %s(%s) is not allowed, the info of this input "
                "is:\n%s",
                input_info(i).name().c_str(), inputs[i].str().c_str(),
                input_info(i).str().c_str());
    }
    std::vector<DType> outputs(output_num());
    OpImplRef(m_impl.get())->infer_output_dtype_func(inputs, param, outputs);

    for (size_t i = 0; i < outputs_info().size(); i++) {
        std::unordered_set<std::string> legal_output_dtypes_i = output_info(i).dtypes();
        mgb_assert(
                legal_output_dtypes_i.find(outputs[i].str()) !=
                        legal_output_dtypes_i.end(),
                "dtypes of output: %s is %s, the info of this output is:\n%s",
                output_info(i).name().c_str(), outputs[i].str().c_str(),
                output_info(i).str().c_str());
    }

    assert_outputs_size_right(outputs);
    return outputs;
}

std::vector<Format> CustomOp::infer_output_format(
        const std::vector<Format>& inputs, const Param& param) const {
    assert_inputs_size_right(inputs);
    for (size_t i = 0; i < inputs.size(); i++) {
        mgb_assert(
                inputs[i].is_default(), "the tensor format of %s:%s is not default",
                op_type().c_str(), input_info(i).name().c_str());
    }
    std::vector<Format> outputs(output_num());
    OpImplRef(m_impl.get())->infer_output_format_func(inputs, param, outputs);

    for (size_t i = 0; i < outputs.size(); i++) {
        mgb_assert(
                outputs[i].is_default(), "the tensor format of %s:%s is not default",
                op_type().c_str(), output_info(i).name().c_str());
    }

    assert_outputs_size_right(outputs);
    return outputs;
}

void CustomOp::compute(
        const std::vector<Tensor>& inputs, const Param& param,
        std::vector<Tensor>& outputs) const {
    assert_inputs_size_right(inputs);
    assert_outputs_size_right(outputs);
    if (outputs.size() == 0) {
        return;
    }

    Device device = outputs[0].device();
    std::string device_str = device.str();
    for (size_t i = 1; i < outputs.size(); ++i) {
        mgb_assert(
                outputs[i].device().str() == device_str,
                "all output tensors should have the same device attribute");
    }

    // need to add other input/output check
    mgb_assert(
            Device::is_legal(device_str), "unsupported device type: %s",
            device_str.c_str());

    auto preprocess_func = OpImplRef(m_impl.get())->preprocess_funcs[device_str];
    auto forward_func = OpImplRef(m_impl.get())->compute_funcs[device_str];
    auto postprocess_func = OpImplRef(m_impl.get())->postprocess_funcs[device_str];

    RuntimeArgs rt_args(device);

    preprocess_func(inputs, param, outputs, rt_args);
    forward_func(inputs, param, outputs, rt_args);
    postprocess_func(outputs, param, outputs, rt_args);
    assert_outputs_size_right(outputs);
}

}  // namespace custom

#endif
