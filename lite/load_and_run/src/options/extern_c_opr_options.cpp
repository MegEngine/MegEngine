#include "extern_c_opr_options.h"
#include "megbrain/utils/debug.h"
#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"

namespace lar {
template <>
void COprLibOption::config_model_internel(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    MGB_MARK_USED_VAR(model);
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (!lib_path.empty()) {
            lite::set_loader_lib_path(lib_path);
        }
        if (c_opr_args.is_run_c_opr_with_param) {
            LITE_THROW(
                    "lite model dont't support run with external c opr "
                    "parmeter");
        }
    }
}
template <>
void COprLibOption::config_model_internel(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (!lib_path.empty()) {
            load_lib();
        }
        if (c_opr_args.is_run_c_opr_with_param) {
            mgb_assert(
                    c_opr_args.is_run_c_opr &&
                            c_opr_args.copr_param_device_ptr_malloc &&
                            c_opr_args.copr_param_device_ptr_free &&
                            c_opr_args.copr_param_device_ptr_h2d,
                    "--c-opr-lib-with-param need config with --c-opr-lib, also "
                    "extern c opr loader need implemente "
                    "copr_param_device_ptr_malloc, copr_param_device_ptr_free "
                    "and copr_param_device_ptr_h2d symbols");
        }
    } else if (runtime_param.stage == RunStage::MODEL_RUNNING) {
        if (model->get_testcase_num() && c_opr_args.is_run_c_opr_with_param) {
            init_extern_param(model);
            set_Copr_IO(model);
        }
    } else if (runtime_param.stage == RunStage::AFTER_RUNNING_ITER) {
        if (model->get_testcase_num() && c_opr_args.is_run_c_opr_with_param) {
            c_opr_args.copr_param_device_ptr_free(c_opr_param.get());
            free(c_opr_param->input);
        }
    }
}
}  // namespace lar

using namespace lar;

MGBDType COprLibOption::dtype_cpp2c(megdnn::DType dtype) {
    switch (dtype.enumv()) {
        case megdnn::DTypeEnum::Float32:
            return MGB_DTYPE_FLOAT32;
        case megdnn::DTypeEnum::Int32:
            return MGB_DTYPE_INT32;
        case megdnn::DTypeEnum::Int16:
            return MGB_DTYPE_INT16;
        case megdnn::DTypeEnum::Uint8:
            return MGB_DTYPE_UINT8;
#if !MEGDNN_DISABLE_FLOAT16
        case megdnn::DTypeEnum::Float16:
            return MGB_DTYPE_FLOAT16;
#endif
        default:
            mgb_throw(
                    mgb::InternalError, "unsupported dtype for extern C API: %s",
                    dtype.name());
    }
}

void COprLibOption::tensor_shape_to_c(
        const megdnn::TensorShape& shape, MGBTensorShape& mgb_shape) {
    mgb_assert(
            shape.ndim <= MGB_TENSOR_MAX_NDIM, "shape ndim too large: %zu", shape.ndim);
    mgb_shape.ndim = shape.ndim;
    for (size_t i = 0; i < shape.ndim; ++i) {
        mgb_shape.shape[i] = shape[i];
    }
}

void COprLibOption::init_extern_param(std::shared_ptr<ModelBase> model_ptr) {
    auto model = std::static_pointer_cast<ModelMdl>(model_ptr);
    auto inp_tensors = model->get_test_input();

    c_opr_param = std::make_shared<ExternCOprParam>();
    memset(c_opr_param.get(), 0, sizeof(ExternCOprParam));

    //! we just test input on npu case, do not test output on
    //! npu case, so we just init input shape and type

    c_opr_param->nr_input = inp_tensors.size();
    c_opr_param->input = (ExternDeviceTensor*)malloc(
            sizeof(ExternDeviceTensor) * inp_tensors.size());
    memset(c_opr_param->input, 0, sizeof(ExternDeviceTensor) * inp_tensors.size());

    //! init input ExternDeviceTensor shape and dtype
    for (size_t input_idx = 0; input_idx < inp_tensors.size(); input_idx++) {
        auto& mgb_tensor_layout = c_opr_param->input[input_idx].layout;
        auto host_tensor_nd_p = inp_tensors[input_idx].second;
        mgb_tensor_layout.dtype = dtype_cpp2c(host_tensor_nd_p->dtype());
        tensor_shape_to_c(
                inp_tensors[input_idx].second->shape(), mgb_tensor_layout.shape);
    }
    c_opr_param->nr_output = 0;

    //! now call copr_param_device_ptr_malloc to malloc
    //! device_ptr
    c_opr_args.copr_param_device_ptr_malloc(c_opr_param.get());
}

void COprLibOption::load_lib() {
    auto handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    mgb_assert(
            handle, "failed to open c opr lib %s:\n errmsg: %s", lib_path.c_str(),
            dlerror());

    const char* entry = m_c_opr_init_func.c_str();
    auto func = dlsym(handle, entry);
    mgb_assert(
            func,
            "can not resolve %s: %s, please use '--c-opr-init-interface' to set the "
            "init API of your loader",
            entry, dlerror());
    typedef void (*entry_f_t)(void*);
    reinterpret_cast<entry_f_t>(func)(
            reinterpret_cast<void*>(&mgb_get_extern_c_opr_api_versioned));
    printf("loaded C opr library: %s\n", lib_path.c_str());
    entry = "copr_param_device_ptr_malloc";
    func = dlsym(handle, entry);
    if (func) {
        printf("get %s from: %s\n", entry, lib_path.c_str());
        c_opr_args.copr_param_device_ptr_malloc =
                reinterpret_cast<COprArgs::COPR_PARAM_DEVICE_PTR_MEM_T>(func);
    }

    entry = "copr_param_device_ptr_free";
    func = dlsym(handle, entry);
    if (func) {
        printf("get %s from: %s\n", entry, lib_path.c_str());
        c_opr_args.copr_param_device_ptr_free =
                reinterpret_cast<COprArgs::COPR_PARAM_DEVICE_PTR_MEM_T>(func);
    }

    entry = "copr_param_device_ptr_h2d";
    func = dlsym(handle, entry);
    if (func) {
        printf("get %s from: %s\n", entry, lib_path.c_str());
        c_opr_args.copr_param_device_ptr_h2d =
                reinterpret_cast<COprArgs::COPR_PARAM_DEVICE_PTR_H2D_T>(func);
    }
}

void COprLibOption::set_Copr_IO(std::shared_ptr<ModelBase> model_ptr) {
    auto model = std::static_pointer_cast<ModelMdl>(model_ptr);
    auto inp_tensors = model->get_test_input();
    auto loader = model->reset_loader();
    auto testcase = loader->load(model->get_mdl_config(), false);
    mgb_assert(testcase.output_var_list.size() == inp_tensors.size());
    for (size_t i = 0; i < inp_tensors.size(); ++i) {
        auto&& opr = testcase.output_var_list[i]
                             .node()
                             ->owner_opr()
                             ->cast_final_safe<mgb::opr::SharedDeviceTensor>();
        c_opr_args.copr_param_device_ptr_h2d(
                c_opr_param.get(), opr.dev_data()->raw_ptr(), i);
    }

    //! now config c opr dynamic param
    config_extern_c_opr_dynamic_param(model->get_async_func(), c_opr_param);
}

COprLibOption::COprLibOption() {
    m_option_name = "c_opr_lib";
    lib_path = FLAGS_c_opr_lib;
    c_opr_args.is_run_c_opr = !lib_path.empty();
    c_opr_args.is_run_c_opr_with_param = FLAGS_c_opr_lib_with_param;
    m_c_opr_init_func = FLAGS_c_opr_init_interface;
}

bool COprLibOption::is_valid() {
    return !FLAGS_c_opr_lib.empty() || FLAGS_c_opr_lib_with_param;
}

std::shared_ptr<OptionBase> COprLibOption::create_option() {
    static std::shared_ptr<COprLibOption> option(new COprLibOption);
    if (COprLibOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void COprLibOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}
DEFINE_string(
        c_opr_lib, "",
        "Load external operator library. It must implement "
        "MGB_C_OPR_INIT_FUNC_STR as the entry point");
DEFINE_bool(
        c_opr_lib_with_param, false,
        "Run c opr lib with param, use to benchmark speed and check result, "
        "need c opr loader implemente `copr_param_device_ptr_malloc, "
        "copr_param_device_ptr_free and copr_param_device_ptr_h2d' symbols");
DEFINE_string(
        c_opr_init_interface, MGB_C_OPR_INIT_FUNC_STR,
        "set the C_OPR_INIT_FUNC interface when running");
REGIST_OPTION_CREATOR(c_opr_lib, lar::COprLibOption::create_option);
