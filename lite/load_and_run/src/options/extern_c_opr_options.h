#pragma once
#include <gflags/gflags.h>
#include "megbrain/graph/extern_copr_api.h"
#include "models/model.h"
#include "option_base.h"

DECLARE_bool(c_opr_lib_with_param);
DECLARE_string(c_opr_lib);
DECLARE_string(c_opr_init_interface);
namespace lar {

struct COprArgs {
    //! for run c opr
    bool is_run_c_opr = false;
    bool is_run_c_opr_with_param = false;
    typedef void (*COPR_PARAM_DEVICE_PTR_MEM_T)(ExternCOprParam* param);
    typedef void (*COPR_PARAM_DEVICE_PTR_H2D_T)(
            ExternCOprParam* param, void* host_ptr, size_t extern_device_tensor_id);
    COPR_PARAM_DEVICE_PTR_MEM_T copr_param_device_ptr_malloc = nullptr;
    COPR_PARAM_DEVICE_PTR_MEM_T copr_param_device_ptr_free = nullptr;
    COPR_PARAM_DEVICE_PTR_H2D_T copr_param_device_ptr_h2d = nullptr;
};

class COprLibOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

private:
    COprLibOption();
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    void load_lib();

    MGBDType dtype_cpp2c(megdnn::DType dtype);

    void tensor_shape_to_c(const megdnn::TensorShape& shape, MGBTensorShape& mgb_shape);

    void init_extern_param(std::shared_ptr<ModelBase> model);

    void set_Copr_IO(std::shared_ptr<ModelBase> model);

    std::string m_option_name;
    COprArgs c_opr_args;
    std::string lib_path;
    std::shared_ptr<ExternCOprParam> c_opr_param;
    std::string m_c_opr_init_func;
};
}  // namespace lar