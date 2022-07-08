#pragma once
#include <gflags/gflags.h>
#include "helpers/common.h"
#include "models/model.h"
#include "option_base.h"

DECLARE_bool(enable_fuse_preprocess);
DECLARE_bool(weight_preprocess);
DECLARE_bool(enable_fuse_conv_bias_nonlinearity);
DECLARE_bool(enable_fuse_conv_bias_with_z);

DECLARE_bool(const_shape);
DECLARE_bool(fake_first);
DECLARE_bool(no_sanity_check);
DECLARE_bool(record_comp_seq);
DECLARE_bool(record_comp_seq2);
DECLARE_bool(disable_mem_opt);
DECLARE_uint64(workspace_limit);

DECLARE_bool(enable_jit);
#if MGB_ENABLE_TENSOR_RT
DECLARE_bool(tensorrt);
DECLARE_string(tensorrt_cache);
#endif
namespace lar {
///////////////////////// fuse_preprocess optimize options //////////////
class FusePreprocessOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

    static void set_valid(bool val) { m_valid = val; }

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    FusePreprocessOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    bool enable_fuse_preprocess;
    static bool m_valid;
    OptionValMap m_option;
};

///////////////////////// weight preprocess optimize options //////////////
class WeightPreprocessOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };
    static void set_valid(bool val) { m_valid = val; };

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    WeightPreprocessOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    bool weight_preprocess;
    static bool m_valid;
    OptionValMap m_option;
};

/////////////// fuse_conv_bias_nonlinearity optimize options ///////////////
class FuseConvBiasNonlinearOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };
    static void set_valid(bool val) { m_valid = val; }

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    FuseConvBiasNonlinearOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    bool enable_fuse_conv_bias_nonlinearity;
    static bool m_valid;
    OptionValMap m_option;
};

///////////////////////// fuse_conv_bias_with_z optimize options //////////////
class FuseConvBiasElemwiseAddOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };
    static void set_valid(bool val) { m_valid = val; }

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    FuseConvBiasElemwiseAddOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};
    std::string m_option_name;
    bool enable_fuse_conv_bias_with_z;
    static bool m_valid;
    OptionValMap m_option;
};

///////////////////////// graph record options ///////////////////////////
class GraphRecordOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

    static void set_valid(bool val) { m_valid = val; }

    OptionValMap* get_option() override { return &m_option; }

    void update() override;

private:
    GraphRecordOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    size_t m_record_comp_seq;
    bool const_shape;
    bool fake_first;
    bool no_sanity_check;
    static bool m_valid;
    OptionValMap m_option;
};

///////////////////////// memory optimize options /////////////////////////
class MemoryOptimizeOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

    void update() override;

private:
    MemoryOptimizeOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    bool disable_mem_opt;
    uint64_t workspace_limit;
};

///////////////////////// other options for optimization /////////////////
class JITOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

    void update() override;

private:
    JITOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    bool enable_jit;
};
///////////////////////// TensorRT options for optimization /////////////////
#if MGB_ENABLE_TENSOR_RT
class TensorRTOption final : public OptionBase {
public:
    static bool is_valid();

    static std::shared_ptr<OptionBase> create_option();

    void config_model(
            RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) override;

    std::string option_name() const override { return m_option_name; };

    void update() override;

private:
    TensorRTOption() = default;
    template <typename ModelImpl>
    void config_model_internel(RuntimeParam&, std::shared_ptr<ModelImpl>){};

    std::string m_option_name;
    bool enable_tensorrt;
    std::string tensorrt_cache;
};
#endif
}  // namespace lar