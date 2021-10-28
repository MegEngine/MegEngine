/**
 * \file lite/load_and_run/src/options/optimize_options.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include "megbrain/gopt/inference.h"
#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_engine_cache.h"
#endif
#include "lite/global.h"
#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"
#include "optimize_options.h"

///////////////////////// fuse and preprocess optimize options ///////////////
namespace lar {
template <>
void FusePreprocessOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (enable_fuse_preprocess) {
            LITE_WARN("enable fuse-preprocess optimization");
            model->get_config().options.fuse_preprocess = true;
        }
    }
}

template <>
void FusePreprocessOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (enable_fuse_preprocess) {
            mgb_log_warn("enable fuse-preprocess optimization");
            graph_option.graph_opt.enable_fuse_preprocess();
        }
    }
}
}  // namespace lar
using namespace lar;

FusePreprocessOption::FusePreprocessOption() {
    m_option_name = "fuse_preprocess";
    enable_fuse_preprocess = FLAGS_enable_fuse_preprocess;
}

bool FusePreprocessOption::is_valid() {
    bool ret = FLAGS_enable_fuse_preprocess;
    return ret;
}

std::shared_ptr<OptionBase> FusePreprocessOption::create_option() {
    static std::shared_ptr<FusePreprocessOption> option(new FusePreprocessOption);
    if (FusePreprocessOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void FusePreprocessOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

///////////////////////// weight preprocess optimize options ///////////////
namespace lar {
template <>
void WeightPreprocessOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (weight_preprocess) {
            LITE_WARN("enable weight-preprocess optimization");
            model->get_config().options.weight_preprocess = true;
            //! FIXME: algo searcher enable weight preprocess for opencl(
            //! implement below has some problem);
            // #if MGB_OPENCL
            //             megdnn::opencl::algo_searcher::AlgoSearcherBase::
            //                     enable_weight_preprocess();
            // #endif
        }
    }
}

template <>
void WeightPreprocessOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (weight_preprocess) {
            mgb_log_warn("enable weight-preprocess optimization");
            graph_option.graph_opt.enable_weight_preprocess();
            //! FIXME: this implemment is not right
            // #if MGB_OPENCL
            //             megdnn::opencl::algo_searcher::AlgoSearcherBase::
            //                     enable_weight_preprocess();
            // #endif
        }
    }
}
}  // namespace lar

WeightPreprocessOption::WeightPreprocessOption() {
    m_option_name = "weight_preprocess";
    weight_preprocess = FLAGS_weight_preprocess;
}

bool WeightPreprocessOption::is_valid() {
    bool ret = FLAGS_weight_preprocess;
    return ret;
}

std::shared_ptr<OptionBase> WeightPreprocessOption::create_option() {
    static std::shared_ptr<WeightPreprocessOption> option(new WeightPreprocessOption);
    if (WeightPreprocessOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void WeightPreprocessOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

///// fuse conv bias and nonlinear activation opr optimize options ////////
namespace lar {
template <>
void FuseConvBiasNonlinearOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    LITE_MARK_USED_VAR(model);
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (enable_fuse_conv_bias_nonlinearity) {
            LITE_THROW("fuse conv+bias+nonlinearity not supported in lite model");
        }
    }
}

template <>
void FuseConvBiasNonlinearOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (enable_fuse_conv_bias_nonlinearity) {
            mgb_log_warn("enable fuse conv+bias+nonlinearity optimization");
            graph_option.graph_opt.enable_fuse_conv_bias_nonlinearity();
        }
    }
}
}  // namespace lar

FuseConvBiasNonlinearOption::FuseConvBiasNonlinearOption() {
    m_option_name = "fuse_conv_bias_nonlinear";
    enable_fuse_conv_bias_nonlinearity = FLAGS_enable_fuse_conv_bias_nonlinearity;
}

bool FuseConvBiasNonlinearOption::is_valid() {
    bool ret = FLAGS_enable_fuse_conv_bias_nonlinearity;
    return ret;
}

std::shared_ptr<OptionBase> FuseConvBiasNonlinearOption::create_option() {
    static std::shared_ptr<FuseConvBiasNonlinearOption> option(
            new FuseConvBiasNonlinearOption);
    if (FuseConvBiasNonlinearOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void FuseConvBiasNonlinearOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

///////////////////////// fuse and preprocess optimize options ///////////////
namespace lar {
template <>
void FuseConvBiasElemwiseAddOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    LITE_MARK_USED_VAR(model);
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (enable_fuse_conv_bias_with_z) {
            LITE_THROW(
                    "fuse conv+bias+z optimization not supported in lite "
                    "model");
        }
    }
}

template <>
void FuseConvBiasElemwiseAddOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (enable_fuse_conv_bias_with_z) {
            mgb_log_warn("enable fuse conv+bias+z optimization");
            graph_option.graph_opt.enable_fuse_conv_bias_with_z();
        }
    }
}
}  // namespace lar

FuseConvBiasElemwiseAddOption::FuseConvBiasElemwiseAddOption() {
    m_option_name = "fuse_conv_bias_z";
    enable_fuse_conv_bias_with_z = FLAGS_enable_fuse_conv_bias_with_z;
}

bool FuseConvBiasElemwiseAddOption::is_valid() {
    bool ret = FLAGS_enable_fuse_conv_bias_with_z;
    return ret;
}

std::shared_ptr<OptionBase> FuseConvBiasElemwiseAddOption::create_option() {
    static std::shared_ptr<FuseConvBiasElemwiseAddOption> option(
            new FuseConvBiasElemwiseAddOption);
    if (FuseConvBiasElemwiseAddOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void FuseConvBiasElemwiseAddOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

///////////////////////// graph retrict options /////////////////////////
namespace lar {
template <>
void GraphRecordOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& config_option = model->get_config().options;
        if (const_shape) {
            LITE_WARN("enable const var shape");
            config_option.const_shape = true;
        }
        if (fake_first) {
            LITE_WARN("enable fake-first optimization");
            config_option.fake_next_exec = true;
        }
        if (no_sanity_check) {
            LITE_WARN("disable var sanity check optimization");
            config_option.var_sanity_check_first_run = false;
        }
        if (m_record_comp_seq == 1) {
            LITE_WARN("set record_comp_seq_level to 1");
        }
        if (m_record_comp_seq == 2) {
            mgb_assert(
                    no_sanity_check,
                    "--no-sanity-check should be set before "
                    "--record-comp-seq2");
            LITE_WARN("set record_comp_seq_level to 2");
        }
        config_option.comp_node_seq_record_level = m_record_comp_seq;
    }
}

template <>
void GraphRecordOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (const_shape) {
            mgb_log_warn("enable const var shape");
            model->get_mdl_config().const_var_shape = true;
        }
        if (fake_first) {
            mgb_log_warn("enable fake-first optimization");
            graph_option.fake_next_exec = true;
        }
        if (no_sanity_check) {
            mgb_log_warn("disable var sanity check optimization");
            graph_option.var_sanity_check_first_run = false;
        }
        if (m_record_comp_seq == 1) {
            mgb_log_warn("set record_comp_seq_level to 1");
        }
        if (m_record_comp_seq == 2) {
            mgb_assert(
                    no_sanity_check && !fake_first,
                    "--no-sanity-check should be set before "
                    "--record-comp-seq2 and --fake-first should not be set");
            mgb_log_warn("set record_comp_seq_level to 2");
        }
        graph_option.comp_node_seq_record_level = m_record_comp_seq;
    }
}
}  // namespace lar

GraphRecordOption::GraphRecordOption() {
    m_option_name = "graph_record";
    m_record_comp_seq = 0;
    const_shape = FLAGS_const_shape;
    fake_first = FLAGS_fake_first;
    no_sanity_check = FLAGS_no_sanity_check;
    if (FLAGS_record_comp_seq) {
        m_record_comp_seq = 1;
    }
    if (FLAGS_record_comp_seq2) {
        m_record_comp_seq = 2;
    }
}

bool GraphRecordOption::is_valid() {
    bool ret = FLAGS_const_shape;
    ret = ret || FLAGS_fake_first;
    ret = ret || FLAGS_no_sanity_check;
    ret = ret || FLAGS_record_comp_seq;
    ret = ret || FLAGS_record_comp_seq2;
    return ret;
}

std::shared_ptr<OptionBase> GraphRecordOption::create_option() {
    static std::shared_ptr<GraphRecordOption> option(new GraphRecordOption);
    if (GraphRecordOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void GraphRecordOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}
///////////////////////// graph retrict options /////////////////////////
namespace lar {
template <>
void MemoryOptimizeOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    LITE_MARK_USED_VAR(model);
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (disable_mem_opt) {
            LITE_THROW("lite model don't support disable memory optimization");
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_LOAD) {
        if (workspace_limit != SIZE_MAX) {
            LITE_WARN("set workspace limit to %ld", workspace_limit);
            lite::Runtime::set_network_algo_workspace_limit(
                    model->get_lite_network(), workspace_limit);
        }
    }
}

template <>
void MemoryOptimizeOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (disable_mem_opt) {
            mgb_log_warn("disable memory optimization");
            graph_option.seq_opt.enable_mem_plan_opt = false;
            graph_option.seq_opt.enable_mem_reuse_alloc = false;
        }
        if (workspace_limit < SIZE_MAX) {
            mgb_log_warn("set workspace limit to %ld", workspace_limit);
            auto output_spec = model->get_output_spec();
            mgb::SymbolVarArray vars;
            for (auto i : output_spec) {
                vars.push_back(i.first);
            }
            mgb::gopt::set_opr_algo_workspace_limit_inplace(vars, workspace_limit);
        }
    }
}
}  // namespace lar

MemoryOptimizeOption::MemoryOptimizeOption() {
    m_option_name = "memory_optimize";
    disable_mem_opt = FLAGS_disable_mem_opt;
    workspace_limit = FLAGS_workspace_limit;
}

bool MemoryOptimizeOption::is_valid() {
    bool ret = FLAGS_disable_mem_opt;
    ret = ret || FLAGS_workspace_limit < SIZE_MAX;
    return ret;
}

std::shared_ptr<OptionBase> MemoryOptimizeOption::create_option() {
    static std::shared_ptr<MemoryOptimizeOption> option(new MemoryOptimizeOption);
    if (MemoryOptimizeOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void MemoryOptimizeOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

///////////////////////// other options for optimization /////////////////
namespace lar {
template <>
void JITOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& config_option = model->get_config().options;
        if (enable_jit) {
            LITE_WARN("enable JIT (level 1)");
            config_option.jit_level = 1;
        }
    }
}

template <>
void JITOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (enable_jit) {
            mgb_log_warn("enable JIT (level 1)");
            graph_option.graph_opt.jit = 1;
        }
    }
}
}  // namespace lar
JITOption::JITOption() {
    m_option_name = "JIT";
    enable_jit = FLAGS_enable_jit;
}

bool JITOption::is_valid() {
    bool ret = FLAGS_enable_jit;
    return ret;
}

std::shared_ptr<OptionBase> JITOption::create_option() {
    static std::shared_ptr<JITOption> option(new JITOption);
    if (JITOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void JITOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}
///////////////////////// other options for optimization /////////////////
#if MGB_ENABLE_TENSOR_RT
namespace lar {
template <>
void TensorRTOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (!tensorrt_cache.empty()) {
            LITE_WARN("set tensorrt cache as %s", tensorrt_cache.c_str());
            lite::set_tensor_rt_cache(tensorrt_cache);
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_LOAD) {
        if (enable_tensorrt) {
            LITE_WARN("enable TensorRT");
            lite::Runtime::use_tensorrt(model->get_lite_network());
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_RUNNING) {
        if (!tensorrt_cache.empty()) {
            lite::dump_tensor_rt_cache();
        }
    }
}

template <>
void TensorRTOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        auto&& graph_option = model->get_mdl_config().comp_graph->options();
        if (enable_tensorrt) {
            mgb_log_warn("using tensorRT");
            graph_option.graph_opt.tensorrt = true;
        }
        if (!tensorrt_cache.empty()) {
            mgb_log_warn("use tensorrt cache: %s", tensorrt_cache.c_str());
            mgb::TensorRTEngineCache::enable_engine_cache(true);
            mgb::TensorRTEngineCache::set_impl(
                    std::make_shared<mgb::TensorRTEngineCacheIO>(
                            tensorrt_cache.c_str()));
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_RUNNING) {
        if (!tensorrt_cache.empty()) {
            if (mgb::TensorRTEngineCache::enable_engine_cache()) {
                mgb::TensorRTEngineCache::inst().dump_cache();
            }
        }
    }
}
}  // namespace lar

TensorRTOption::TensorRTOption() {
    m_option_name = "tensorRT";
    enable_tensorrt = FLAGS_tensorrt;
    tensorrt_cache = FLAGS_tensorrt_cache;
}

bool TensorRTOption::is_valid() {
    bool ret = FLAGS_tensorrt;
    ret = ret || !FLAGS_tensorrt_cache.empty();
    return ret;
}

std::shared_ptr<OptionBase> TensorRTOption::create_option() {
    static std::shared_ptr<TensorRTOption> option(new TensorRTOption);
    if (TensorRTOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void TensorRTOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}
#endif
///////////////////////// fuse and preprocess optimize options ///////////////
DEFINE_bool(
        enable_fuse_preprocess, false,
        "Fusion astype | pad_channel | dimshuffle and etc opr from h2d opr");
DEFINE_bool(
        weight_preprocess, false,
        "Execute operators with weight preprocess, which can optimize the "
        "operator execution time with algo of winograd, im2col ,etc., but "
        "it may consume more memory.");
DEFINE_bool(
        enable_fuse_conv_bias_nonlinearity, false,
        "whether to fuse conv+bias+nonlinearity");
DEFINE_bool(
        enable_fuse_conv_bias_with_z, false,
        "fuse conv，bias (elemwise add)，z(elemwise add) into one opr "
        "(only support on GPU)");

///////////////////////// graph retrict options /////////////////////////
DEFINE_bool(
        const_shape, false,
        "set const_var_shape to reduce memory usage, since some static "
        "inference data structures can be omitted");
DEFINE_bool(
        fake_first, false,
        "Enable fake exec for the first run. In fake exec mode, some "
        "initialization job would be done, but no actual computing is "
        "performed.");
DEFINE_bool(no_sanity_check, false, "Disable var sanity check on the first run");
DEFINE_bool(
        record_comp_seq, false,
        "Record the computing sequence, in level 1 . It reduces overhead of API"
        "calls of some asynchronous computing devices");
DEFINE_bool(
        record_comp_seq2, false,
        "Record the computing sequence, in level 2, the computing graph can be"
        "destructed to reduce memory usage");
DEFINE_bool(disable_mem_opt, false, "disable memory optimization!!");
DEFINE_uint64(workspace_limit, SIZE_MAX, "set workspace upbound limit");

///////////////////////// other options for optimization /////////////////
DEFINE_bool(
        enable_jit, false,
        " Execute supported operators with JIT(now only support NVRTC). "
        "Can only be used on Nvidia GPUs");
#if MGB_ENABLE_ANDROID_NN
DEFINE_bool(
        android_nn, false,
        "Execute supported operators with Android NN. Can only be used "
        "with --cpu.");
#endif
#if MGB_ENABLE_TENSOR_RT
DEFINE_bool(
        tensorrt, false,
        " Execute supported operators with TensorRT. Can only be used on "
        "Nvidia GPUs,i.e. comp node is xpu or gpu.");
DEFINE_string(
        tensorrt_cache, "",
        "Set the TensorRT engine cache path for serialized prebuilt "
        "ICudaEngine");
#endif
REGIST_OPTION_CREATOR(fuse_preprocess, lar::FusePreprocessOption::create_option);
REGIST_OPTION_CREATOR(weight_preprocess, lar::WeightPreprocessOption::create_option);
REGIST_OPTION_CREATOR(
        fuse_conv_bias_nonlinear, lar::FuseConvBiasNonlinearOption::create_option);
REGIST_OPTION_CREATOR(
        fuse_conv_bias_z, lar::FuseConvBiasElemwiseAddOption::create_option);
REGIST_OPTION_CREATOR(graph_record, lar::GraphRecordOption::create_option);
REGIST_OPTION_CREATOR(memory_optimize, lar::MemoryOptimizeOption::create_option);
REGIST_OPTION_CREATOR(JIT, lar::JITOption::create_option);
#if MGB_ENABLE_TENSOR_RT
REGIST_OPTION_CREATOR(tensorRT, lar::TensorRTOption::create_option);
#endif