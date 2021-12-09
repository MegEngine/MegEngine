/**
 * \file lite/load_and_run/src/options/layout_trans_options.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
#include "layout_trans_options.h"
#include <gflags/gflags.h>
#include "megbrain/serialization/serializer.h"
#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"
namespace lar {

template <>
void GoptLayoutOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> /* model */) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        LITE_THROW("lite model don't support global graph optimization");
    }
}

template <>
void GoptLayoutOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::GLOBAL_OPTIMIZATION) {
        if (layout_transform) {
            auto&& load_result = model->get_mdl_load_result();
            load_result.output_var_list = mgb::gopt::layout_transform(
                    load_result.output_var_list, layout_transform_target);

            if (!layout_transform_dump_file.empty()) {
                auto out_file = mgb::serialization::OutputFile::make_fs(
                        layout_transform_dump_file.c_str(), 'w');
                auto testcase_num = model->get_testcase_num();

                if (testcase_num) {
                    const char* magic = "mgbtest0";
                    constexpr size_t len = sizeof(magic);
                    out_file->write(magic, len);
                    out_file->write(&testcase_num, sizeof(testcase_num));
                }

                using DumpConfig = mgb::serialization::GraphDumper::DumpConfig;
                DumpConfig config{1, false, false};
                auto dumper = model->get_dumper(std::move(out_file));
                dumper->dump(load_result.output_var_list, config);

                if (testcase_num) {
                    auto input_file = model->get_loader()->reset_file();
                    auto current_offset = input_file->tell();
                    auto loader = model->reset_loader(std::move(input_file));
                    auto testcase = loader->load(model->get_mdl_config(), false);
                    mgb::serialization::GraphDumper::DumpConfig config{1, false, false};
                    for (size_t i = 0; i < testcase_num; ++i) {
                        auto casefile = mgb::serialization::OutputFile::make_fs(
                                layout_transform_dump_file.c_str(), 'a');
                        auto casedumper = model->get_dumper(std::move(casefile));
                        casedumper->dump(testcase.output_var_list, config);
                        if (i != testcase_num - 1) {
                            loader = model->reset_loader();
                            testcase = loader->load(model->get_mdl_config(), false);
                        }
                    }
                    input_file = model->get_loader()->reset_file();
                    input_file->rewind();
                    input_file->skip(current_offset);
                    model->reset_loader(std::move(input_file));
                }
            }
        }
    }
}

}  // namespace lar

using namespace lar;

GoptLayoutOption::GoptLayoutOption() {
    m_option_name = "gopt_layout";
    if (FLAGS_layout_transform != "cuda" && FLAGS_layout_transform != "cpu" &&
        FLAGS_layout_transform != "opencl") {
        layout_transform = false;
        layout_transform_target = mgb::gopt::GraphTuningOptions::Target::UNSPEC;

    } else {
        layout_transform = true;
        if (FLAGS_layout_transform == "cuda") {
            layout_transform_target = mgb::gopt::GraphTuningOptions::Target::CUDA;
        } else if (FLAGS_layout_transform == "cpu") {
            layout_transform_target = mgb::gopt::GraphTuningOptions::Target::CPU;
        } else if (FLAGS_layout_transform == "opencl") {
            layout_transform_target = mgb::gopt::GraphTuningOptions::Target::OPENCL;
        }
    }
    layout_transform_dump_file = FLAGS_layout_transform_dump;
}

bool GoptLayoutOption::is_valid() {
    bool ret = false;
    if (!FLAGS_layout_transform.empty()) {
        if (FLAGS_layout_transform != "cuda" && FLAGS_layout_transform != "cpu" &&
            FLAGS_layout_transform != "opencl") {
            mgb_assert(
                    false,
                    "unsupported target(got:%s) for global layout "
                    "transform",
                    FLAGS_layout_transform.c_str());
            ret = false;
        } else {
            ret = true;
        }
    }
    ret = ret || !FLAGS_layout_transform_dump.empty();
    return ret;
}

std::shared_ptr<OptionBase> GoptLayoutOption::create_option() {
    static std::shared_ptr<GoptLayoutOption> option(new GoptLayoutOption);
    if (GoptLayoutOption::is_valid()) {
        return std::static_pointer_cast<OptionBase>(option);
    } else {
        return nullptr;
    }
}

void GoptLayoutOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    CONFIG_MODEL_FUN;
}

DEFINE_string(
        layout_transform, "",
        "Enable global layout transform optimization for computing graph. User should "
        "specify the device target for the optimization, and a series of passes will "
        "be applied on the computing graph. The passes will benchmark the elapsed time "
        "of operators on different tensor layouts, and select fastest implementation "
        "for the operators. The optimization process will take some time. The default "
        "target is unspec, which all the available for operators will be profiled. So "
        "the optimize time will be longer.");
DEFINE_string(
        layout_transform_dump, "",
        "The computing graph after global layout transform will be dumped to the given "
        "file path.");

REGIST_OPTION_CREATOR(gopt_layout, lar::GoptLayoutOption::create_option);
