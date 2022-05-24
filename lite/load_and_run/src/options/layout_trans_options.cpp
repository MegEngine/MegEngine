#include "layout_trans_options.h"
#include <gflags/gflags.h>
#include "megbrain/serialization/serializer.h"
#include "misc.h"
#include "models/model_lite.h"
#include "models/model_mdl.h"
namespace lar {

template <>
void GoptLayoutOption::config_model_internel<ModelLite>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelLite> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        if (m_layout_transform) {
            LITE_WARN("using global layout transform optimization\n");
            if (m_layout_transform_target ==
                mgb::gopt::GraphTuningOptions::Target::CPU) {
                model->get_config().device_type = LiteDeviceType::LITE_CPU;
            }
#if LITE_WITH_CUDA
            else if (
                    m_layout_transform_target ==
                    mgb::gopt::GraphTuningOptions::Target::CUDA) {
                model->get_config().device_type = LiteDeviceType::LITE_CUDA;
            }
#endif
            model->set_layout_transform(true);
        }
    } else if (runtime_param.stage == RunStage::GLOBAL_OPTIMIZATION) {
        if (m_layout_transform) {
            auto&& network = model->get_lite_network();
            if (!m_layout_transform_dump_file.empty()) {
                lite::Runtime::dump_layout_transform_model(
                        network, m_layout_transform_dump_file);
            }
        }
    }
}

template <>
void GoptLayoutOption::config_model_internel<ModelMdl>(
        RuntimeParam& runtime_param, std::shared_ptr<ModelMdl> model) {
    if (runtime_param.stage == RunStage::GLOBAL_OPTIMIZATION) {
        if (m_layout_transform) {
            mgb_log_warn("using global layout transform optimization\n");
            auto&& load_result = model->get_mdl_load_result();
            for (auto&& item : load_result.output_var_list) {
                if (item.shape()[0] > 1) {
                    mgb_log_warn(
                            " model may be dumped with multi batch and will cost lots "
                            "of time to profile during global layout transform!!!\n");
                }
            }
            //! update output varlist when input shape maybe change(some pass excution
            //! time depends on the shape of init input)
            mgb::thin_hash_table::ThinHashMap<mgb::cg::SymbolVar, mgb::cg::SymbolVar>
                    varmap;
            mgb::cg::DepOprIter dep([&](mgb::cg::OperatorNodeBase* opr) {
                if (auto h2d = opr->try_cast_final<mgb::opr::Host2DeviceCopy>()) {
                    auto param = h2d->param();
                    mgb::TensorShape new_shape = h2d->host_data()->shape();
                    std::shared_ptr<mgb::HostTensorND> new_tensor =
                            std::make_shared<mgb::HostTensorND>(
                                    h2d->host_data()->comp_node(), new_shape,
                                    h2d->host_data()->dtype());
                    new_tensor->only_reset_raw_storage(h2d->host_data()->storage());
                    auto h2d_opr = mgb::opr::Host2DeviceCopy::make(
                            *h2d->owner_graph(), new_tensor, param, h2d->config());
                    varmap[h2d->output(0)] = h2d_opr;
                }
            });

            for (auto&& i : load_result.output_var_list)
                dep.add(i);

            if (!varmap.empty()) {
                auto output_vars =
                        mgb::cg::replace_vars(load_result.output_var_list, varmap);
                for (size_t i = 0; i < load_result.output_var_list.size(); ++i) {
                    output_vars[i].rename(
                            load_result.output_var_list[i].node()->name());
                }
                load_result.output_var_list = output_vars;
            }

            load_result.output_var_list = mgb::gopt::layout_transform(
                    load_result.output_var_list, m_layout_transform_target);

            if (!m_layout_transform_dump_file.empty()) {
                auto out_file = mgb::serialization::OutputFile::make_fs(
                        m_layout_transform_dump_file.c_str(), 'w');
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
                                m_layout_transform_dump_file.c_str(), 'a');
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
bool GoptLayoutOption::m_valid;
GoptLayoutOption::GoptLayoutOption() {
    m_option_name = "gopt_layout";
    if (FLAGS_layout_transform != "cpu"
#if LITE_WITH_CUDA
        && FLAGS_layout_transform != "cuda"
#endif
    ) {
        m_layout_transform = false;
        m_layout_transform_target = mgb::gopt::GraphTuningOptions::Target::UNSPEC;

    } else {
        m_layout_transform = true;

        if (FLAGS_layout_transform == "cpu") {
            m_layout_transform_target = mgb::gopt::GraphTuningOptions::Target::CPU;
        }
#if LITE_WITH_CUDA
        else if (FLAGS_layout_transform == "cuda") {
            m_layout_transform_target = mgb::gopt::GraphTuningOptions::Target::CUDA;
        }
#endif
    }
    m_layout_transform_dump_file = FLAGS_layout_transform_dump;

    m_option = {
            {"layout_transform", lar::String::make("")},
    };
    std::static_pointer_cast<lar::String>(m_option["layout_transform"])
            ->set_value(FLAGS_layout_transform);
}

bool GoptLayoutOption::is_valid() {
    bool ret = false;
    if (!FLAGS_layout_transform.empty()) {
        if (FLAGS_layout_transform != "cpu"
#if LITE_WITH_CUDA
            && FLAGS_layout_transform != "cuda"
#endif
        ) {
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
    return ret || m_valid;
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
    auto value = std::static_pointer_cast<lar::String>(m_option["layout_transform"])
                         ->get_value();
    if (value.empty()) {
        return;
    }
    if (value == "cpu") {
        m_layout_transform = true;
        m_layout_transform_target = mgb::gopt::GraphTuningOptions::Target::CPU;
    }
#if LITE_WITH_CUDA
    else if (value == "cuda") {
        m_layout_transform = true;
        m_layout_transform_target = mgb::gopt::GraphTuningOptions::Target::CUDA;
    }
#endif
    else {
        mgb_throw(
                mgb::AssertionError, "invalid options of global layout transform %s",
                value.c_str());
    }
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
REGIST_OPTION_VALIDATER(gopt_layout, lar::GoptLayoutOption::set_valid);