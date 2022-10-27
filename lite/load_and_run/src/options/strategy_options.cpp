#include "strategy_options.h"
#include "megbrain/utils/persistent_cache.h"
#include "megdnn/algorithm_cache.h"
#include "models/model_mdl.h"
using namespace lar;

DECLARE_bool(c_opr_lib_with_param);
DECLARE_bool(fitting);
void StrategyOption::update() {
    m_option_name = "run_strategy";
    warmup_iter = FLAGS_fitting ? 3 : FLAGS_warmup_iter;
    run_iter = FLAGS_fitting ? 10 : FLAGS_iter;
    threads = FLAGS_fitting ? 1 : FLAGS_thread;
    m_option = {
            {"iter", lar::NumberInt32::make(run_iter)},
            {"warmup_iter", lar::NumberInt32::make(warmup_iter)},
            {"thread", lar::NumberInt32::make(threads)},

    };
}

std::shared_ptr<OptionBase> StrategyOption::create_option() {
    static std::shared_ptr<StrategyOption> option(new StrategyOption);
    option->update();
    return std::static_pointer_cast<OptionBase>(option);
}

void StrategyOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    if (runtime_param.stage == RunStage::BEFORE_MODEL_LOAD) {
        model->set_shared_mem(FLAGS_share_param_mem);
        runtime_param.warmup_iter = warmup_iter;
        runtime_param.run_iter = run_iter;
        runtime_param.threads = threads;
        runtime_param.testcase_num = 1;
    } else if (runtime_param.stage == RunStage::UPDATE_IO) {
        if (model->type() == ModelType::MEGDL_MODEL) {
            auto model_ptr = std::static_pointer_cast<ModelMdl>(model);
            //! update input and output related varnode
            model_ptr->update_io();
        }
    } else if (runtime_param.stage == RunStage::BEFORE_OUTSPEC_SET) {
        if (model->type() == ModelType::MEGDL_MODEL) {
            auto model_ptr = std::static_pointer_cast<ModelMdl>(model);
            auto num = model_ptr->get_testcase_num();
            if (num != 0)
                runtime_param.testcase_num = num;
            //! make output specification
            model_ptr->make_output_spec();
        }
    } else if (runtime_param.stage == RunStage::AFTER_MODEL_RUNNING) {
        mgb_log_debug("clear cache");
#if LITE_WITH_OPENCL
        megcoreOpenCLClearGlobalData();
#else
        megdnn::AlgorithmCache::instance().clear();
        //! WARNING:this is used to clean the profile result cache
        //! auto profile_cache = mgb::PersistentCache::inst().get_cache();
#endif
    }
}

void TestcaseOption::update() {
    m_option_name = "run_testcase";
}

std::shared_ptr<OptionBase> TestcaseOption::create_option() {
    static std::shared_ptr<TestcaseOption> option(new TestcaseOption);
    option->update();
    return std::static_pointer_cast<OptionBase>(option);
}

void TestcaseOption::config_model(
        RuntimeParam& runtime_param, std::shared_ptr<ModelBase> model) {
    if (model->type() == ModelType::MEGDL_MODEL) {
        auto model_ptr = std::static_pointer_cast<ModelMdl>(model);
        if (model_ptr->get_testcase_num() && !FLAGS_c_opr_lib_with_param) {
            if (runtime_param.stage == RunStage::AFTER_MODEL_LOAD) {
                auto input_tensor = model_ptr->get_test_input();
                auto input_file = model_ptr->get_loader()->reset_file();
                auto current_offset = input_file->tell();
                auto loader = model_ptr->reset_loader(std::move(input_file));
                auto testcase = loader->load(model_ptr->get_mdl_config(), false);
                mgb_assert(testcase.output_var_list.size() == input_tensor.size());
                for (size_t i = 0; i < input_tensor.size(); ++i) {
                    auto&& opr =
                            testcase.output_var_list[i]
                                    .node()
                                    ->owner_opr()
                                    ->cast_final_safe<mgb::opr::SharedDeviceTensor>();
                    input_tensor[i].second->copy_from(
                            mgb::HostTensorND::make_proxy(*opr.dev_data()));
                }
                input_file = model_ptr->get_loader()->reset_file();
                input_file->rewind();
                input_file->skip(current_offset);
                model_ptr->reset_loader(std::move(input_file));
            }
        }
    }
}

DEFINE_int32(iter, 10, "iteration number for run model");

DEFINE_int32(warmup_iter, 1, "iteration number for warm up model before run");

DEFINE_int32(
        thread, 1,
        "thread number for run model while <thread> is supported( NOTE: "
        "this is not a mapper device setting just for load and run)");

DEFINE_bool(share_param_mem, false, "load model from shared memeory");

REGIST_OPTION_CREATOR(run_strategy, lar::StrategyOption::create_option);

REGIST_OPTION_CREATOR(run_testcase, lar::TestcaseOption::create_option);
