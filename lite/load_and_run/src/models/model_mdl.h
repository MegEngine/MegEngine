#pragma once
#include <string>
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/utils/debug.h"

#include "megbrain/plugin/num_range_checker.h"
#include "megbrain/plugin/profiler.h"

#include "helpers/common.h"
#include "helpers/data_parser.h"
#include "model.h"

namespace lar {

class ModelMdl : public ModelBase {
public:
    using Strategy = mgb::opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    //! interface implement of ModelBase
    ModelMdl(const std::string& path);

    ModelType type() override { return ModelType::MEGDL_MODEL; }

    void set_shared_mem(bool state) override { share_model_mem = state; }

    void load_model() override;

    void make_output_spec();

    void run_model() override;

    void wait() override;

#if MGB_ENABLE_JSON
    std::shared_ptr<mgb::json::Object> get_io_info() override;
#endif

    //! get load result for megDL model
    mgb::serialization::GraphLoader::LoadResult& get_mdl_load_result() {
        return m_load_result;
    }

    void update_mdl_load_result(const mgb::SymbolVarArray& output_var_array);

    //! get load config for megDL model
    mgb::serialization::GraphLoadConfig& get_mdl_config() { return m_load_config; }

    /*! reset the underlying graph loader from which further load() would read()
     *
     * \param input_file new input_file, can be null
     * \return new loader
     */
    std::shared_ptr<mgb::serialization::GraphLoader>& reset_loader(
            std::unique_ptr<mgb::serialization::InputFile> input_file = {});

    //! get the underlying graph loader
    std::shared_ptr<mgb::serialization::GraphLoader>& get_loader() { return m_loader; }

    //!  algo strategy for runing model
    void set_mdl_strategy(Strategy& u_strategy) { m_strategy = u_strategy; }
    Strategy& get_mdl_strategy() { return m_strategy; }

    //! get data parser
    DataParser& get_input_parser() { return parser; }

    uint32_t get_testcase_num() { return testcase_num; }

    std::vector<std::pair<std::string, mgb::HostTensorND*>>& get_test_input() {
        return test_input_tensors;
    }

    //! get output specified configuration
    mgb::ComputingGraph::OutputSpec& get_output_spec() { return m_output_spec; }

    std::unique_ptr<mgb::cg::AsyncExecutable>& get_async_func() { return m_asyc_exec; }

    void set_output_callback(std::vector<mgb::ComputingGraph::Callback>& cb) {
        mgb_assert(
                m_callbacks.size() == cb.size(),
                "invalid output callback list to set!!");
        for (size_t i = 0; i < cb.size(); i++) {
            m_callbacks[i] = cb[i];
        }
    }

#if MGB_ENABLE_JSON
    std::unique_ptr<mgb::GraphProfiler>& get_profiler() { return m_profiler; }
    void set_profiler() {
        m_profiler =
                std::make_unique<mgb::GraphProfiler>(m_load_config.comp_graph.get());
    }
#endif

    void set_num_range_checker(float range) {
        m_num_range_checker = std::make_unique<mgb::NumRangeChecker>(
                m_load_config.comp_graph.get(), range);
    }

    std::unique_ptr<mgb::serialization::GraphDumper> get_dumper(
            std::unique_ptr<mgb::serialization::OutputFile> out_file) {
        return mgb::serialization::GraphDumper::make(
                std::move(out_file), m_format.val());
    }

    const std::string& get_model_path() const override { return model_path; }

    std::vector<uint8_t> get_model_data() override;

private:
    bool share_model_mem;
    std::string model_path;
    std::unique_ptr<mgb::serialization::InputFile> m_model_file;
    mgb::serialization::GraphLoadConfig m_load_config;
    mgb::Maybe<mgb::serialization::GraphDumpFormat> m_format;

    mgb::serialization::GraphLoader::LoadResult m_load_result;
    std::shared_ptr<mgb::serialization::GraphLoader> m_loader;
    std::unique_ptr<mgb::cg::AsyncExecutable> m_asyc_exec;

    uint32_t testcase_num;
    std::vector<std::pair<std::string, mgb::HostTensorND*>> test_input_tensors;

    DataParser parser;
    Strategy m_strategy = Strategy::HEURISTIC;
    std::vector<mgb::ComputingGraph::Callback> m_callbacks;
    mgb::ComputingGraph::OutputSpec m_output_spec;

    std::unique_ptr<mgb::NumRangeChecker> m_num_range_checker;
#if MGB_ENABLE_JSON
    std::unique_ptr<mgb::GraphProfiler> m_profiler;
#endif
};

}  // namespace lar

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
