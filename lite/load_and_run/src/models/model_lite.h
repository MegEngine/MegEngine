#pragma once

#include <string>
#include "helpers/common.h"
#include "helpers/data_parser.h"
#include "lite/network.h"
#include "model.h"

namespace lar {
/*!
 * \brief: megengine lite model
 */
class ModelLite : public ModelBase {
public:
    using Strategy = LiteAlgoSelectStrategy;

    ModelLite(const std::string& path);
    //!  model type
    ModelType type() override { return ModelType::LITE_MODEL; }

    //! set to load from shared memory
    void set_shared_mem(bool state) override { share_model_mem = state; }

    //! load model from dump file
    void load_model() override;

    //! run model with given runtime parameter
    void run_model() override;

    //! wait the end of asynchronous function execution
    void wait() override;

#if MGB_ENABLE_JSON
    std::shared_ptr<mgb::json::Object> get_io_info() override;
#endif

    //! enable global layout transform
    void set_layout_transform(bool state) { enable_layout_transform = state; }

    //! get the network of lite model
    std::shared_ptr<lite::Network>& get_lite_network() { return m_network; }

    //! get the config of lite model
    lite::Config& get_config() { return config; }

    //! get the networkIO of lite model
    lite::NetworkIO& get_networkIO() { return IO; }

    //! get the data parser
    DataParser& get_input_parser() { return parser; }

    //! set the strategy before load model
    void set_lite_strategy(Strategy& u_strategy) { m_strategy = u_strategy; }

    //! get algo strategy
    Strategy& get_lite_strategy() { return m_strategy; }

    const std::string& get_model_path() const override { return model_path; }

    std::vector<uint8_t> get_model_data() override;

private:
    bool share_model_mem;
    bool enable_layout_transform;
    std::string model_path;

    DataParser parser;
    lite::Config config;
    lite::NetworkIO IO;

    std::shared_ptr<lite::Network> m_network;

    Strategy m_strategy;
};
}  // namespace lar
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
