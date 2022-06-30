#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "../src/common.h"
#include "../src/mge/network_impl.h"
#include "../src/misc.h"
#include "lite/global.h"

#include "megbrain/tensor.h"
#include "megbrain/utils/infile_persistent_cache.h"
#include "megbrain/utils/persistent_cache.h"
#include "test_common.h"

#include <string.h>
#include <chrono>
#include <memory>
#include <random>

using namespace lite;

TEST(TestNetWorkOptions, no_var_sanity_check_and_record) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.var_sanity_check_first_run = false;
    config.options.comp_node_seq_record_level = 1;

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, auto_optimize_inference_layout) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.auto_optimize_inference = true;

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, const_shape) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.var_sanity_check_first_run = false;
    config.options.const_shape = true;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, record2) {
    Config config;
    std::string model_path = "./shufflenet.mge";

    config.options.var_sanity_check_first_run = false;
    config.options.const_shape = true;
    config.options.comp_node_seq_record_level = 2;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    for (int i = 0; i < 3; i++) {
        network->forward();
        network->wait();
    }
}

TEST(TestNetWorkOptions, NCHW44) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.var_sanity_check_first_run = false;
    config.options.enable_nchw44 = true;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    Runtime::set_network_algo_policy(
            network, LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                             LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, test_cache) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    set_persistent_cache("./algo_cache.txt", true);
    network->load_model(model_path);
    Runtime::set_network_algo_policy(
            network, LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                             LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);

    dump_persistent_cache("./algo_cache.txt");
    ASSERT_TRUE(fopen("./algo_cache.txt", "r"));

    set_persistent_cache("./algo_cache.txt");
    network->forward();
    network->wait();
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, DisableModelInfo) {
    //! clear the cache set by other test
    mgb::PersistentCache::inst().set_impl(
            std::make_shared<mgb::InMemoryPersistentCache>());
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./test_pack_cache_to_model.lite";
    std::string model_path2 = "./test_pack_cache_to_model.lite";
    std::string input_name = "data";
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->extra_configure({true});
    Runtime::set_cpu_inplace_mode(network);
    network->load_model(model_path);
    //! the fast-run cache will not configure, so it is not support dump
    ASSERT_EQ(mgb::PersistentCache::inst().support_dump_cache(), false);
    ASSERT_EQ(Runtime::is_cpu_inplace_mode(network), true);

    std::shared_ptr<Network> network2 = std::make_shared<Network>(config);
    network2->load_model(model_path2);
    //! the fast-run cache is configured by the model information
    ASSERT_EQ(mgb::PersistentCache::inst().support_dump_cache(), true);
}

TEST(TestNetWorkOptions, FastRunIgnorBatch) {
    Config config;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    set_persistent_cache("./algo_cache.txt");
    network->load_model(model_path);
    Runtime::set_network_algo_policy(
            network,
            LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                    LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE,
            1, true);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);

    dump_persistent_cache("./algo_cache.txt");
    ASSERT_TRUE(fopen("./algo_cache.txt", "r"));
}

#if LITE_WITH_CUDA
TEST(TestNetWorkOptions, NCHW4) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.enable_nchw4 = 1;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, NCHW32) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.enable_nchw32 = 1;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_network_algo_policy(
            network, LiteAlgoSelectStrategy::LITE_ALGO_PROFILE |
                             LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE);
    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();
    compare_lite_tensor<float>(output_tensor, result_mgb);
}

TEST(TestNetWorkOptions, jit_level) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    config.options.jit_level = 1;
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();

    compare_lite_tensor<float>(output_tensor, result_mgb);
}
#endif

#if MGB_ENABLE_TENSOR_RT && LITE_WITH_CUDA
TEST(TestNetWorkOptions, TensorRT) {
    Config config;
    config.device_type = LiteDeviceType::LITE_CUDA;
    auto tensor = get_input_data("./input_data.npy");
    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";
    auto result_mgb = mgb_lar(model_path, config, input_name, tensor);

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::use_tensorrt(network);

    set_tensor_rt_cache("./tensorrt_cache.txt");
    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = tensor->get_memory_ptr();
    auto src_layout = tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto result_tensor = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, Layout{{1, 1000}, 2, LiteDataType::LITE_FLOAT});

    void* out_data = result_tensor->get_memory_ptr();
    output_tensor->reset(out_data, result_tensor->get_layout());

    network->forward();
    network->wait();
    dump_tensor_rt_cache();
    ASSERT_TRUE(fopen("./tensorrt_cache.txt", "r"));
    compare_lite_tensor<float>(output_tensor, result_mgb);
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
