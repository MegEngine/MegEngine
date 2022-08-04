#pragma once

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "../src/mge/common.h"
#include "../src/mge/network_impl.h"
#include "../src/misc.h"
#include "lite/network.h"
#include "lite/tensor.h"
#include "megbrain/comp_node.h"
#include "megbrain/graph/bases.h"
#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/serialization/file.h"
#include "megbrain/serialization/load_dump_config.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/tensor.h"
#include "megbrain/utils/thin/hash_table.h"
#include "npy.h"

#include <gtest/gtest.h>

#include <string.h>
#include <chrono>
#include <memory>
#include <random>

namespace lite {

template <typename T>
static ::testing::AssertionResult compare_memory(
        const void* memory0, const void* memory1, size_t length, float maxerr = 1e-3) {
    const T* data_ptr0 = static_cast<const T*>(memory0);
    const T* data_ptr1 = static_cast<const T*>(memory1);
    for (size_t i = 0; i < length; i++) {
        auto diff = std::abs(data_ptr0[i] - data_ptr1[i]);
        if (diff > maxerr) {
            return ::testing::AssertionFailure() << "Unequal value:\n"
                                                 << "value 0 = " << data_ptr0[i] << "\n"
                                                 << "value 1 = " << data_ptr1[i] << "\n"
                                                 << "At index: " << i << "\n";
        }
    }
    return ::testing::AssertionSuccess();
}

template <typename T>
void compare_lite_tensor(
        std::shared_ptr<Tensor> tensor0, std::shared_ptr<Tensor> tensor1,
        float maxerr = 1e-3) {
    size_t elemsize = tensor0->get_layout().get_elem_size();
    T* data_ptr0 = static_cast<T*>(tensor0->get_memory_ptr());
    T* data_ptr1 = static_cast<T*>(tensor1->get_memory_ptr());
    size_t length = tensor0->get_tensor_total_size_in_byte() / elemsize;
    EXPECT_TRUE(compare_memory<T>(data_ptr0, data_ptr1, length, maxerr));
}

__attribute__((unused)) static std::shared_ptr<Tensor> get_input_data(
        std::string path) {
    std::string type_str;
    std::vector<npy::ndarray_len_t> stl_shape;
    std::vector<int8_t> raw;
    npy::LoadArrayFromNumpy(path, type_str, stl_shape, raw);
    auto lite_tensor = std::make_shared<Tensor>(LiteDeviceType::LITE_CPU);
    Layout layout;
    layout.ndim = stl_shape.size();
    const std::map<std::string, LiteDataType> type_map = {
            {"f4", LiteDataType::LITE_FLOAT},  {"f2", LiteDataType::LITE_HALF},
            {"i8", LiteDataType::LITE_INT64},  {"i4", LiteDataType::LITE_INT},
            {"u4", LiteDataType::LITE_UINT},   {"i2", LiteDataType::LITE_INT16},
            {"u2", LiteDataType::LITE_UINT16}, {"i1", LiteDataType::LITE_INT8},
            {"u1", LiteDataType::LITE_UINT8}};
    layout.shapes[0] = 1;
    for (size_t i = 0; i < stl_shape.size(); i++) {
        layout.shapes[i] = static_cast<size_t>(stl_shape[i]);
    }
    for (auto& item : type_map) {
        if (type_str.find(item.first) != std::string::npos) {
            layout.data_type = item.second;
            break;
        }
    }
    lite_tensor->set_layout(layout);
    size_t length = lite_tensor->get_tensor_total_size_in_byte();
    void* dest = lite_tensor->get_memory_ptr();
    memcpy(dest, raw.data(), length);
    return lite_tensor;
}

__attribute__((unused)) static std::shared_ptr<Tensor> mgelite_lar(
        std::string model_path, const Config& config, std::string,
        std::shared_ptr<Tensor> input) {
    std::unique_ptr<Network> network = std::make_unique<Network>(config);

    network->load_model(model_path);

    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    auto src_ptr = input->get_memory_ptr();
    auto src_layout = input->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    network->forward();
    network->wait();

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    Layout out_layout = output_tensor->get_layout();
    auto ret = std::make_shared<Tensor>(LiteDeviceType::LITE_CPU, out_layout);
    void* out_data = output_tensor->get_memory_ptr();
    void* dst_data = ret->get_memory_ptr();
    memcpy(dst_data, out_data, ret->get_tensor_total_size_in_byte());
    return ret;
}

__attribute__((unused)) static std::shared_ptr<Tensor> mgb_lar(
        std::string model_path, const Config& config, std::string input_name,
        std::shared_ptr<Tensor> input) {
    LITE_ASSERT(config.bare_model_cryption_name.size() == 0);
    using namespace mgb;
    serialization::GraphLoader::LoadConfig mgb_config;
    mgb_config.comp_node_mapper = [config](CompNode::Locator& loc) {
        loc = to_compnode_locator(config.device_type);
    };
    mgb_config.comp_graph = ComputingGraph::make();
    auto&& graph_opt = mgb_config.comp_graph->options();
    if (config.options.weight_preprocess) {
        graph_opt.graph_opt.enable_weight_preprocess();
    }
    graph_opt.comp_node_seq_record_level = config.options.comp_node_seq_record_level;

    auto inp_file = mgb::serialization::InputFile::make_fs(model_path.c_str());
    auto format = serialization::GraphLoader::identify_graph_dump_format(*inp_file);
    mgb_assert(
            format.valid(),
            "invalid model: unknown model format, please make sure input "
            "file is generated by GraphDumper");
    auto loader = serialization::GraphLoader::make(std::move(inp_file), format.val());
    auto load_ret = loader->load(mgb_config, false);

    ComputingGraph::OutputSpec out_spec;
    std::vector<HostTensorND> output_tensors(load_ret.output_var_list.size());
    for (size_t i = 0; i < load_ret.output_var_list.size(); i++) {
        auto cb = [&output_tensors, i](const DeviceTensorND& dv) mutable {
            output_tensors[i].copy_from(dv);
        };
        out_spec.emplace_back(load_ret.output_var_list[i], std::move(cb));
    }
    auto func = load_ret.graph_compile(out_spec);

    auto& in = load_ret.tensor_map.find(input_name)->second;
    in->copy_from(*TensorHelper::implement(input)
                           ->cast_final_safe<TensorImplDft>()
                           .host_tensor());
    func->execute();
    func->wait();

    std::shared_ptr<Tensor> ret = std::make_shared<Tensor>(
            LiteDeviceType::LITE_CPU, to_lite_layout(output_tensors[0].layout()));
    auto mge_tensor = TensorHelper::implement(ret)
                              ->cast_final_safe<TensorImplDft>()
                              .host_tensor();
    mge_tensor->copy_from(output_tensors[0]);
    return ret;
}
}  // namespace lite

#endif

static inline bool check_gpu_available(size_t num) {
    if (mgb::CompNode::get_device_count(mgb::CompNode::DeviceType::CUDA) < num) {
        mgb_log_warn("skip test case that requires %zu GPU(s)", num);
        return false;
    }
    return true;
}
#define REQUIRE_CUDA()                 \
    {                                  \
        if (!check_gpu_available(1)) { \
            return;                    \
        }                              \
    }                                  \
    while (0)
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
