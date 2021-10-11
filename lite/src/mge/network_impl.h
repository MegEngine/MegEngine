/**
 * \file src/mge/network_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "lite/network.h"
#include "network_impl_base.h"
#include "tensor_impl.h"

#include "megbrain/graph/bases.h"
#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/serialization/file.h"
#include "megbrain/serialization/load_dump_config.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/utils/thin/hash_table.h"

#include <memory>
#include <unordered_map>

namespace lite {

/*!
 * \brief implement the Network, contain the mgb related member
 */
class NetworkImplDft final : public Network::NetworkImplBase {
    LITE_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using S = megdnn::param::ExecutionPolicy::Strategy;
    //! set the config of the network, include:
    //! the inference device
    //! the other inference options, such as record_level, weight_preprocess...
    void set_config(const Config& config) override;

    //! set the special io infomation, if not set, default io tensor will used,
    //! this is special for input/output is not host tensor, default the
    //! input/output tensors are host tensor
    void set_io(const NetworkIO& network_io) override;

    //! only compute the output tensor in user configured
    void compute_only_configured_output() override {
        m_compute_configured_output_only = true;
    }

    //! get the network input and ouput tensor, the layout of which is
    //! sync from mge tensor
    std::shared_ptr<Tensor> get_io_tensor(
            std::string io_name,
            LiteTensorPhase phase = LiteTensorPhase::LITE_IO) override;

    //! get the input tensor by index in the load_result tensormap
    std::shared_ptr<Tensor> get_input_tensor(size_t index) override;

    //! get the output tensor by index in the load_result output_var_list
    std::shared_ptr<Tensor> get_output_tensor(size_t index) override;

    //! get all the input tensor name in the order in load return
    std::vector<const char*> get_all_input_name() const override;

    //! get all the output tensor name in the order in load return
    std::vector<const char*> get_all_output_name() const override;

    //! get the input tensor name in the order in load return
    const char* get_input_name(size_t index) const override;

    //! get the output tensor name in the order in load return
    const char* get_output_name(size_t index) const override;

    //! set the callback in async model
    void set_async_callback(const AsyncCallback& callback) override;

    //! set the start callback which will execute before network forward
    void set_start_callback(const StartCallback& callback) override {
        m_start_callback = std::move(callback);
    }

    //! set the finish callback which will execute after network forward
    void set_finish_callback(const FinishCallback& callback) override {
        m_finish_callback = std::move(callback);
    }

    //! load the model and get the m_load_result
    void load_model(
            std::shared_ptr<void> model_mem, size_t size,
            std::unordered_map<std::string, LiteAny> separate_config_map = {}) override;

    //! forward the network with filled input data and fill the output data
    //! to the output tensor
    void forward() override;

    //! in sync model, wait utile the inference finish
    void wait() override;

    virtual LiteDeviceType get_device_type() const override {
        return m_user_config->device_type;
    }

    //! Set cpu default mode when device is CPU, in some low computation
    //! device or single core device, this mode will get good performace
    void set_cpu_inplace_mode();
    bool is_cpu_inplace_mode() const { return m_is_cpu_inplace_mode; }

    //! When device is CPU, this interface will set the to be loaded model
    //! run in multi thread mode with the given thread number.
    void set_cpu_threads_number(size_t nr_threads);
    size_t get_cpu_threads_number() const { return m_nr_threads; }

    //! set device id, default device id = 0
    void set_device_id(int device_id) override;
    int get_device_id() const override { return m_compnode_locator.device; };

    LiteBackend get_backend_type() const override { return LiteBackend::LITE_DEFAULT; }
    //! set stream id, default stream id = 0
    void set_stream_id(int stream_id) override;
    int get_stream_id() const override { return m_compnode_locator.stream; };

    //! enable tensorrt
    void use_tensorrt();

    //! enable profile the network, a JSON format file will be generated
    void enable_profile_performance(std::string profile_json_file_path) override;

    /********************** mge special function ************************/
    //! load a new network which will share weights with src network
    void shared_weight_with(const NetworkImplBase* src_network);

    //! share the runtime memory with other network, the weights is not shared
    void share_runtime_memory_with(NetworkImplBase* network);
    //! set threads affinity callback;
    void set_runtime_thread_affinity(
            const ThreadAffinityCallback& thread_affinity_callback);

    //! set the network memroy allocator, the allocator is defined by user
    void set_memory_allocator(std::shared_ptr<Allocator> user_allocator);

    //! set opr algorithm selection strategy in the network
    void set_network_algo_policy(
            LiteAlgoSelectStrategy strategy, uint32_t shared_batch_size,
            bool binary_equal_between_batch);

    //! set workspace_limit for oprs with multiple algorithms, set
    //! workspace limitation can save memory but may influence the performance
    void set_network_algo_workspace_limit(size_t workspace_limit);

    //! Dump input/output values of all internal variables to output file,
    //! in text format
    void enable_io_txt_dump(std::string io_txt_out_file);

    //! Dump input/output values of all internal variables to output
    //! directory, in binary format
    void enable_io_bin_dump(std::string io_bin_out_dir);

private:
    //! construct the outputspec according to the m_network_io, and set the
    //! call_back to the outputspec
    void make_output_spec();

    //! modify the execution policy
    void modify_exection_policy();

    //! if the input is dev tensor, the pass will replace the H2D Opr to
    //! VolatileSharedDeviceTensor Opr
    void replace_dev_input_pass();

    //! check whether the model is cross compnode
    void cross_compnode_model_detect();

    //! when the model have loaded, update the IO, if not set networkio, update
    //! the networkio with the IO of loaded model
    void update_io();

    void update_input();
    void update_output();

    //! when the model info have loaded, update the config according the model
    //! info, finaly use it in compute graph
    void application_config();

    //! after finish forwarding the netwark, output the result of plugin to file
    void output_plugin_result() const;

    //! when finish forwarding the network, the function will be called
    void finish() const;

    //! before forwarding the network, the function will be called
    void start() const;

    //! compile the graph to get the execute function
    void compile_graph();

    //! try to infer output tensor layout
    void try_infer_tensor_layout(
            std::shared_ptr<Tensor> tensor, mgb::cg::SymbolVar var);

private:
    bool m_async = false;
    bool m_is_cpu_inplace_mode = false;
    int m_nr_device_type = 0;
    size_t m_nr_threads = 1;
    bool m_compute_configured_output_only = false;
    mgb::CompNode::Locator m_compnode_locator;

    AsyncCallback m_async_callback = nullptr;
    std::unique_ptr<NetworkIOInner> m_network_io;
    std::unique_ptr<Config> m_user_config;
    std::unique_ptr<mgb::cg::AsyncExecutable> m_execute_func;

    //! The model load related data
    S m_execution_policy = static_cast<S>(0);
    std::unique_ptr<mgb::serialization::InputFile> m_input_file;
    mgb::serialization::GraphLoadConfig m_load_config;
    mgb::serialization::GraphLoader::LoadResult m_load_result;
    mgb::ComputingGraph::OutputSpec m_output_spec;
    std::shared_ptr<mgb::serialization::GraphLoader> m_loader;

    //! start and finish callback
    StartCallback m_start_callback = nullptr;
    FinishCallback m_finish_callback = nullptr;

    //! profile and io dump related data
#if MGB_ENABLE_JSON
    std::unique_ptr<mgb::GraphProfiler> m_profiler;
    std::string m_profiler_output_file;
#endif
    std::unique_ptr<mgb::OprIODumpBase> m_iodump;
};

}  // namespace lite

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
