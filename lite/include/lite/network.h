/**
 * \file inlude/lite/network.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "macro.h"
#include "tensor.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace lite {

LITE_API inline LiteAlgoSelectStrategy operator|(
        LiteAlgoSelectStrategy x, LiteAlgoSelectStrategy y) {
    return static_cast<LiteAlgoSelectStrategy>(
            static_cast<uint32_t>(x) | static_cast<uint32_t>(y));
}

/*!
 * \brief the inference options which will be translated to megenine
 *
 * \param weight_preprocess is the option wich optimize the inferece performance
 * with preprocess the const weights
 *
 * \param fuse_preprocess fuse preprocess patten, like astype + pad_channel +
 * dimshuffle
 *
 * \param fake_next_exec  whether only to perform non-computing tasks (like
 * memory allocation and queue initialization) for next exec. This would be
 * reset to false when the graph is executed.
 *
 * \param var_sanity_check_first_run Disable var sanity check on the first run.
 * Var sanity check is enabled on the first-time execution by default, and can
 * be used to find some potential memory access errors in the operator
 * implementation.
 *
 * \param const_shape This can be used to reduce memory usage since some
 * static inference data structures can be omitted.
 *
 * \param force_dynamic_alloc force dynamic memory alloc for all vars
 *
 * \param force_output_dynamic_alloc force dynamic memory alloc for output vars
 * which are used as CallbackCaller input when call compile() function
 *
 * \param no_profiling_on_shape_change do not re-profile to select best impl
 * algo when input shape changes (use previous algo)
 *
 * \param jit_level Execute supported operators with JIT (support MLIR,
 * NVRTC). Can only be used on Nvidia GPUs, this value indicates JIT level:
 * 1 for basic elemwise opr;
 * 2 for including reduce operator
 *
 * \param record_level flag optimize the inference performace with record the
 * kernel tasks in first run, hereafter the inference all need to execute the
 * recorded tasks.
 * level = 0 means the normal inference,
 * level = 1 means use record inference,
 * level = 2 means record inference with free the extra memory
 *
 * \param graph_opt_level optimization level:
 * 0: disable
 * 1: level-1: inplace arith transformations during graph
 *    construction
 * 2: level-2: level-1, plus global optimization before graph
 *    compiling
 * 3: also enable JIT
 * <0: corresponding level, with result check for debug
 *
 * \param async_exec_level exec: dispatch on separate threads for different
 * comp_node.
 * 0: do not perform async dispatch
 * 1: dispatch async if there are more than one comp node with limited queue
 * mask 0b10: async if there are multiple comp nodes with
 * mask 0b100: always async
 */
struct LITE_API Options {
    bool weight_preprocess = false;
    bool fuse_preprocess = false;
    bool fake_next_exec = false;
    bool var_sanity_check_first_run = true;
    bool const_shape = false;
    bool force_dynamic_alloc = false;
    bool force_output_dynamic_alloc = false;
    bool force_output_use_user_specified_memory = false;
    bool no_profiling_on_shape_change = false;
    uint8_t jit_level = 0;
    uint8_t comp_node_seq_record_level = 0;
    uint8_t graph_opt_level = 2;
    uint16_t async_exec_level = 1;

    //! layout transform options
    bool enable_nchw44 = false;
    bool enable_nchw44_dot = false;
    bool enable_nchw88 = false;
    bool enable_nhwcd4 = false;
    bool enable_nchw4 = false;
    bool enable_nchw32 = false;
    bool enable_nchw64 = false;
};

/*!
 * \brief Configuration when load and compile the graph
 *
 * \param bare_model_cryption_name is the bare model cryption method name, bare
 *model is not pack json info inside
 *
 *\param has_compression flag whether the model is compressed, the compress
 *method will read form the model
 */
struct LITE_API Config {
    bool has_compression = false;
    int device_id = 0;
    LiteDeviceType device_type = LiteDeviceType::LITE_CPU;
    LiteBackend backend = LiteBackend::LITE_DEFAULT;
    std::string bare_model_cryption_name = {};
    Options options = {};
};

/*!
 * \brief config the network input and output item
 *
 */
struct LITE_API IO {
    //! the tensor name in the graph corresponding to the IO
    std::string name;

    //! Used to mark where the input tensor comes from and the output where copy
    //! to, if is_host is true, the input is from host and output copy to host,
    //! otherwise device. Sometimes The input is from device and output no need
    //! copy to host, default is true.
    bool is_host = true;

    //! The IO type, it can be SHAPE or VALUE, when SHAPE is set, the input or
    //! output tensor value is invaid, only shape will be set, default is VALUE
    LiteIOType io_type = LiteIOType::LITE_IO_VALUE;

    //! The layout of the config from user, if other layout is set before
    //! forward or get after forward by input tensor reset, this layout will by
    //! pass. if no other layout is set before forward, this layout will work.
    //! if this layout is no set, the model will forward with its origin layout.
    //! if in output, it will used to check.
    Layout config_layout = {};
};

/*!
 * \brief the input and output information when load the network
 * the NetworkIO will remain in the network until the network is destroyed
 */
struct LITE_API NetworkIO {
    std::vector<IO> inputs = {};
    std::vector<IO> outputs = {};
};

/*!
 * \brief A user-implemented allocator interface
 */
class LITE_API Allocator {
public:
    virtual ~Allocator() = default;

    //! allocate memory of size in the given device with the given align
    virtual void* allocate(
            LiteDeviceType device_type, int device_id, size_t size, size_t align) = 0;

    //! free the memory pointed by ptr in the given device
    virtual void free(LiteDeviceType device_type, int device_id, void* ptr) = 0;
};

/*!
 * \brief the thread affinith callback type
 * \param thread_id thread_id is the a number begin from 0 to (nr_threads - 1),
 * thread_id of (nr_threads - 1) is the main worker thread.
 */
using ThreadAffinityCallback = std::function<void(int thread_id)>;

using AsyncCallback = std::function<void(void)>;

/*!
 * \brief the start/finish callback function
 * \param unordered_map map from the io tensor name to the pair of which is the
 * corresponding IO of user config and the realy input or output tensor.
 */
using StartCallback =
        std::function<void(const std::unordered_map<
                           std::string, std::pair<IO, std::shared_ptr<Tensor>>>&)>;
using FinishCallback =
        std::function<void(const std::unordered_map<
                           std::string, std::pair<IO, std::shared_ptr<Tensor>>>&)>;

/*!
 * \brief The network is construct form a model, implement model load, init,
 * forward, and display some model information
 */
class LITE_API Network {
public:
    class NetworkImplBase;

    ~Network();

    Network(const Config& config = {}, const NetworkIO& networkio = {});

    Network(const NetworkIO& networkio, const Config& config = {});

    //! load the model form memory
    void load_model(void* model_mem, size_t size);

    //! load the model from a model path
    void load_model(std::string model_path);

    //! only compute the output tensor in user configured
    void compute_only_configured_output();

    //! get the network input and output tensor, the layout of which is
    //! sync from mge tensor, when the name of input and output tensor  are the
    //! same, use LiteTensorPhase to separate
    std::shared_ptr<Tensor> get_io_tensor(
            std::string io_name, LiteTensorPhase phase = LiteTensorPhase::LITE_IO);

    //! get the network input by index
    std::shared_ptr<Tensor> get_input_tensor(size_t index);

    //! get the network output tensor by index
    std::shared_ptr<Tensor> get_output_tensor(size_t index);

    //! set the network forward in async mode and set the async callback
    //! function
    Network& set_async_callback(const AsyncCallback& async_callback);

    //! set the start forward callback function, which will be execute before
    //! forward. this can be used to check network input or dump model inputs
    //! for debug
    Network& set_start_callback(const StartCallback& start_callback);

    //! set the finish forward callback function, which will be execute after
    //! forward. this can be used to dump model outputs for debug
    Network& set_finish_callback(const FinishCallback& finish_callback);

    //! forward the network with filled input data and fill the output data
    //! to the output tensor
    void forward();

    //! waite until forward finish in sync model
    void wait();

    //! get the input tensor name in the order in load return
    std::string get_input_name(size_t index) const;

    //! get the output tensor name in the order in load return
    std::string get_output_name(size_t index) const;

    //! get all the input tensor name in the order in load return
    std::vector<std::string> get_all_input_name() const;

    //! get all the output tensor name in the order in load return
    std::vector<std::string> get_all_output_name() const;

    //! set/get device id, default device id = 0
    Network& set_device_id(int device_id);
    int get_device_id() const;

    //! set/get stream id, default stream id = 0
    Network& set_stream_id(int stream_id);
    int get_stream_id() const;

    //! enable profile the network, a file will be generated
    void enable_profile_performance(std::string profile_file_path);

    //! get model extra info
    const std::string& get_model_extra_info();

    //! get device type
    LiteDeviceType get_device_type() const;

    //! get static peak memory info showed by Graph visualization
    void get_static_memory_alloc_info(const std::string& log_dir = "logs/test") const;

public:
    friend class NetworkHelper;

private:
    //! update member from implement
    void update_from_implement();

    //! decrypt and parse the model file
    void prase_model(std::shared_ptr<void> model_data, size_t size);

private:
    bool m_loaded = false;
    Config m_config;
    NetworkIO m_network_io;
    std::unique_ptr<NetworkImplBase> m_impl;
    std::string m_extra_info;
};

/*********************** MGE special network function ***************/
class LITE_API Runtime {
public:
    //! When device is CPU, this interface will set the to be loaded model
    //! run in multi thread mode with the given thread number.
    static void set_cpu_threads_number(
            std::shared_ptr<Network> dst_network, size_t nr_threads);
    static size_t get_cpu_threads_number(std::shared_ptr<Network> dst_network);

    //! set threads affinity callback;
    static void set_runtime_thread_affinity(
            std::shared_ptr<Network> network,
            const ThreadAffinityCallback& thread_affinity_callback);

    //! Set cpu default mode when device is CPU, in some low computation
    //! device or single core device, this mode will get good performace
    static void set_cpu_inplace_mode(std::shared_ptr<Network> dst_network);
    static bool is_cpu_inplace_mode(std::shared_ptr<Network> dst_network);

    //! Set use tensorrt forward
    static void use_tensorrt(std::shared_ptr<Network> dst_network);

    //! set opr algorithm selection strategy in the network
    //! shared_batch_size: the batch size used by fastrun,
    //!                    Non-zero value means that fastrun use this batch size
    //!                    regardless of the batch size of the model. Zero means
    //!                    fastrun use batch size of the model
    //! binary_equal_between_batch: if the content of each input batch is binary
    //!                             equal,whether the content of each output
    //!                             batch is promised to be equal
    static void set_network_algo_policy(
            std::shared_ptr<Network> dst_network, LiteAlgoSelectStrategy strategy,
            uint32_t shared_batch_size = 0, bool binary_equal_between_batch = false);

    //! set workspace_limit for oprs with multiple algorithms, set
    //! workspace limitation can save memory but may influence the performance
    static void set_network_algo_workspace_limit(
            std::shared_ptr<Network> dst_network, size_t workspace_limit);

    //! set the network memroy allocator, the allocator is defined by user
    static void set_memory_allocator(
            std::shared_ptr<Network> dst_network,
            std::shared_ptr<Allocator> user_allocator);

    //! share the runtime memory with other network, the weights is not shared
    static void share_runtime_memory_with(
            std::shared_ptr<Network> dst_network, std::shared_ptr<Network> src_network);

    //! Dump input/output values of all internal variables to output
    //! file, in txt format
    static void enable_io_txt_dump(
            std::shared_ptr<Network> dst_network, std::string io_txt_out_file);

    //! Dump input/output values of all internal variables to output
    //! directory, in binary format
    static void enable_io_bin_dump(
            std::shared_ptr<Network> dst_network, std::string io_bin_out_dir);

    //! load a new network which will share weights with src network
    static void shared_weight_with_network(
            std::shared_ptr<Network> dst_network,
            const std::shared_ptr<Network> src_network);
};

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
