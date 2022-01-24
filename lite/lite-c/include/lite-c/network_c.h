/**
 * \file lite-c/include/lite-c/network_c.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef LITE_C_NETWORK_H_
#define LITE_C_NETWORK_H_

#include "tensor_c.h"

#ifdef __cplusplus
extern "C" {
#endif

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
typedef struct Options {
    int weight_preprocess;
    int fuse_preprocess;
    int fake_next_exec;
    int var_sanity_check_first_run;
    int const_shape;
    int force_dynamic_alloc;
    int force_output_dynamic_alloc;
    int force_output_use_user_specified_memory;
    int no_profiling_on_shape_change;
    int jit_level;
    int comp_node_seq_record_level;
    int graph_opt_level;
    int async_exec_level;

    //! layout transform options
    int enable_nchw44;
    int enable_nchw44_dot;
    int enable_nchw88;
    int enable_nhwcd4;
    int enable_nchw4;
    int enable_nchw32;
    int enable_nchw64;
} LiteOptions;

//! define a default Options
extern LITE_API const LiteOptions default_option;

/*!
 * \brief Configuration when load and compile the graph
 *
 * \param bare_model_cryption_name is the bare model cryption method name, bare
 *model is not pack json info inside
 *
 *\param has_compression flag whether the model is compressed, the compress
 *method will read form the model
 */
typedef struct LiteConfig {
    int has_compression;
    int device_id;
    LiteDeviceType device_type;
    LiteBackend backend;
    const char* bare_model_cryption_name;
    LiteOptions options;
} LiteConfig;

//! get default config
LITE_API LiteConfig* default_config();

/*!
 * \brief config the network input and output item
 *
 */
typedef struct LiteIO {
    //! the tensor name in the graph corresponding to the IO
    const char* name;

    //! Used to mark where the input tensor comes from and the output where copy
    //! to, if is_host is true, the input is from host and output copy to host,
    //! otherwise device. Sometimes The input is from device and output no need
    //! copy to host, default is true.
    int is_host;

    //! The IO type, it can be SHAPE or VALUE, when SHAPE is set, the input or
    //! output tensor value is invaid, only shape will be set, default is VALUE
    LiteIOType io_type;

    //! The layout of the config from user, if other layout is set before
    //! forward or get after forward, this layout will by pass. if no other
    //! layout is set before forward, this layout will work. if this layout is
    //! no set, the model will forward with its origin layout. if in output, it
    //! will used to check.
    LiteLayout config_layout;
} LiteIO;

//! define a default IO
extern LITE_API const LiteIO default_io;

/*!
 * \brief the input and output information when load the network
 * the NetworkIO will remain in the network until the network is destroyed
 */
typedef struct LiteNetworkIO {
    LiteIO* inputs;
    LiteIO* outputs;
    size_t input_size;   //! the number IO in inputs
    size_t output_size;  //! the number IO in outputs
} LiteNetworkIO;

//! get default NetworkIO
LITE_API LiteNetworkIO* default_network_io();

/*!
 * \brief A user-implemented allocator function
 */
//! allocate memory of size in the given device with the given align
typedef void* (*LiteAllocate)(
        LiteDeviceType device_type, int device_id, size_t size, size_t align);
//! free the memory pointed by ptr in the given device
typedef void (*LiteFree)(LiteDeviceType device_type, int device_id, void* ptr);

/*!
 * \brief the thread affinith callback type
 * \param thread_id thread_id is the a number begin from 0 to (nr_threads - 1),
 * thread_id of (nr_threads - 1) is the main worker thread.
 */
typedef int (*LiteThreadAffinityCallback)(int thread_id);

typedef int (*LiteAsyncCallback)();

typedef int (*LiteAsyncCallbackWithData)(void* user_data);

/*!
 * \brief the start/finish callback function
 * \param unordered_map map from the io tensor name to the pair of which is the
 * corresponding IO of user config and the realy input or output tensor.
 */

typedef int (*LiteStartCallback)(
        const LiteIO* inputs, const LiteTensor* input_tensors, size_t size);

typedef int (*LiteStartCallbackWithData)(
        const LiteIO* inputs, const LiteTensor* input_tensors, size_t size,
        void* user_data);

typedef int (*LiteFinishCallback)(
        const LiteIO* outputs, const LiteTensor* output_tensors, size_t size);

typedef int (*LiteFinishCallbackWithData)(
        const LiteIO* outputs, const LiteTensor* output_tensors, size_t size,
        void* user_data);

/*!
 * \brief The network is construct form a model, implement model load, init,
 * forward, and display some model information
 */
typedef void* LiteNetwork;

/**
 * \brief Create a lite Network object with default config and networkIO.
 * \param[out] network The netwrok pointer
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_make_default_network(LiteNetwork* network);

/**
 * \brief Create a lite Network object from the given config and networkIO.
 * \param[in] config The configration to create the network
 * \param[in] network_io The configration io to create the network
 * \param[out] network The network pointer
 */
LITE_API int LITE_make_network(
        LiteNetwork* network, const LiteConfig config, const LiteNetworkIO network_io);

/**
 * \brief Create a lite Network object from the given config and networkIO.
 * \param[in] config The configration to create the network
 * \param[out] network The network pointer
 */
LITE_API int LITE_make_network_config(LiteNetwork* network, const LiteConfig config);

/**
 * \brief load the model to network form memory
 * \param[in] model_mem The model in memory
 * \param[in] size The size of the model memory
 * \param[out] network The network to be load model in
 */
LITE_API int LITE_load_model_from_mem(
        LiteNetwork network, void* model_mem, size_t size);

/**
 * \brief load the model to network form given path
 * \param[in] model_path The model path
 * \param[out] network The network to be load model in
 */
LITE_API int LITE_load_model_from_path(LiteNetwork network, const char* model_path);

/**
 * \brief load a new network which will share weights with src network
 * \param[in] origin_network The origin network pointer
 * \param[out] network The network pointer
 */
LITE_API int LITE_shared_weight_with_network(
        LiteNetwork dst_network, const LiteNetwork src_network);

/**
 * \brief Destroy a lite network object.
 * \param[in] network The network pointer
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_destroy_network(LiteNetwork network);

/**
 * \brief forward the network with filled input data and fill the output data
 * to the output tensor
 * \param[in] network The loaded model
 */
LITE_API int LITE_forward(const LiteNetwork network);

/**
 * \brief waite until forward finish in sync model
 * \param[in] network The loaded model
 */
LITE_API int LITE_wait(const LiteNetwork network);

/**
 * \brief get the network input and ouput tensor, the layout of which is
 * get from model
 * \param[in] network The loaded model
 * \param[in] io_name The input or output name
 * \param[in] phase The tensor phase
 * \param[out] tensor The IO tensor get from the network
 */
LITE_API int LITE_get_io_tensor(
        LiteNetwork network, const char* io_name, LiteTensorPhase phase,
        LiteTensor* tensor);

/**
 * \brief get the input tensor name in the order in loaded model
 * \param[in] network The loaded model
 * \param[in] index The index of input tensor
 * \param[out] name The input tensor name
 */
LITE_API int LITE_get_input_name(
        const LiteNetwork network, size_t index, const char** name);

/**
 * \brief get the output tensor name in the order in loaded model
 * \param[in] network The loaded model
 * \param[in] index The index of output tensor
 * \param[out] name The output tensor name
 */
LITE_API int LITE_get_output_name(
        const LiteNetwork network, size_t index, const char** name);

/**
 * \brief get all the input tensor name in the order in loaded model
 * \param[in] network The loaded model
 * \param[in] size The number of the input tensor
 * \param[out] name The input tensor names
 */
LITE_API int LITE_get_all_input_name(
        const LiteNetwork network, size_t* size, const char** name);

/**
 * \brief get all the output tensor name in the order in loaded model
 * \param[in] network The loaded model
 * \param[in] size The number of output tensor
 * \param[out] name The output tensor name
 */
LITE_API int LITE_get_all_output_name(
        const LiteNetwork network, size_t* size, const char** name);

/**
 * \brief get whether the model is running in cpu inplace mode
 * \param[in] network The loaded model
 * \param[out] is_cpu_inplace_mode whether is in cpu inplace mode
 */
LITE_API int LITE_is_cpu_inplace_mode(
        const LiteNetwork network, int* is_cpu_inplace_mode);

/**
 * \brief get the number of thread the network will run with
 * \param[in] network The loaded model
 * \param[out] nr_threads the thread number when the network running
 */
LITE_API int LITE_get_cpu_threads_number(const LiteNetwork network, size_t* nr_threads);

/**
 * \brief get the device id the network will run with
 * \param[in] network The loaded model
 * \param[out] device_id the device id of the network will run
 */
LITE_API int LITE_get_device_id(const LiteNetwork network, int* device_id);

/**
 * \brief get the stream id the network will run with
 * \param[in] network The loaded model
 * \param[out] stream_id the stream id of the network will run
 */
LITE_API int LITE_get_stream_id(const LiteNetwork network, int* stream_id);

/**
 * \brief get the device type the network will run with
 * \param[in] network The loaded model
 * \param[out] device_type the device type of the network will run
 */
LITE_API int LITE_get_device_type(
        const LiteNetwork network, LiteDeviceType* device_type);

/**
 * \brief get the device type the network will run with
 * \param[in] network The loaded model
 * \param[out] info  : the json format memory
 * \param[out] info_size: the json format memory size
 */
LITE_API int LITE_get_model_extra_info(
        const LiteNetwork network, const char** info, int* info_size);

/**
 * \brief Set cpu default mode when device is CPU, in some low computation
 * device or single core device, this mode will get good performace
 * \param[in] network The loaded model
 */
LITE_API int LITE_set_cpu_inplace_mode(LiteNetwork network);

/**
 * \brief When device is CPU, this interface will set the to be loaded model
 * run in multi thread mode with the given thread number.
 * \param[in] network The loaded model
 * \param[in] nr_threads The threads number
 */
LITE_API int LITE_set_cpu_threads_number(LiteNetwork network, size_t nr_threads);

/**
 * \brief set device id, default device id = 0
 * \param[in] network The loaded model
 * \param[in] device_id The device id to be set
 */
LITE_API int LITE_set_device_id(LiteNetwork network, int device_id);

/**
 * \brief set stream id, default stream id = 0
 * \param[in] network The loaded model
 * \param[in] stream_id The stream id to be set
 */
LITE_API int LITE_set_stream_id(LiteNetwork network, int stream_id);

/**
 * \brief enable tensorrt
 * \param[in] network The loaded model
 */
LITE_API int LITE_use_tensorrt(LiteNetwork network);

/**
 * \brief set opr algorithm selection strategy in the network
 * \param[in] network The loaded model
 * \param[in] select_strategy The operator algorithm selection strategy
 */
LITE_API int LITE_set_network_algo_policy(
        LiteNetwork network, LiteAlgoSelectStrategy strategy);

/**
 * \brief set opr algorithm selection strategy in the network
 * \param[in] network The loaded model
 * \param[in] shared_batch_size: the batch size used by fastrun,
 *                      Non-zero value means that fastrun use this batch size
 *                      regardless of the batch size of the model. Zero means
 *                      fastrun use batch size of the model
 * \param[in] binary_equal_between_batch: if the content of each input batch is
 *                      binary equal,whether the content of each output batch is
 *                      promised to be equal
 */
LITE_API int LITE_set_network_algo_fastrun_config(
        LiteNetwork network, unsigned int shared_batch_size,
        int binary_equal_between_batch);

/**
 * \brief set workspace_limit for oprs with multiple algorithms, set
 * workspace limit can save memory but may influence the performance
 * \param[in] network The loaded model
 * \param[in] workspace_limit The operator algorithm workspace limit
 */
LITE_API int LITE_set_network_algo_workspace_limit(
        LiteNetwork network, size_t workspace_limit);

/**
 * \brief set the network forward in async mode and set the async callback
 * function
 * \param[in] network The loaded model
 * \param[in] async_callback when network finish forwarding, the callbak
 * will be called
 */
LITE_API int LITE_set_async_callback(
        LiteNetwork network, const LiteAsyncCallback async_callback);

/**
 * \brief set the network forward in async mode and set the async callback
 * function
 * \param[in] network The loaded model
 * \param[in] async_callback when network finish forwarding, the callback
 * will be called
 * \param[in] user_data user defined data for something user want to deploy
 * at forward finish stage
 */
LITE_API int LITE_set_async_callback_with_userdata(
        LiteNetwork network, const LiteAsyncCallbackWithData async_callback,
        void* user_data);

/**
 * \brief set the start forward callback function, which will be execute beform
 *  forward, this can be used to check network input or dump model inputs
 *  for debug
 * \param[in] network The loaded model
 * \param[in] start_callback when network start forwarding, the callbak
 * will be called
 */
LITE_API int LITE_set_start_callback(
        LiteNetwork network, const LiteStartCallback start_callback);

/**
 * \brief set the start forward callback function, which will be execute beform
 *  forward, this can be used to check network input or dump model inputs
 *  for debug
 * \param[in] network The loaded model
 * \param[in] start_callback when network start forwarding, the callbak
 * will be called
 * \param[in] user_data user defined data for something user want to deploy
 * at forward start stage
 */
LITE_API int LITE_set_start_callback_with_userdata(
        LiteNetwork network, const LiteStartCallbackWithData start_callback,
        void* user_data);

/**
 * \brief set the finish forward callback function, which will be execute after
 * forward, this can be used to dump model outputs for debug
 * \param[in] network The loaded model
 * \param[in] finish_callback when network finish forwarding, the callbak
 * will be called
 */
LITE_API int LITE_set_finish_callback(
        LiteNetwork network, const LiteFinishCallback finish_callback);

/**
 * \brief set the finish forward callback function, which will be execute after
 * forward, this can be used to dump model outputs for debug
 * \param[in] network The loaded model
 * \param[in] finish_callback when network finish forwarding, the callbak
 * will be called
 * \param[in] user_data user defined data for something user want to deploy
 * at finish stage
 */
LITE_API int LITE_set_finish_callback_with_userdata(
        LiteNetwork network, const LiteFinishCallbackWithData finish_callback,
        void* user_data);

/**
 * \brief set threads affinity callback
 * \param[in] network The loaded model
 * \param[in] thread_affinity_callback
 */
LITE_API int LITE_set_runtime_thread_affinity(
        LiteNetwork network, const LiteThreadAffinityCallback thread_affinity_callback);

/**
 * \brief set the network memroy allocator, the allocator is defined by user
 * \param[in] network The loaded model
 * \param[in] allocate_fun The allocate function of the user defined allocator
 * \param[in] free_fun The free function of the user defined allocator
 */
LITE_API int LITE_set_memory_allocator(
        LiteNetwork network, const LiteAllocate allocate_fun, const LiteFree free_fun);

/**
 * \brief the dst_network share the runtime memory with src_network
 * \param[in] src_network The source network
 * \param[in] dst_network The dst network to shared memory with src_network
 */
LITE_API int LITE_share_runtime_memroy(
        LiteNetwork src_network, LiteNetwork dst_network);

/**
 * \brief enable profile the network, a JSON format file will be generated
 * \param[in] network The loaded model
 * \param[in] profile_json_file_path The profile result file path
 */
LITE_API int LITE_enable_profile_performance(
        LiteNetwork network, const char* profile_json_file_path);

/**
 * \brief Dump input/output values of all internal variables to output file,
 * in text format
 * \param[in] network The loaded model
 * \param[in] io_txt_out_file The dumped txt file name
 */
LITE_API int LITE_enable_io_txt_dump(LiteNetwork network, const char* io_txt_out_file);

/**
 * \brief Dump input/output values of all internal variables to output
 * directory, in binary format
 * \param[in] network The loaded model
 * \param[in] io_bin_out_dir The dumped bin file directory
 */
LITE_API int LITE_enable_io_bin_dump(LiteNetwork network, const char* io_bin_out_dir);

/**
 * \brief get static peak memory info showed by Graph visualization
 * \param[in] log_dir The dumped json file directory
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_get_static_memory_alloc_info(
        LiteNetwork network, const char* log_dir);

/**
 * \brief enable the global layout transform optimization
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_enable_global_layout_transform(LiteNetwork network);

/**
 * \brief dump the model after the global layout transform optimization
 * \param[in] dump_file_path The model file path need to dump
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_dump_layout_transform_model(
        LiteNetwork network, const char* dump_file_path);

/**! get the model io information before model loaded by model path.
 * \param[in] model_path The model file path
 * \param[in] config The model config for loading
 * \param[out] ios The model io infermation
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_get_model_io_info_by_path(
        const char* model_path, const LiteConfig config, LiteNetworkIO* ios);

/** get the model io information before model loaded by model memory.
 * \param[in] model_mem The model memory ptr
 * \param[in] size The model memory ptr length
 * \param[in] config The model config for loading
 * \param[out] ios The model io infermation
 * \return int if the return is not zero, error happened, the error message
 * can get by LITE_get_last_error
 */
LITE_API int LITE_get_model_io_info_by_memory(
        const void* model_mem, size_t size, const LiteConfig config,
        LiteNetworkIO* ios);

#ifdef __cplusplus
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
