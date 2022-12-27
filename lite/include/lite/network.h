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
 * @brief the inference options which can optimize the network forwarding
 * performance
 *
 * @param weight_preprocess is the option which optimize the inference performance
 * with processing the weights of the network ahead
 *
 * @param fuse_preprocess fuse preprocess patten, like astype + pad_channel +
 * dimshuffle
 *
 * @param fake_next_exec  whether only to perform non-computing tasks (like
 * memory allocation and queue initialization) for next exec. This will be
 * reset to false when the graph is executed.
 *
 * @param var_sanity_check_first_run Disable var sanity check on the first run.
 * Var sanity check is enabled on the first-time execution by default, and can
 * be used to find some potential memory access errors in the operator
 *
 * @param const_shape used to reduce memory usage and improve performance since some
 * static inference data structures can be omitted and some operators can be
 * compute before forwarding
 *
 * @param force_dynamic_alloc force dynamic allocate memory for all vars
 *
 * @param force_output_dynamic_alloc force dynamic allocate memory for output tensor
 * which are used as the input of CallbackCaller Operator
 *
 * @param no_profiling_on_shape_change do not re-profile to select best implement
 * algo when input shape changes (use previous algo)
 *
 * @param jit_level Execute supported operators with JIT, please check with
 * MGB_JIT_BACKEND for more details, this value indicates JIT level.
 * 1: for JIT execute with basic elemwise operator
 * 2: for JIT execute elemwise and reduce operators
 *
 * @param record_level flags to optimize the inference performance with record the
 * kernel tasks in first run, hereafter the inference all need is to execute the
 * recorded tasks.
 * level = 0 means the normal inference,
 * level = 1 means use record inference,
 * level = 2 means record inference with free the extra memory
 *
 * @param graph_opt_level network optimization level:
 * 0: disable
 * 1: level-1: inplace arith transformations during graph
 *    construction
 * 2: level-2: level-1, plus global optimization before graph
 *    compiling
 * 3: also enable JIT
 *
 * @param async_exec_level level of dispatch on separate threads for different
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
    bool enable_f16_io_comp = false;  // convert to fp16
};

/**
 * @brief Configuration when load and compile a network
 *
 * @param has_compression flag whether the model is compressed, the compress
 * method is stored in the model
 *
 * @param device_id configure the device id of a network
 * @param device_type configure the device type of a network
 * @param backend configure the inference backend of a network, now only support
 * megengine
 *
 * @param bare_model_cryption_name is the bare model encryption method name, bare
 * model is not pack json information data inside
 *
 * @param options configuration of Options
 *
 * @param auto_optimize_inference lite will detect the device information add
 * set the options heuristically
 *
 * @param discrete_input_name configure which input is composed of discrete
 * multiple tensors
 */
struct LITE_API Config {
    bool has_compression = false;
    int device_id = 0;
    LiteDeviceType device_type = LiteDeviceType::LITE_CPU;
    LiteBackend backend = LiteBackend::LITE_DEFAULT;
    std::string bare_model_cryption_name = {};
    Options options = {};
    bool auto_optimize_inference = false;
    std::string discrete_input_name = {};
};

/*!
 * @brief Extra Configuration for a network
 *
 * @param disable_configure_by_model_info disable the configuration dumped with model,
 * if set true, all configuration in the model will not apply, users should configure
 * the network.
 */
struct LITE_API ExtraConfig {
    bool disable_configure_by_model_info = false;
};

/**
 * @brief config the network input and output item, the input and output tensor
 * information will describe there
 *
 * @param name the input/output tensor name
 *
 * @param is_host Used to mark where the input tensor comes from and where the output
 * tensor will copy to, if is_host is true, the input is from host and output copy
 * to host, otherwise in device. Sometimes the input is from device and output no need
 * copy to host, default is true.
 *
 * @param io_type The IO type, it can be SHAPE or VALUE, when SHAPE is set, the input or
 * output tensor value is invaid, only shape will be set, default is VALUE
 *
 * @param config_layout The layout of input or output tensor
 *
 * \verbatim embed:rst:leading-asterisk
 *
 *  .. note::
 *
 *      * if other layout is set to input tensor before forwarding, this layout will not
 *        work
 *      * if no layout is set before forwarding, the model will forward with its origin
 *        layout
 *      * if layout is set in output tensor, it will used to check whether the
 *        layout computed from the network is correct
 *
 * \endverbatim
 */
struct LITE_API IO {
    std::string name;

    bool is_host = true;

    LiteIOType io_type = LiteIOType::LITE_IO_VALUE;

    Layout config_layout = {};
};

/**
 * @brief the input and output information when load the network
 * the NetworkIO will remain in the network until the network is destroyed.
 *
 * @param inputs The all input tensors information that will configure to the network
 * @param outputs The all output tensors information that will configure to the network
 */
struct LITE_API NetworkIO {
    std::vector<IO> inputs = {};
    std::vector<IO> outputs = {};
};

/**
 * @brief A user-implemented allocator interface, user can register an allocator
 * to the megengine, then all the runtime memory will allocate by this allocator
 */
class LITE_API Allocator {
public:
    virtual ~Allocator() = default;

    /** @brief allocate memory of size in the given device with the given align
     *
     * @param device_type the device type the memory will allocate from
     * @param device_id the device id the memory will allocate from
     * @param size the byte size of memory will be allocated
     * @param align the align size require when allocate the memory
     */
    virtual void* allocate(
            LiteDeviceType device_type, int device_id, size_t size, size_t align) = 0;

    /** @brief free the memory pointed by ptr in the given device
     *
     * @param device_type the device type the memory will allocate from
     * @param device_id the device id the memory will allocate from
     * @param ptr the memory pointer to be free
     */
    virtual void free(LiteDeviceType device_type, int device_id, void* ptr) = 0;
};

/**
 * @brief the thread affinith callback function type
 *
 * @param thread_id the id of the current thread, the id is a number begin from 0 to
 * (nr_threads - 1), thread id of (nr_threads - 1) is the main worker thread.
 */
using ThreadAffinityCallback = std::function<void(int thread_id)>;

/**
 * @brief the network async callback function type
 */
using AsyncCallback = std::function<void(void)>;

/**
 * @brief the start/finish callback function type
 *
 * @param unordered_map map from the io tensor name to the pair of the
 * user configuration information and the really input or output tensor.
 */
//@{
using StartCallback =
        std::function<void(const std::unordered_map<
                           std::string, std::pair<IO, std::shared_ptr<Tensor>>>&)>;
using FinishCallback =
        std::function<void(const std::unordered_map<
                           std::string, std::pair<IO, std::shared_ptr<Tensor>>>&)>;
//@}

/**
 * @brief The network is the main class to perform forwarding, which is construct form a
 * model, and implement model load, init, forward, and display some model information
 */
class LITE_API Network {
public:
    class NetworkImplBase;
    friend class NetworkHelper;

    ~Network();

    /*! @brief Construct a network with given configuration and IO information
     *
     * @name Constructor
     *
     * @param config  The configuration to create the network
     * @param networkio The NetworkIO to describe the input and output
     * tensor of the network
     */
    //@{
    Network(const Config& config = {}, const NetworkIO& networkio = {});
    Network(const NetworkIO& networkio, const Config& config = {});
    //@}

    //! load the model form memory
    void load_model(void* model_mem, size_t size);

    //! load the model from a model path
    void load_model(std::string model_path);

    //! only compute the output tensor configured by the IO information
    void compute_only_configured_output();

    /** @brief get the network input and output tensor, the layout of which is
     * sync from megengine tensor, when the name of input and output tensor are the
     * same, use LiteTensorPhase to separate them
     *
     * @param io_name the name of the tensor
     * @param phase indicate whether the tensor is input tensor or output tensor,
     * maybe the input tensor name is the same with the output tensor name
     */
    std::shared_ptr<Tensor> get_io_tensor(
            std::string io_name, LiteTensorPhase phase = LiteTensorPhase::LITE_IO);

    /** @brief get the network input tensors which input consists of discrete multiple
     * tensors, layout (1, c, h, w)
     *
     * @param io_name the name of the tensor
     * @param phase indicate the tensor is input tensor
     */
    std::vector<std::shared_ptr<Tensor>> get_discrete_tensors(
            std::string io_name, LiteTensorPhase phase = LiteTensorPhase::LITE_INPUT);

    //! get the network input tensor by index
    std::shared_ptr<Tensor> get_input_tensor(size_t index);

    //! get the network input tensors which input consists of discrete multiple tensors
    //! by index
    std::vector<std::shared_ptr<Tensor>> get_input_tensors(size_t index);

    //! get the network output tensor by index
    std::shared_ptr<Tensor> get_output_tensor(size_t index);

    //! set the network forwarding in async mode and set the AsyncCallback callback
    //! function
    Network& set_async_callback(const AsyncCallback& async_callback);

    //! set the start forwarding callback function of type StartCallback, which will be
    //! execute before forward. this can be used to check network input or dump model
    //! inputs for debug
    Network& set_start_callback(const StartCallback& start_callback);

    //! set the finish forwarding callback function of type FinishCallback, which will
    //! be execute after forward. this can be used to dump model outputs for debug
    Network& set_finish_callback(const FinishCallback& finish_callback);

    //! forward the network with filled input data and fill the output data
    //! to the output tensor
    void forward();

    //! waite until forward finish in sync model
    void wait();

    //! get the input tensor name by index
    std::string get_input_name(size_t index) const;

    //! get the output tensor name by index
    std::string get_output_name(size_t index) const;

    //! get all the input tensor names
    std::vector<std::string> get_all_input_name() const;

    //! get all the output tensor names
    std::vector<std::string> get_all_output_name() const;

    //! set the network forwarding device id, default device id = 0
    Network& set_device_id(int device_id);

    //! get the network forwarding device id
    int get_device_id() const;

    //! set the network stream id, default stream id = 0
    Network& set_stream_id(int stream_id);

    //! get the network stream id
    int get_stream_id() const;

    //! enable profile the network, a file will be generated to the given path
    void enable_profile_performance(std::string profile_file_path);

    //! get model extra info, the extra information is packed into model by user
    const std::string& get_model_extra_info();

    //! get the network device type
    LiteDeviceType get_device_type() const;

    //! get static peak memory info showed by Graph visualization
    void get_static_memory_alloc_info(const std::string& log_dir = "logs/test") const;

    /** @brief the extra configuration
     *
     * @param extra_config the extra configuration to set into the network
     */
    void extra_configure(const ExtraConfig& extra_config);

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
    ExtraConfig m_extra_config;
    NetworkIO m_network_io;
    std::unique_ptr<NetworkImplBase> m_impl;
    std::string m_extra_info;
};

/*********************** MGE special network function ***************/
/*!
 * @brief All the runtime configuration function is define in Runtime class, as
 * a static member function
 */
class LITE_API Runtime {
public:
    /** @brief The multithread number setter and getter interface
     * When device is CPU, this interface will set the network
     * running in multi thread mode with the given thread number.
     *
     * @param dst_network the target network to set/get the thread number
     * @param nr_threads the thread number set to the target network
     */
    //@{
    static void set_cpu_threads_number(
            std::shared_ptr<Network> dst_network, size_t nr_threads);
    static size_t get_cpu_threads_number(std::shared_ptr<Network> dst_network);
    //@}

    /** @brief set threads affinity callback
     *
     * @param dst_network the target network to set the thread affinity callback
     * @param thread_affinity_callback the ThreadAffinityCallback callback to set the
     * thread affinity
     */
    static void set_runtime_thread_affinity(
            std::shared_ptr<Network> network,
            const ThreadAffinityCallback& thread_affinity_callback);

    /** @brief Set cpu default mode when device is CPU, in some low computation
     * device or single core device, this mode will get good performace
     *
     * @param dst_network the target network to set/get cpu inplace model
     */
    //@{
    static void set_cpu_inplace_mode(std::shared_ptr<Network> dst_network);
    static bool is_cpu_inplace_mode(std::shared_ptr<Network> dst_network);
    //@}

    //! Set the network forwarding use tensorrt
    static void use_tensorrt(std::shared_ptr<Network> dst_network);

    /** @brief set opr algorithm selection strategy in the target network
     *
     * @param dst_network the target network to set the algorithm strategy
     * @param strategy the algorithm strategy will set to the network, if multi
     * strategy should set, use | operator can pack them together
     * @param shared_batch_size the batch size used by fast-run, Non-zero value means
     * that fast-run use this batch size regardless of the batch size of the model, if
     * set to zero means fast-run use batch size of the model
     *
     * @param binary_equal_between_batch if set true means if the content of each input
     * batch is binary equal, whether the content of each output batch is promised to be
     * equal, otherwise not
     */
    static void set_network_algo_policy(
            std::shared_ptr<Network> dst_network, LiteAlgoSelectStrategy strategy,
            uint32_t shared_batch_size = 0, bool binary_equal_between_batch = false);

    /** @brief set the opr workspace limitation in the target network, some opr
     * maybe use large of workspace to get good performance, set workspace limitation
     * can save memory but may influence the performance
     *
     * @param dst_network the target network to set/get workspace limitation
     * @param workspace_limit the byte size of workspace limitation
     */
    static void set_network_algo_workspace_limit(
            std::shared_ptr<Network> dst_network, size_t workspace_limit);

    /** @brief set the network runtime memory Allocator, the Allocator is defined by
     * user, through this method, user can implement a memory pool for network
     * forwarding
     *
     * @param dst_network the target network
     * @param user_allocator the user defined Allocator
     */
    static void set_memory_allocator(
            std::shared_ptr<Network> dst_network,
            std::shared_ptr<Allocator> user_allocator);

    /** @brief share the runtime memory with other network, the weights is not shared
     *
     * \verbatim embed:rst:leading-asterisk
     *
     *  .. warning::
     *
     *     the src network and the dst network can not execute in simultaneous
     *
     * \endverbatim
     *
     * @param dst_network the target network to share the runtime memory from
     * src_network
     * @param src_network the source network to shared runtime memory to dst_network
     */
    static void share_runtime_memory_with(
            std::shared_ptr<Network> dst_network, std::shared_ptr<Network> src_network);

    /** @brief dump all input/output tensor of all operators to the output file, in txt
     * format, user can use this function to debug compute error
     *
     * @param dst_network the target network to dump its tensors
     * @param io_txt_out_file the txt file
     */
    static void enable_io_txt_dump(
            std::shared_ptr<Network> dst_network, std::string io_txt_out_file);

    /** @brief dump all input/output tensor of all operators to the output file, in
     * binary format, user can use this function to debug compute error
     *
     * @param dst_network the target network to dump its tensors
     * @param io_bin_out_dir the binary file director
     */
    static void enable_io_bin_dump(
            std::shared_ptr<Network> dst_network, std::string io_bin_out_dir);

    /** @brief load a new network which will share weights with src network,
     * this can reduce memory usage when user want to load the same model multi
     * times
     *
     * @param dst_network the target network to share weights from src_network
     * @param src_network the source network to shared weights to dst_network
     */
    static void shared_weight_with_network(
            std::shared_ptr<Network> dst_network,
            const std::shared_ptr<Network> src_network);

    /** @brief set global layout transform optimization for network, global
     * layout optimization can auto determine the layout of every operator in
     * the network by profile, thus it can improve the performance of the
     * network forwarding
     */
    static void enable_global_layout_transform(std::shared_ptr<Network> network);

    /** @brief dump network after global layout transform optimization to the
     * specific path
     */
    static void dump_layout_transform_model(
            std::shared_ptr<Network> network, std::string optimized_model_path);

    /** @brief get the model io information before model loaded by model path.
     *
     * @param model_path the model path to get the model IO information
     * @param config the model configuration
     *
     * @return the model NetworkIO information
     */
    static NetworkIO get_model_io_info(
            const std::string& model_path, const Config& config = {});

    /** @brief get the model io information before model loaded by model memory.
     *
     * @param model_mem the model memory to get the model IO information
     * @param size model memory size in byte
     * @param config the model configuration
     *
     * @return the model NetworkIO information
     */
    static NetworkIO get_model_io_info(
            const void* model_mem, size_t size, const Config& config = {});
};

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
