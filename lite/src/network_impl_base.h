/**
 * \file src/network_impl_base.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once

#include "lite/network.h"
#include "misc.h"
#include "tensor_impl_base.h"
#include "type_info.h"

#include <unordered_map>

namespace lite {

/*!
 * \brief the Inner IO data struct, add some inner data from IO
 */
class IOInner : public IO {
public:
    //! use to flag the corresponding lite_tensor is filled, when the
    //! value of lite_tensor is filled, the have_sync is true, other wise false,
    //! this is used in async mode
    bool have_sync = false;
    //! Real input and output data location
    std::shared_ptr<Tensor> lite_tensor = nullptr;

    IOInner() = default;
    IOInner(const IO& io) {
        name = io.name;
        is_host = io.is_host;
        io_type = io.io_type;
        config_layout = io.config_layout;
    }
};

/*!
 * \brief the realy network IO info when network run
 */
struct NetworkIOInner {
    std::vector<IOInner> inputs;
    std::vector<IOInner> outputs;
};

/*!
 * \brief implement the Network, contain the mgb related member
 */
class Network::NetworkImplBase : public DynTypeObj {
public:
    virtual ~NetworkImplBase() = default;

    //! set the config of the network, include:
    //! the inference device
    //! the other inference options, such as record_level, weight_preprocess...
    virtual void set_config(const Config& config) = 0;

    //! set the special io infomation, if not set, default io tensor will used,
    //! this is special for input/output is not host tensor, default the
    //! input/output tensors are host tensor
    virtual void set_io(const NetworkIO& network_io) = 0;

    //! only compute the output tensor in user configured
    virtual void compute_only_configured_output() = 0;

    //! get the network input and ouput tensor, the layout of which is
    //! sync from mge tensor
    virtual std::shared_ptr<Tensor> get_io_tensor(
            std::string io_name,
            LiteTensorPhase phase = LiteTensorPhase::LITE_IO) = 0;

    //! get the input tensor by index in the load_result tensormap
    virtual std::shared_ptr<Tensor> get_input_tensor(size_t index) = 0;

    //! get the output tensor by index in the load_result output_var_list
    virtual std::shared_ptr<Tensor> get_output_tensor(size_t index) = 0;

    //! get all the input tensor name in the order in load return
    virtual std::vector<const char*> get_all_input_name() const = 0;

    //! get all the output tensor name in the order in load return
    virtual std::vector<const char*> get_all_output_name() const = 0;

    //! get the input tensor name in the order in load return
    virtual const char* get_input_name(size_t index) const = 0;

    //! get the output tensor name in the order in load return
    virtual const char* get_output_name(size_t index) const = 0;

    //! set the callback in async model
    virtual void set_async_callback(const AsyncCallback& callback) = 0;

    //! set the start callback which will execute before network forward
    virtual void set_start_callback(const StartCallback& callback) = 0;

    //! set the finish callback which will execute after network forward
    virtual void set_finish_callback(const FinishCallback& callback) = 0;

    //! load the model and get the m_load_result
    virtual void load_model(std::shared_ptr<void> model_mem, size_t size,
                            std::unordered_map<std::string, LiteAny>
                                    separate_config_map = {}) = 0;

    //! forward the network with filled input data and fill the output data
    //! to the output tensor
    virtual void forward() = 0;

    //! in sync model, wait utile the inference finish
    virtual void wait() = 0;

    //! set device id, default device id = 0
    virtual void set_device_id(int device_id) = 0;
    virtual int get_device_id() const = 0;
    virtual LiteBackend get_backend_type() const = 0;
    //! set stream id, default stream id = 0
    virtual void set_stream_id(int stream_id) = 0;
    virtual int get_stream_id() const = 0;

    virtual LiteDeviceType get_device_type() const = 0;

    //! enable profile the network, a file will be generated
    virtual void enable_profile_performance(std::string profile_file_path) = 0;
};

/******************************** friend class *****************************/
/*!
 * \brief friend class of Network, for convenient accessing the Network members
 */
class NetworkHelper {
public:
    static bool loaded(const std::shared_ptr<Network> network) {
        LITE_ASSERT(network);
        return network->m_loaded;
    }
    static void loaded(const std::shared_ptr<Network> network, bool loaded) {
        LITE_ASSERT(network);
        network->m_loaded = loaded;
    }
    static Network::NetworkImplBase* implement(const Network* network) {
        LITE_ASSERT(network);
        return network->m_impl.get();
    }
    static Network::NetworkImplBase* implement(
            const std::shared_ptr<Network> network) {
        LITE_ASSERT(network);
        return network->m_impl.get();
    }
    static void implement(const std::shared_ptr<Network> network,
                          std::unique_ptr<Network::NetworkImplBase> impl) {
        LITE_ASSERT(network);
        network->m_impl = std::move(impl);
    }
};

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
