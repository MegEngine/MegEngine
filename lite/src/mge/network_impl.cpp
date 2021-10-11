/**
 * \file src/mge/network_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "common.h"
#include "lite/network.h"
#include "memory_allocator.h"
#include "network_impl.h"
#include "parse_info/parse_info_base.h"
#include "parse_model/model_parser.h"

#include "megbrain/common.h"
#include "megbrain/comp_node.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/graph.h"
#include "megbrain/graph/cg.h"
#include "megbrain/opr/io.h"
#include "megbrain/tensor.h"

#if MGB_OPENCL
#include "megcore_opencl.h"
#endif

#include <fstream>
#include <memory>
#include <set>

using namespace lite;
using namespace mgb;

LITE_DYN_TYPE_OBJ_FINAL_IMPL(NetworkImplDft);

void NetworkImplDft::set_config(const Config& config) {
    m_user_config = std::make_unique<Config>();
    *m_user_config = config;
    m_load_config.comp_graph = mgb::ComputingGraph::make();
    m_compnode_locator = to_compnode_locator(m_user_config->device_type);
    m_compnode_locator.device = config.device_id;
}

void NetworkImplDft::shared_weight_with(const NetworkImplBase* src_network) {
    application_config();
    const auto& src_impl = src_network->cast_final_safe<NetworkImplDft>();
    LITE_ASSERT(src_impl.m_loader, "Clone network must after the network is loaded.");
    m_load_result = src_impl.m_loader->load(m_load_config, true);

    //! flag weather the mode is cross compnode model
    cross_compnode_model_detect();

    //! update the IO of the network
    update_io();

    //! replace the IO when there is device input or output
    compile_graph();
}

void NetworkImplDft::application_config() {
    auto device_type = m_user_config->device_type;
    m_compnode_locator.type = to_compnode_locator(device_type).type;
    m_compnode_locator.device = m_user_config->device_id;
    if (m_nr_threads > 1 && device_type == LiteDeviceType::LITE_CPU) {
        m_compnode_locator.type = mgb::CompNode::DeviceType::MULTITHREAD;
        m_compnode_locator.device = m_user_config->device_id;
    }
    //! model options
#define ConfigOption(mge_name, lite_name) \
    options.mge_name = m_user_config->options.lite_name;

    auto&& options = m_load_config.comp_graph->options();
    ConfigOption(graph_opt.weight_preprocess, weight_preprocess);
    ConfigOption(graph_opt.fuse_preprocess, fuse_preprocess);
    ConfigOption(fake_next_exec, fake_next_exec);
    ConfigOption(var_sanity_check_first_run, var_sanity_check_first_run);
    m_load_config.const_var_shape = m_user_config->options.const_shape;
    ConfigOption(force_dynamic_alloc, force_dynamic_alloc);
    ConfigOption(force_output_dynamic_alloc, force_output_dynamic_alloc);
    ConfigOption(no_profiling_on_shape_change, no_profiling_on_shape_change);
    LITE_ASSERT(
            m_user_config->options.jit_level == 0 ||
                    (m_user_config->options.jit_level > 0 &&
                     device_type == LiteDeviceType::LITE_CUDA),
            "jit only support in cuda device.");
    ConfigOption(graph_opt.jit, jit_level);
    ConfigOption(comp_node_seq_record_level, comp_node_seq_record_level);
    ConfigOption(graph_opt_level, graph_opt_level);
    ConfigOption(async_exec_level, async_exec_level);

#undef ConfigOption
#define ConfigOptionLayoutTransform(name) \
    if (m_user_config->options.name) {    \
        options.graph_opt.name();         \
    }
    ConfigOptionLayoutTransform(enable_nchw44);
    ConfigOptionLayoutTransform(enable_nchw44_dot);
    ConfigOptionLayoutTransform(enable_nchw88);
    ConfigOptionLayoutTransform(enable_nhwcd4);
    ConfigOptionLayoutTransform(enable_nchw4);
    ConfigOptionLayoutTransform(enable_nchw32);
    ConfigOptionLayoutTransform(enable_nchw64);
#undef ConfigOptionLayoutTransform
    if (m_user_config->has_compression) {
        m_load_config.tensor_value_loader = decompressed_tensor_value_loader;
    }

    //! if device is LITE_NONE, the compnode information is stored in model
    if (device_type != LiteDeviceType::LITE_DEVICE_DEFAULT) {
        //! currently not set Locator type because an atlas mgb model is a
        //! cross-compnode graph
        if (device_type == LiteDeviceType::LITE_ATLAS) {
            m_load_config.comp_node_mapper = [this](mgb::CompNode::Locator& loc) {
                if (loc.type == mgb::CompNode::DeviceType::ATLAS) {
                    loc.device = m_compnode_locator.device;
                    loc.stream = m_compnode_locator.stream;
                } else if (loc.type == mgb::CompNode::DeviceType::MULTITHREAD) {
                    loc.stream = m_nr_threads;
                }
            };
        } else {
            m_load_config.comp_node_mapper = [this](mgb::CompNode::Locator& loc) {
                loc = m_compnode_locator;
            };
        }
    }
}

void NetworkImplDft::set_memory_allocator(std::shared_ptr<Allocator> user_allocator) {
    auto allocator = std::make_shared<UserStaticMemAlloc>(user_allocator);
    LITE_ASSERT(m_load_config.comp_graph);
    m_load_config.comp_graph->set_device_memory_allocator(allocator);
}

//! share the runtime memory with other network, the weights is not shared
void NetworkImplDft::share_runtime_memory_with(Network::NetworkImplBase* network_impl) {
    LITE_ASSERT(network_impl);
    LITE_ASSERT(m_load_config.comp_graph);
    m_load_config.comp_graph->share_device_memory_with(*(
            network_impl->cast_final_safe<NetworkImplDft>().m_load_config.comp_graph));
}

void NetworkImplDft::set_cpu_inplace_mode() {
    LITE_ASSERT(
            m_user_config->device_type == LiteDeviceType::LITE_CPU,
            "cpu inplace mode is only avaliable in CPU.");
    m_is_cpu_inplace_mode = true;
    if (m_compnode_locator.type == mgb::CompNode::DeviceType::CPU) {
        m_compnode_locator.device = mgb::CompNode::Locator::DEVICE_CPU_DEFAULT;
    } else {
        LITE_ASSERT(
                m_compnode_locator.type == CompNode::DeviceType::MULTITHREAD,
                "cpu inplace mode is only avaliable in CPU.");
        m_compnode_locator.device = mgb::CompNode::Locator::DEVICE_MULTITHREAD_DEFAULT;
    }
}

void NetworkImplDft::set_cpu_threads_number(size_t nr_threads) {
    LITE_ASSERT(
            m_user_config->device_type == LiteDeviceType::LITE_CPU,
            "multi threads mode is only avaliable in CPU.");
    if (nr_threads > 1) {
        m_nr_threads = nr_threads;
        m_compnode_locator.type = mgb::CompNode::DeviceType::MULTITHREAD;
        m_compnode_locator.nr_threads = nr_threads;
    }
}

void NetworkImplDft::set_runtime_thread_affinity(
        const ThreadAffinityCallback& thread_affinity_callback) {
    LITE_ASSERT(
            m_user_config->device_type == LiteDeviceType::LITE_CPU,
            "multi threads mode is only avaliable in CPU.");
    mgb::CompNode::Locator loc;
    m_load_config.comp_node_mapper(loc);
    auto cn = mgb::CompNode::load(loc);
    if (m_nr_threads > 1) {
        mgb::CompNodeEnv::from_comp_node(cn).cpu_env().set_affinity(
                thread_affinity_callback);
    } else {
        mgb::CompNodeEnv::from_comp_node(cn).cpu_env().dispatch(
                [thread_affinity_callback](void) { thread_affinity_callback(0); });
    }
}

void NetworkImplDft::set_device_id(int device_id) {
    m_compnode_locator.device = device_id;
    m_user_config->device_id = device_id;
}

void NetworkImplDft::set_stream_id(int stream_id) {
    m_compnode_locator.stream = stream_id;
}

void NetworkImplDft::use_tensorrt() {
    auto&& options = m_load_config.comp_graph->options();
    options.graph_opt.tensorrt = true;
}

//! set the callback in async model
void NetworkImplDft::set_async_callback(const AsyncCallback& callback) {
    LITE_ASSERT(!m_is_cpu_inplace_mode, "cpu inplace mode not support async mode");
    LITE_ASSERT(
            m_user_config->device_type == LiteDeviceType::LITE_CPU ||
                    m_user_config->device_type == LiteDeviceType::LITE_CUDA,
            "Now only cpu and cuda>10.0 support async mode");
    m_async = true;
    m_async_callback = std::move(callback);
}

void NetworkImplDft::make_output_spec() {
    m_output_spec.clear();
    for (auto&& out : m_network_io->outputs) {
        if (m_load_result.output_var_map.count(out.name)) {
            auto&& load_out = m_load_result.output_var_map[out.name];
            auto cb = [&out, this](const mgb::DeviceTensorND& dv) mutable {
                mgb::CompNode comp_node = dv.comp_node();
                if (out.io_type == LiteIOType::LITE_IO_SHAPE) {
                    auto mgb_layout = dv.layout();
                    out.lite_tensor->set_layout(to_lite_layout(mgb_layout));
                } else {
                    TensorHelper::implement(out.lite_tensor)
                            ->cast_final_safe<TensorImplDft>()
                            .copy_from_mge_tensor(dv);
                    out.lite_tensor->update_from_implement();
                }
                if (m_async) {
                    out.have_sync = true;
                    bool need_exec_cb = true;
                    for (auto&& j : m_network_io->outputs) {
                        if (!j.have_sync) {
                            need_exec_cb = false;
                        }
                    }
                    if (need_exec_cb) {
                        for (auto&& j : m_network_io->outputs) {
                            j.have_sync = false;
                        }
                        comp_node.add_callback([this]() { finish(); });
                    }
                }
            };
            m_output_spec.emplace_back(load_out, std::move(cb));
        } else {
            LITE_THROW(ssprintf("no output named : %s in the mode", out.name.c_str()));
        }
    }
}

void NetworkImplDft::replace_dev_input_pass() {
    mgb::CompNode::Locator locator;
    m_load_config.comp_node_mapper(locator);
    //! CPU is not need use device input
    if (locator.type == mgb::CompNode::DeviceType::CPU) {
        return;
    }
    //! repalce the H2D with VolatileSharedDeviceTensor, and keep the dev tensor
    //! in m_network_io.input, user can directly change the dev tensor
    //! storage through m_network_io.input.lite_tensor->reset() befor forward
    using DeviceTensorMap =
            std::unordered_map<std::string, std::shared_ptr<mgb::DeviceTensorND>>;
    DeviceTensorMap name2dev_tensor;

    mgb::ThinHashMap<mgb::HostTensorND*, mgb::SymbolVar> host_val2var;

    //! construct host_val2var that maps from host tensor to corresponding var
    auto on_opr = [&](mgb::cg::OperatorNodeBase* opr) {
        if (opr->same_type<mgb::opr::Host2DeviceCopy>()) {
            mgb::HostTensorND* tensor =
                    opr->cast_final<mgb::opr::Host2DeviceCopy>().host_data().get();
            host_val2var[tensor] = opr->output(0);
        }
    };
    mgb::cg::DepOprIter dep_iter{on_opr};
    for (auto i : m_load_result.output_var_list) {
        dep_iter.add(i.node()->owner_opr());
    }

    mgb::ThinHashMap<mgb::SymbolVar, mgb::SymbolVar> inp_var_map, out_var_map;

    mgb::SmallVector<std::string> to_clear;
    for (auto&& config_in : m_network_io->inputs) {
        if (!config_in.is_host) {
            auto host_val = m_load_result.tensor_map[config_in.name];
            auto dev_val = TensorHelper::implement(config_in.lite_tensor)
                                   ->cast_final_safe<TensorImplDft>()
                                   .m_dev_tensor;
            auto dev_var = mgb::opr::VolatileSharedDeviceTensor::make(
                    *m_load_result.graph, dev_val, {config_in.name});
            inp_var_map[host_val2var.at(host_val.get())] = dev_var;
            name2dev_tensor[config_in.name] = dev_val;
        }
    }
    auto new_ovar = mgb::cg::replace_vars(m_load_result.output_var_list, inp_var_map);
    for (size_t i = 0; i < new_ovar.size(); ++i) {
        out_var_map[m_load_result.output_var_list[i]] = new_ovar[i];
    }
    for (auto&& i : m_load_result.output_var_map) {
        i.second = out_var_map.at(i.second);
    }
    for (auto&& i : m_load_result.output_var_map_id) {
        i.second = out_var_map.at(i.second);
    }
    for (size_t i = 0; i < m_load_result.output_var_list.size(); i++) {
        new_ovar[i].rename(m_load_result.output_var_list[i].node()->name());
    }
    m_load_result.output_var_list = std::move(new_ovar);
}

void NetworkImplDft::cross_compnode_model_detect() {
    mgb::ThinHashSet<LiteDeviceType> nr_used_device_type;
    auto on_opr = [&](mgb::cg::OperatorNodeBase* opr) {
        for (auto j : opr->output()) {
            if (j->comp_node() != mgb::CompNode::default_cpu()) {
                nr_used_device_type.insert(
                        get_device_from_locator(j->comp_node().locator()));
            }
        }
    };
    mgb::cg::DepOprIter dep_iter{on_opr};
    for (auto i : m_load_result.output_var_list) {
        dep_iter.add(i.node()->owner_opr());
    }
    m_nr_device_type = nr_used_device_type.size();
}

void NetworkImplDft::load_model(
        std::shared_ptr<void> model_mem, size_t size,
        std::unordered_map<std::string, LiteAny> separate_config_map) {
    if (!m_loader) {
        m_input_file =
                mgb::serialization::InputFile::make_mem_proxy(model_mem, size, false);
        auto format = mgb::serialization::GraphLoader::identify_graph_dump_format(
                *m_input_file);
        if (!format.valid()) {
            LITE_THROW("invalid model format");
        }
        m_loader = mgb::serialization::GraphLoader::make(
                std::move(m_input_file), format.val());
    }

    //! applay the user configration to mge model
    application_config();

    //! config some flag get from json config file
    if (separate_config_map.find("device_id") != separate_config_map.end()) {
        set_device_id(separate_config_map["device_id"].unsafe_cast<int>());
    }
    if (separate_config_map.find("number_threads") != separate_config_map.end() &&
        separate_config_map["number_threads"].unsafe_cast<size_t>() > 1) {
        set_cpu_threads_number(
                separate_config_map["number_threads"].unsafe_cast<size_t>());
    }
    if (separate_config_map.find("enable_inplace_model") != separate_config_map.end() &&
        separate_config_map["enable_inplace_model"].unsafe_cast<bool>()) {
        set_cpu_inplace_mode();
    }
    if (separate_config_map.find("use_tensorrt") != separate_config_map.end() &&
        separate_config_map["use_tensorrt"].unsafe_cast<bool>()) {
        use_tensorrt();
    }

    m_load_result = m_loader->load(m_load_config, true);

    cross_compnode_model_detect();

    //! update the IO of the network
    update_io();

    //! replace the IO when there is device input or output
    compile_graph();
}

void NetworkImplDft::compile_graph() {
    modify_exection_policy();
    replace_dev_input_pass();
    make_output_spec();
    m_execute_func = m_load_result.graph_compile(m_output_spec);
}

void NetworkImplDft::start() const {
    if (m_start_callback) {
        std::unordered_map<std::string, std::pair<IO, std::shared_ptr<Tensor>>>
                input_io_map;
        for (auto&& io_inner : m_network_io->inputs) {
            input_io_map[io_inner.name] = {
                    IO{io_inner.name, io_inner.is_host, io_inner.io_type,
                       io_inner.config_layout},
                    io_inner.lite_tensor};
        }
        m_start_callback(input_io_map);
    }
}

void NetworkImplDft::forward() {
    start();
    LITE_ASSERT(m_execute_func, "forward must be called after network loaded.");
    m_execute_func->execute();
}

void NetworkImplDft::wait() {
    if (!m_async) {
        m_execute_func->wait();
    }
    finish();
}

void NetworkImplDft::finish() const {
    if (m_async) {
        LITE_ASSERT(m_async_callback, "The callback func must set when async mode.");
        m_async_callback();
    }
    if (m_finish_callback) {
        std::unordered_map<std::string, std::pair<IO, std::shared_ptr<Tensor>>>
                output_io_map;
        for (auto&& io_inner : m_network_io->outputs) {
            output_io_map[io_inner.name] = {
                    IO{io_inner.name, io_inner.is_host, io_inner.io_type,
                       io_inner.config_layout},
                    io_inner.lite_tensor};
        }
        m_finish_callback(output_io_map);
    }
    output_plugin_result();
}

void NetworkImplDft::set_io(const NetworkIO& network_io) {
    m_network_io = std::make_unique<NetworkIOInner>();
    for (auto&& in : network_io.inputs) {
        m_network_io->inputs.emplace_back(in);
    }
    for (auto&& out : network_io.outputs) {
        m_network_io->outputs.emplace_back(out);
    }
}

void NetworkImplDft::try_infer_tensor_layout(
        std::shared_ptr<Tensor> tensor, mgb::cg::SymbolVar var) {
    auto&& static_infer_mgr = m_load_config.comp_graph->static_infer_manager();
    auto infer_trait = var.node()->get_static_infer_trait();
    if (std::get<0>(infer_trait)) {
        auto shape = static_infer_mgr.infer_shape_fallible(var.node());
        if (!shape) {
            LITE_WARN(
                    "Lite infer output shape failed, maybe the model is "
                    "dynamic "
                    "shape.\n");
            return;
        }
        Layout layout = to_lite_layout(mgb::TensorLayout{*shape, var.dtype()});
        tensor->set_layout(layout);
    }
}

void NetworkImplDft::update_io() {
    update_input();
    update_output();
}

void NetworkImplDft::update_input() {
    auto device_type = m_user_config->device_type;
    auto device_id = m_compnode_locator.device;
    auto stream_id = m_compnode_locator.stream;
    //! if cpu all input and output are host
    if (device_type == LiteDeviceType::LITE_CPU) {
        for (auto&& in : m_network_io->inputs) {
            in.is_host = true;
        }
    }
    //! if cross compnode model, modify the device input if it is not valid
    if (m_nr_device_type > 1) {
        for (auto&& in_tensor_iter : m_load_result.tensor_map) {
            for (auto&& config_in : m_network_io->inputs) {
                //! if tensor is set to device input
                if (in_tensor_iter.first == config_in.name && !config_in.is_host) {
                    //! if the origin compnode of the tensor is not the device,
                    //! set the input to host
                    if (get_device_from_locator(
                                in_tensor_iter.second->comp_node().locator()) ==
                        LiteDeviceType::LITE_CPU) {
                        config_in.is_host = true;
                        LITE_WARN(
                                "The input tensor %s of the cross device model "
                                "should not from device.",
                                config_in.name.c_str());
                    }
                }
            }
        }
    }
    for (auto&& in_tensor_iter : m_load_result.tensor_map) {
        bool found = false;
        for (auto&& config_in : m_network_io->inputs) {
            if (in_tensor_iter.first == config_in.name) {
                found = true;
                if (config_in.is_host) {
                    config_in.lite_tensor = std::make_shared<Tensor>(
                            device_id, stream_id, device_type, true);
                    TensorHelper::implement(config_in.lite_tensor)
                            ->cast_final_safe<TensorImplDft>()
                            .m_host_tensor = in_tensor_iter.second;
                    config_in.lite_tensor->update_from_implement();
                } else {
                    config_in.lite_tensor =
                            std::make_shared<Tensor>(device_id, stream_id, device_type);
                    config_in.lite_tensor->set_layout(
                            to_lite_layout(in_tensor_iter.second->layout()));
                }
                if (config_in.config_layout.ndim &&
                    !(config_in.config_layout == config_in.lite_tensor->get_layout())) {
                    config_in.lite_tensor->set_layout(config_in.config_layout);
                }
            }
        }
        if (!found) {
            IOInner io_in;
            io_in.name = in_tensor_iter.first;
            io_in.lite_tensor =
                    std::make_shared<Tensor>(device_id, stream_id, device_type, true);
            TensorHelper::implement(io_in.lite_tensor)
                    ->cast_final_safe<TensorImplDft>()
                    .m_host_tensor = in_tensor_iter.second;
            io_in.lite_tensor->update_from_implement();
            m_network_io->inputs.push_back(io_in);
        }
    }
    //! delete the IO that is not the network
    for (auto it = m_network_io->inputs.begin(); it != m_network_io->inputs.end();) {
        if (it->lite_tensor == nullptr) {
            LITE_LOG("%s is not the network input, ignore it.", it->name.c_str());
            it = m_network_io->inputs.erase(it);
        } else {
            it++;
        }
    }
}

void NetworkImplDft::update_output() {
    auto device_type = m_user_config->device_type;
    auto device_id = m_compnode_locator.device;
    auto stream_id = m_compnode_locator.stream;
    if (device_type == LiteDeviceType::LITE_CPU) {
        for (auto&& out : m_network_io->outputs) {
            out.is_host = true;
        }
    }
    //! delete the output that is not the network
    for (auto out_it = m_network_io->outputs.begin();
         out_it != m_network_io->outputs.end();) {
        if (std::find_if(
                    m_load_result.output_var_list.begin(),
                    m_load_result.output_var_list.end(),
                    [out_it](const mgb::SymbolVar var) {
                        return var.node()->name() == out_it->name;
                    }) == m_load_result.output_var_list.end()) {
            LITE_LOG("%s is not the network output, ignore it.", out_it->name.c_str());
            out_it = m_network_io->outputs.erase(out_it);
        } else {
            out_it++;
        }
    }
    //! user config the output tensor, so only compute the config output
    if (m_compute_configured_output_only) {
        LITE_ASSERT(
                m_network_io->outputs.size() > 0,
                "compute configured output only with no configure output.");
        for (auto out_it = m_network_io->outputs.begin();
             out_it != m_network_io->outputs.end(); out_it++) {
            //! use pinned memory to copy form device
            if (out_it->is_host) {
                out_it->lite_tensor = std::make_shared<Tensor>(
                        device_id, stream_id, device_type, true);
            } else {
                out_it->lite_tensor =
                        std::make_shared<Tensor>(device_id, stream_id, device_type);
            }
            mgb::SymbolVar var;
            for (auto&& out_var : m_load_result.output_var_list) {
                if (out_var.node()->name() == out_it->name) {
                    var = out_var;
                    break;
                }
            }
            try_infer_tensor_layout(out_it->lite_tensor, var);
        }
        //! user not set, use default output
    } else {
        for (auto&& out : m_load_result.output_var_list) {
            auto it = std::find_if(
                    m_network_io->outputs.begin(), m_network_io->outputs.end(),
                    [&out](const IOInner io) { return io.name == out.node()->name(); });
            if (it != m_network_io->outputs.end()) {
                if (it->is_host) {
                    it->lite_tensor = std::make_shared<Tensor>(
                            device_id, stream_id, device_type, true);
                } else {
                    it->lite_tensor =
                            std::make_shared<Tensor>(device_id, stream_id, device_type);
                }
                try_infer_tensor_layout(it->lite_tensor, out);
            } else {
                IOInner output;
                output.name = out.node()->name();
                output.lite_tensor = std::make_shared<Tensor>(
                        device_id, stream_id, device_type, true);
                m_network_io->outputs.push_back({output});
                try_infer_tensor_layout(output.lite_tensor, out);
            }
        }
    }
}

std::shared_ptr<Tensor> NetworkImplDft::get_io_tensor(
        std::string io_name, LiteTensorPhase phase) {
    if (phase == LiteTensorPhase::LITE_INPUT || phase == LiteTensorPhase::LITE_IO) {
        for (auto&& config_in : m_network_io->inputs) {
            if (io_name == config_in.name) {
                return config_in.lite_tensor;
            }
        }
    }
    if (phase == LiteTensorPhase::LITE_OUTPUT || phase == LiteTensorPhase::LITE_IO) {
        for (auto&& config_out : m_network_io->outputs) {
            if (io_name == config_out.name) {
                config_out.lite_tensor->update_from_implement();
                return config_out.lite_tensor;
            }
        }
    }
    LITE_THROW(mgb::ssprintf(
            "tensor name must be %s input tensor name or the registered "
            "output tensor name if NetworkIO is set, if NetworkIO is not set, "
            "the output tensor is all the network output tensor, or the output "
            "tensor is only the registered tensor.",
            io_name.c_str()));
    return nullptr;
}

std::shared_ptr<Tensor> NetworkImplDft::get_input_tensor(size_t index) {
    return get_io_tensor(get_input_name(index));
}

std::shared_ptr<Tensor> NetworkImplDft::get_output_tensor(size_t index) {
    return get_io_tensor(get_output_name(index));
}

//! set opr algorithm selection strategy in the network
void NetworkImplDft::set_network_algo_policy(
        LiteAlgoSelectStrategy strategy, uint32_t shared_batch_size,
        bool binary_equal_between_batch) {
    using S = megdnn::param::ExecutionPolicy::Strategy;
    auto dst_strategy = static_cast<S>(0);
    if (static_cast<uint32_t>(strategy) & LiteAlgoSelectStrategy::LITE_ALGO_HEURISTIC) {
        dst_strategy = dst_strategy | S::HEURISTIC;
    }
    if (static_cast<uint32_t>(strategy) & LiteAlgoSelectStrategy::LITE_ALGO_PROFILE) {
        dst_strategy = dst_strategy | S::PROFILE;
    }
    if (static_cast<uint32_t>(strategy) &
        LiteAlgoSelectStrategy::LITE_ALGO_REPRODUCIBLE) {
        dst_strategy = dst_strategy | S::REPRODUCIBLE;
    }
    if (static_cast<uint32_t>(strategy) & LiteAlgoSelectStrategy::LITE_ALGO_OPTIMIZED) {
        dst_strategy = dst_strategy | S::OPTIMIZED;
    }
    m_execution_policy = dst_strategy;

    auto&& fast_run_config = m_load_config.comp_graph->options().fast_run_config;
    fast_run_config.binary_equal_between_batch = binary_equal_between_batch;
    fast_run_config.shared_batch_size = shared_batch_size;

    if (m_execute_func) {
        LITE_WARN(
                "set_network_algo_policy maybe cause error after loaded "
                "network!!!!");
        modify_exection_policy();
    }
}

void NetworkImplDft::modify_exection_policy() {
    mgb::SymbolVarArray vars;
    for (auto i : m_output_spec) {
        vars.push_back(i.first);
    }
    if (static_cast<uint32_t>(m_execution_policy) != 0)
        mgb::gopt::modify_opr_algo_strategy_inplace(vars, m_execution_policy);
}

//! set opr algorithm selection strategy in the network
void NetworkImplDft::set_network_algo_workspace_limit(size_t workspace_limit) {
    mgb::SymbolVarArray vars;
    for (auto i : m_output_spec) {
        vars.push_back(i.first);
    }
    mgb::gopt::set_opr_algo_workspace_limit_inplace(vars, workspace_limit);
}

//! get the input tensor name in the order of graph
std::vector<const char*> NetworkImplDft::get_all_output_name() const {
    std::vector<const char*> output_names;
    for (auto& output : m_network_io->outputs) {
        output_names.push_back(output.name.c_str());
    }
    return output_names;
}

//! get the input tensor name in the order of graph
std::vector<const char*> NetworkImplDft::get_all_input_name() const {
    std::vector<const char*> input_names;
    for (auto& input : m_load_result.tensor_map) {
        input_names.push_back(input.first.c_str());
    }
    return input_names;
}

//! get the output tensor name in the order of graph
const char* NetworkImplDft::get_output_name(size_t index) const {
    LITE_ASSERT(
            index < m_load_result.output_var_list.size(),
            "The output tensor index is large than the total outputs number.");
    return m_load_result.output_var_list[index].node()->name().c_str();
}

//! get the input tensor name in the order of graph
const char* NetworkImplDft::get_input_name(size_t index) const {
    LITE_ASSERT(
            index < m_load_result.tensor_map.size(),
            "The input tensor index is large than the total inputs number.");
    size_t i = 0;
    for (auto& input : m_load_result.tensor_map) {
        if (i == index) {
            return input.first.c_str();
        }
        i++;
    }
    LITE_THROW(ssprintf("no input tensor of index %zu.", index));
}

//! Plugin part
void NetworkImplDft::enable_profile_performance(std::string profile_json_file) {
#if MGB_ENABLE_JSON
#if MGB_OPENCL
    mgb::CompNode::enable_opencl_profile(true);
#endif
    m_profiler = std::make_unique<mgb::GraphProfiler>(m_load_config.comp_graph.get());
    m_profiler_output_file = profile_json_file;
#else
    LITE_MARK_USED_VAR(profile_json_file);
    LITE_THROW("JSON is disable at compile time.");
#endif
}

void NetworkImplDft::enable_io_txt_dump(std::string io_txt_out_file) {
    auto iodump = std::make_unique<mgb::TextOprIODump>(
            m_load_config.comp_graph.get(), io_txt_out_file.c_str());
    iodump->print_addr(false);
    m_iodump = std::move(iodump);
}

void NetworkImplDft::enable_io_bin_dump(std::string io_bin_out_dir) {
    m_iodump = std::make_unique<mgb::BinaryOprIODump>(
            m_load_config.comp_graph.get(), io_bin_out_dir.c_str());
}

void inline NetworkImplDft::output_plugin_result() const {
#if MGB_ENABLE_JSON
    if (m_profiler && m_execute_func) {
        m_profiler->to_json_full(m_execute_func.get())
                ->writeto_fpath(m_profiler_output_file);
    }
#endif
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
