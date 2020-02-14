/**
 * \file src/serialization/include/megbrain/serialization/load_dump_config.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief graph loader and dumper config
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */
#pragma once

#include "megbrain/serialization/file.h"
#include "megbrain/serialization/opr_registry.h"

namespace mgb {
namespace serialization {
//! config for dumping a whole graph; setup in GraphDumper
struct GraphDumpConfig {
    /*!
     * \brief write tensor value (excluding metainfo like layout or dtype)
     *      to output file
     * \param opr the operator that requests to dump this tensor
     * \param tensor tensor to be dumped; layout guaranteed to be contiguous
     */
    using TensorValueDumper = thin_function<void(
            OutputFile& fout, const cg::OperatorNodeBase& opr,
            const HostTensorND& tensor)>;

    //! a fallback to implement custom tensor value dumper; it just writes
    //! the raw tensor value to output file. Implemented in serializer.cpp
    static void default_tensor_value_dumper(OutputFile& fout,
                                            const cg::OperatorNodeBase& opr,
                                            const HostTensorND& tensor);

    //! specify the vars whose names should be kept: 0 for none; 1 for
    //! output vars; 2 for all vars (internal + output vars)
    int keep_var_name;

    //! whether to keep param names
    bool keep_param_name;

    //! whether to keep operator priorities
    bool keep_opr_priority;

    //! extra user data to be passed by dump caller into opr dump
    //! implementations; useful for implementing nested opr dump
    std::shared_ptr<UserDataContainer> user_data;

    //! intercept how a single tensor is dumped; it should only dump the
    //! tensor value without layout; useful for compression or encryption
    TensorValueDumper tensor_value_dumper;

    GraphDumpConfig(int keep_var_name_ = 1, bool keep_param_name_ = false,
                    bool keep_opr_priority_ = false,
                    const std::shared_ptr<UserDataContainer>& user_data_ =
                            std::make_shared<UserDataContainer>(),
                    const TensorValueDumper& tensor_value_dumper_ = {})
            : keep_var_name{keep_var_name_},
              keep_param_name{keep_param_name_},
              keep_opr_priority{keep_opr_priority_},
              user_data{user_data_},
              tensor_value_dumper{tensor_value_dumper_} {}
};

//! config for loading a whole graph; setup in GraphLoader
struct GraphLoadConfig {
    using CompNodeMapper = thin_function<void(CompNode::Locator&)>;

    /*!
     * \brief load tensor value into given memory address
     * \param ptr dest pointer or nullptr; if it is NULL, fin should be
     *      advanced (by calling InputFile::skip()) to skip storage of this
     *      tensor
     * \param layout tensor layout, guaranteed to be contiguous
     */
    using TensorValueLoader = thin_function<void(
            void* ptr, const TensorLayout& layout, InputFile& fin)>;

    /*!
     * \brief callback to modify loaded tensors
     * \param name tensor name; it is empty for unnamed tensors
     * \param has_value whether tensor value is dumped (params usually have
     *      value)
     * \param tensor the tensor that can be modified inplace
     */
    using TensorModifier = thin_function<void(
            const std::string& name, bool has_value, HostTensorND& tensor)>;

    using OprLoaderMaker = thin_function<OprLoader(const std::string&)>;

    //! a fallback to implement custom tensor value reader; it just reads
    //! the raw tensor value from input file. Implemented in serializer.cpp
    static void default_tensor_value_loader(void* ptr,
                                            const TensorLayout& layout,
                                            InputFile& fin);

    //! whether to make all SharedDeviceTensor and Host2DeviceCopy shapes
    //! immutable so static inference can be eagerly performed; this can be
    //! used to reduce memory usage; tensor_modifier can be used to modify
    //! the shape
    bool const_var_shape = false;

    //! callback to modify loaded tensors before they are inserted into the
    //! graph
    TensorModifier tensor_modifier;

    //! callback to modify comp node locator inplace
    CompNodeMapper comp_node_mapper;

    //! map from any identifier to an opr loader; see
    //! OprRegistry::add_using_dynamic_loader
    OprLoaderMaker opr_loader_maker;

    //! extra user data to be passed by load caller into opr load
    //! implementations; useful for implementing nested opr load
    std::shared_ptr<UserDataContainer> user_data;

    //! computing graph to add new oprs; a new graph would be created if it
    //! is null
    std::shared_ptr<ComputingGraph> comp_graph;

    //! tensor value loader that must match tensor_value_dumper used in
    //! GraphDumpConfig
    TensorValueLoader tensor_value_loader;

    GraphLoadConfig(const CompNodeMapper& comp_node_mapper_ = {},
                    const OprLoaderMaker& opr_loader_maker_ = {},
                    const std::shared_ptr<UserDataContainer>& user_data_ = {},
                    const std::shared_ptr<ComputingGraph>& comp_graph_ = {},
                    const TensorValueLoader tensor_value_loader_ = {})
            : comp_node_mapper{comp_node_mapper_ ? comp_node_mapper_
                                                 : [](CompNode::Locator&) {}},
              opr_loader_maker{opr_loader_maker_},
              user_data{user_data_},
              comp_graph{comp_graph_},
              tensor_value_loader{tensor_value_loader_} {}
};
}  // namespace serialization
}  // namespace mgb
