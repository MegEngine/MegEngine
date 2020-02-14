/**
 * \file src/serialization/include/megbrain/serialization/serializer.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/serialization/dump_format.h"
#include "megbrain/serialization/file.h"
#include "megbrain/serialization/load_dump_config.h"

namespace mgb {
namespace serialization {
    /*!
     * \brief load graph from megbrain dump file
     *
     * Each GraphLoader instance can create multiple graphs, but all the created
     * graphs share underlying params (i.e. values for SharedDeviceTensor are
     * shared)
     */
    class GraphLoader {
        public:
            using LoadConfig = GraphLoadConfig;
            struct LoadResult {
                //! expliit dtor decl to reduce binary size
                ~LoadResult() noexcept;

                using TensorMap = std::unordered_map<
                    std::string, std::shared_ptr<HostTensorND>>;

                std::shared_ptr<ComputingGraph> graph;

                //! name to host tensor used in this graph, usually for input
                //! tensors
                TensorMap tensor_map;

                //! name to output var nodes specified during serializing
                std::unordered_map<std::string, SymbolVar> output_var_map;

                //! map from original id to loaded output vars
                std::unordered_map<size_t, SymbolVar> output_var_map_id;

                //! original output vars in the order passed to
                //! GraphDumper::dump
                SymbolVarArray output_var_list;

                /*!
                 * \brief call graph->compile() but also checks for comp seq rec
                 *
                 * graph would be destructed if comp_node_seq_record_level == 2;
                 * this method should be called in favor of graph->compile().
                 */
                std::unique_ptr<cg::AsyncExecutable> graph_compile(
                        const ComputingGraph::OutputSpec &outspec);
            };

            //! mem_node => tensor_value
            using SharedTensorMapEntry =
                    ThinHashMap<MemNode, std::shared_ptr<DeviceTensorND>>;

            /*!
             * tensor_id => (tensor_name, (mem_node => tensor_value))
             *
             * Since tensor IDs are guaranteed to be consecutive, a vector is
             * used to implement the map.
             *
             * Either all tensor names are empty, or they are guaranteed to be
             * distinct non-empty strings at dump time.
             */
            using SharedTensorIDMap =
                std::vector<std::pair<std::string, SharedTensorMapEntry>>;

            //! tensor_name => SharedTensorMapEntry
            using SharedTensorNameMap = std::unordered_map<
                std::string, const SharedTensorMapEntry*>;

            static std::unique_ptr<GraphLoader> make(
                    std::unique_ptr<InputFile> file,
                    GraphDumpFormat format = {});

            static Maybe<GraphDumpFormat> identify_graph_dump_format(
                    InputFile& file);

            virtual ~GraphLoader() = default;

            /*!
             * \brief reset underlying input file from which further load()
             *      would read
             *
             * This method can be used to release the currently owned file to
             * the caller.
             *
             * \param file new input file, can be null
             * \return original input file that is currently used
             */
            virtual std::unique_ptr<InputFile> reset_file(
                    std::unique_ptr<InputFile> file = {}) = 0;

            /*!
             * \brief create a new graph instance; not thread safe
             * \param rewind whether to call InputFile::rewind before loading
             */
            virtual LoadResult load(const LoadConfig &config = {},
                    bool rewind = true) = 0;

            /*!
             * \brief get mapping from tensor ID to device tensor shared
             *      between instances
             *
             * The shared tensors are usually used as model params in a machine
             * learning context. For each param name, the returned value has a
             * map from a memory node to the first param loaded on that mem node
             */
            virtual const SharedTensorIDMap& shared_tensor_id_map() const = 0;

            //! helper for constructing SharedTensorNameMap from
            //! SharedTensorIDMap
            SharedTensorNameMap shared_tensor_name_map();

            virtual GraphDumpFormat format() const = 0;
    };

    /*!
     * \brief dump graph into given output file
     */
    class GraphDumper {
        public:
            using DumpConfig = GraphDumpConfig;
            struct DumpResult {
                //! number of oprs written
                size_t nr_opr = 0;

                //! hash of the graph
                uint64_t content_hash;

                //! full dump size and param value size
                size_t tot_bytes = 0, tensor_value_bytes = 0;

                std::vector<std::string>
                    inputs,     //!< input tensor names
                    outputs,    //!< output var names
                    params;     //!< dumped param names
            };

            static std::unique_ptr<GraphDumper> make(
                    std::unique_ptr<OutputFile> file,
                    GraphDumpFormat format = {});

            virtual ~GraphDumper() = default;

            /*!
             * \brief whether an operator should be removed in graph
             *      serialization file
             */
            static bool should_remove_in_dump(cg::OperatorNodeBase *opr);

            virtual DumpResult dump(
                    const SymbolVarArray &output_vars,
                    const DumpConfig &config = {}) = 0;
            
            virtual GraphDumpFormat format() const = 0;
    };

} // namespace serialization
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

