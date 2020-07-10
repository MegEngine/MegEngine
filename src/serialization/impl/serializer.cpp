/**
 * \file src/serialization/impl/serializer.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/serialization/serializer.h"
#include "megbrain/opr/utility.h"

namespace mgb {
namespace serialization {

/* ====================== helper impls ====================== */
GraphLoader::LoadResult::~LoadResult() noexcept = default;

std::unique_ptr<cg::AsyncExecutable> GraphLoader::LoadResult::graph_compile(
        const ComputingGraph::OutputSpec& outspec) {
    auto ret = graph->compile(outspec);
    if (graph->options().comp_node_seq_record_level == 2) {
        ComputingGraph::assert_destroy(graph);
    }
    return ret;
}

GraphLoader::SharedTensorNameMap
GraphLoader::shared_tensor_name_map() {
    SharedTensorNameMap ret;
    for (auto &&i: shared_tensor_id_map()) {
        mgb_assert(!i.first.empty(), "name stripped during graph dump");
        auto ins = ret.emplace(i.first, &i.second);
        mgb_assert(ins.second);
    }
    return ret;
}

std::unique_ptr<GraphLoader> make_fbs_loader(std::unique_ptr<InputFile> file);
std::unique_ptr<GraphDumper> make_fbs_dumper(std::unique_ptr<OutputFile> file);
bool is_fbs_file(InputFile& file);

bool GraphDumper::should_remove_in_dump(cg::OperatorNodeBase *opr) {
#if MGB_ENABLE_GRAD
    return opr->same_type<opr::SetGrad>();
#else
    return false;
#endif
}

std::unique_ptr<GraphDumper> GraphDumper::make(std::unique_ptr<OutputFile> file,
                                               GraphDumpFormat format) {
    switch (format) {
        case GraphDumpFormat::FLATBUFFERS:
#if MGB_ENABLE_FBS_SERIALIZATION
            return make_fbs_dumper(std::move(file));
#endif
        MGB_FALLTHRU
        default:
            mgb_throw(SerializationError,
                      "unsupported serialization format requested");
    }
    mgb_assert(false, "unreachable");
}

std::unique_ptr<GraphLoader> GraphLoader::make(std::unique_ptr<InputFile> file, GraphDumpFormat format) {
    switch (format) {
        case GraphDumpFormat::FLATBUFFERS:
#if MGB_ENABLE_FBS_SERIALIZATION
            return make_fbs_loader(std::move(file));
#endif
        MGB_FALLTHRU
        default:
            mgb_throw(SerializationError,
                      "unsupported serialization format requested");
    }
    mgb_assert(false, "unreachable");
}

Maybe<GraphDumpFormat> GraphLoader::identify_graph_dump_format(
        InputFile& file) {
#if MGB_ENABLE_FBS_SERIALIZATION
    if (is_fbs_file(file)) {
        return GraphDumpFormat::FLATBUFFERS;
    }
#endif
    return {};
}

}  // namespace serialization
}  // namespace mgb
