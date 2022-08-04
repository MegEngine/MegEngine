#include "megbrain/serialization/serializer.h"
#include "megbrain/gopt/inference.h"
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

void GraphLoader::LoadResult::update_output_var_list(
        const SymbolVarArray& output_var_array) {
    mgb::ThinHashMap<mgb::SymbolVar, mgb::SymbolVar> out_var_map;
    mgb_assert(output_var_array.size() == output_var_list.size());
    // replace symvar in output_var_list
    for (size_t idx = 0; idx < output_var_array.size(); ++idx) {
        out_var_map[output_var_list[idx]] = output_var_array[idx];
        output_var_list[idx] = output_var_array[idx];
    }
    // replace symvar in output_var_map_id
    for (auto&& item : output_var_map_id) {
        item.second = out_var_map[item.second];
    }
    // replace symvar in output_var_map
    for (auto&& item : output_var_map) {
        item.second = out_var_map[item.second].rename(item.first);
    }
}

void GraphLoader::LoadResult::graph_compile_ahead() {
    //! when force_output_use_user_specified_memory is set, the output var may
    //! be changed by gopt, then the var in LoadResult can not exist, so here
    //! just do basic optimize_for_inference ahead, and replace the var in
    //! LoadResult
    if (graph->options().force_output_use_user_specified_memory) {
        auto options = gopt::OptimizeForInferenceOptions{};
        auto new_vars = gopt::optimize_for_inference(output_var_list, options);
        output_var_list = new_vars;
        output_var_map.clear();
        for (auto& var : new_vars) {
            output_var_map[var.node()->cname()] = var;
        }
        std::unordered_map<size_t, SymbolVar> var_map_id;
        for (auto& var : new_vars) {
            bool found = false;
            for (auto& old_var_it : output_var_map_id) {
                if (old_var_it.second.node()->name() == var.node()->name()) {
                    found = true;
                    var_map_id[old_var_it.first] = var;
                }
            }
            mgb_assert(
                    found, "can't find var name %s when optimize_for_inference. ",
                    var.node()->cname());
        }
    }
}

GraphLoader::SharedTensorNameMap GraphLoader::shared_tensor_name_map() {
    SharedTensorNameMap ret;
    for (auto&& i : shared_tensor_id_map()) {
        mgb_assert(!i.first.empty(), "name stripped during graph dump");
        auto ins = ret.emplace(i.first, &i.second);
        mgb_assert(ins.second);
    }
    return ret;
}
std::unique_ptr<GraphLoader> make_fbs_loader(std::unique_ptr<InputFile> file);
std::unique_ptr<GraphDumper> make_fbs_dumper(std::unique_ptr<OutputFile> file);

std::unique_ptr<GraphLoader> make_fbs_v2_loader(std::unique_ptr<InputFile> file);
std::unique_ptr<GraphDumper> make_fbs_v2_dumper(
        std::unique_ptr<OutputFile> file, int version);
bool is_fbs_file(InputFile& file);
bool is_fbs_v2_file(InputFile& file);

bool GraphDumper::should_remove_in_dump(cg::OperatorNodeBase* opr) {
#if MGB_ENABLE_GRAD
    return opr->same_type<opr::SetGrad>();
#else
    return false;
#endif
}

std::unique_ptr<GraphDumper> GraphDumper::make(
        std::unique_ptr<OutputFile> file, GraphDumpFormat format, int version) {
    switch (format) {
        case GraphDumpFormat::FLATBUFFERS:
#if MGB_ENABLE_FBS_SERIALIZATION
            return make_fbs_dumper(std::move(file));
#endif
            MGB_FALLTHRU
        case GraphDumpFormat::FLATBUFFERS_V2:
#if MGB_ENABLE_FBS_SERIALIZATION
            return make_fbs_v2_dumper(std::move(file), version);
#endif
            MGB_FALLTHRU
        default:
            mgb_throw(SerializationError, "unsupported serialization format requested");
    }
    mgb_assert(false, "unreachable");
}

std::unique_ptr<GraphLoader> GraphLoader::make(
        std::unique_ptr<InputFile> file, GraphDumpFormat format) {
    switch (format) {
        case GraphDumpFormat::FLATBUFFERS:
#if MGB_ENABLE_FBS_SERIALIZATION
            return make_fbs_loader(std::move(file));
#endif
            MGB_FALLTHRU
        case GraphDumpFormat::FLATBUFFERS_V2:
#if MGB_ENABLE_FBS_SERIALIZATION
            return make_fbs_v2_loader(std::move(file));
#endif
            MGB_FALLTHRU
        default:
            mgb_throw(SerializationError, "unsupported serialization format requested");
    }
    mgb_assert(false, "unreachable");
}

Maybe<GraphDumpFormat> GraphLoader::identify_graph_dump_format(InputFile& file) {
#if MGB_ENABLE_FBS_SERIALIZATION
    if (is_fbs_file(file)) {
        return GraphDumpFormat::FLATBUFFERS;
    }
    if (is_fbs_v2_file(file)) {
        return GraphDumpFormat::FLATBUFFERS_V2;
    }
#endif
    return {};
}

}  // namespace serialization
}  // namespace mgb
