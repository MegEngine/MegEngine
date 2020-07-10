/**
 * \file src/serialization/impl/serializer_oss.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

/*
 * Dump file layout:
 * [uint32_t fourcc]
 * [00 00 00 00]
 * [uint64_t offset to graph from tensor start]
 * [Tensor 1]
 * [Tensor 2]
 * [...]
 * [Tensor N]
 * [SizePrefixed FlatBuffers Graph]
 */
#if MGB_ENABLE_FBS_SERIALIZATION

#include "batched_device_value_loader.h"

#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/helper.h"
#include "megbrain/serialization/internal/flatbuffers_helper.h"
#include "megbrain/serialization/internal/schema_generated.h"
#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/version.h"

#include <flatbuffers/flatbuffers.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>

using namespace mgb;
using namespace mgb::serialization;

namespace {

constexpr uint32_t MGB_VERSION =
        (MGB_MAJOR * 1000 + MGB_MINOR) * 100 + MGB_PATCH;

constexpr uint32_t MGB_MAGIC = 0x5342474D;

template <typename T>
bool contains_any_in_set(const SmallVector<T>& list,
                         const ThinHashSet<T>& set) {
    for (const auto& x : list) {
        if (set.count(x)) {
            return true;
        }
    }
    return false;
}

void check_tensor_value_valid(const std::string& name,
                              const HostTensorND& tensor) {
    mgb_assert(tensor.layout().is_physical_contiguous(),
               "non-contiguous tensor: name=%s layout=%s", name.c_str(),
               tensor.layout().to_string().c_str());
    if (tensor.dtype() == dtype::Float32()) {
        auto ptr = tensor.ptr<float>();
        for (size_t i = 0, it = tensor.shape().total_nr_elems(); i < it; ++i) {
            if (!std::isfinite(ptr[i])) {
                mgb_log_warn("invalid tensor value in %s: %g", name.c_str(),
                             ptr[i]);
                break;
            }
        }
    }
}

}  // namespace

namespace mgb {
namespace serialization {

class GraphDumperOSS final : public GraphDumper, OprDumpContextFlatBuffers {
    const std::unique_ptr<OutputFile> m_file;
    flatbuffers::FlatBufferBuilder m_builder;

    DumpConfig m_config;
    DumpResult m_cur_rst;

    size_t m_nr_shared_tensor;

    std::vector<std::pair<cg::OperatorNodeBase*, const OprRegistry*>>
            m_oprs_to_dump;
    ThinHashMap<VarNode*, size_t> m_var2id;

    //! set of output vars specified by user
    ThinHashSet<VarNode*> m_output_vars;

    std::unordered_set<std::string> m_used_input_names, m_used_param_names;

    //! current opr to be dumped
    cg::OperatorNodeBase* m_cur_opr = nullptr;

    // Will be filled in dump_tensor
    std::vector<flatbuffers::Offset<fbs::Tensor>> m_cur_opr_tensor;
    std::vector<flatbuffers::Offset<fbs::Blob>> m_blobs;
    std::vector<fbs::OperatorParam> m_cur_opr_param_type;
    std::vector<flatbuffers::Offset<void>> m_cur_opr_param;

    void init_oprs_to_dump(const SymbolVarArray& endpoints);
    flatbuffers::Offset<fbs::Operator> build_single_opr(
            cg::OperatorNodeBase* opr, const OprRegistry* registry);

    flatbuffers::Offset<fbs::DType> build_dtype(DType dtype);

public:
    GraphDumperOSS(std::unique_ptr<OutputFile> file) : m_file{std::move(file)} {}
    DumpResult dump(const SymbolVarArray& output_vars,
                    const DumpConfig& config = {}) override;
    const GraphDumpConfig& config() const override { return m_config; }
    void dump_tensor(const std::string& name, const HostTensorND& tensor,
                     TensorWriteMethod method) override;
    flatbuffers::FlatBufferBuilder& builder() override { return m_builder; }
    void append_param(uint32_t type, uint32_t value) override {
        static_assert(std::is_same<uint32_t, flatbuffers::uoffset_t>::value,
                      "append_param depends on uoffset_t being uint32_t");
        static_assert(std::is_standard_layout<flatbuffers::Offset<void>>::value,
                      "append_param depends on flatbuffers::Offset having "
                      "standard memory layout");
        mgb_assert(type != fbs::OperatorParam_NONE);
        m_cur_opr_param_type.emplace_back(
                static_cast<fbs::OperatorParam>(type));
        m_cur_opr_param.emplace_back(value);
    }
    void dump_buf_with_len(const void* data, uint32_t size) override;
    GraphDumpFormat format() const override {
        return GraphDumpFormat::FLATBUFFERS;
    }
};

flatbuffers::Offset<fbs::DType> GraphDumperOSS::build_dtype(DType dtype) {
    return fbs::intl::build_dtype(m_builder, dtype);
}

void GraphDumperOSS::init_oprs_to_dump(const SymbolVarArray& endpoints) {
    m_oprs_to_dump.clear();
    m_var2id.clear();

    // iterate oprs to init m_var2id
    size_t next_id = 0;
    auto on_opr = [&](cg::OperatorNodeBase* opr) {
        if (should_remove_in_dump(opr)) {
            mgb_assert(opr->input().size() == 1);
            // Copy input ID to output
            auto id = m_var2id.at(opr->input(0));
            for (auto i : opr->output())
                m_var2id[i] = id;
        } else {
            auto registry = OprRegistry::find_by_type(opr->dyn_typeinfo());
            if (!registry || !registry->dumper) {
                mgb_throw(cg::OperatorNodeExcExtraInfo::ExcMaker{opr}
                                  .make<MegBrainError>,
                          "serialization as FlatBuffers is not supported for "
                          "operator %s",
                          opr->dyn_typeinfo()->name);
            }
            m_oprs_to_dump.emplace_back(opr, registry);
            for (auto i : opr->output()) {
                if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    m_var2id[i] = next_id++;
                }
            }
        }
    };
    cg::DepOprIter dep_opr_iter{on_opr};
    for (auto i : endpoints) {
        dep_opr_iter.add(i.node()->owner_opr());
    }
}

flatbuffers::Offset<fbs::Operator> GraphDumperOSS::build_single_opr(
        cg::OperatorNodeBase* opr, const OprRegistry* registry) {
    m_cur_opr = opr;
    ++m_cur_rst.nr_opr;

    using namespace flatbuffers;
    Offset<Vector<Offset<fbs::CompNode>>> comp_node;
    auto& config = opr->config();
    if (config.has_comp_node_set()) {
        std::vector<flatbuffers::Offset<fbs::CompNode>> cns;
        for (const auto& cn : config.comp_node()) {
            cns.emplace_back(fbs::CreateCompNode(
                    m_builder,
                    m_builder.CreateSharedString(cn.to_string_logical())));
        }
        comp_node = m_builder.CreateVector(cns);
    }

    Offset<Vector<uint32_t>> inputs;
    if (opr->input().size()) {
        std::vector<uint32_t> v;
        v.reserve(opr->input().size());
        for (auto inp : opr->input()) {
            v.emplace_back(m_var2id.at(inp));
        }
        inputs = m_builder.CreateVector(v);
    }

    Offset<Vector<Offset<String>>> output_names;
    if (m_config.keep_var_name >= 2 ||
        (m_config.keep_var_name == 1 &&
         contains_any_in_set(opr->output(), m_output_vars))) {
        std::vector<std::string> onames;
        for (auto i : opr->output()) {
            if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                onames.emplace_back(i->name());
            }
        }
        output_names = m_builder.CreateVectorOfStrings(onames);
    }

    auto output_dtype = build_dtype(config.output_dtype());

    m_cur_opr_tensor.clear();
    m_blobs.clear();
    m_cur_opr_param.clear();
    m_cur_opr_param_type.clear();
    registry->dumper(*this, *opr);

    Offset<Vector<Offset<fbs::Tensor>>> tensors;
    if (m_cur_opr_tensor.size())
        tensors = m_builder.CreateVector(m_cur_opr_tensor);

    Offset<Vector<Offset<fbs::Blob>>> blobs;
    if (m_blobs.size())
        blobs = m_builder.CreateVector(m_blobs);

    Offset<Vector<uint8_t>> additional_params_type;
    Offset<Vector<Offset<void>>> additional_params;
    auto param_cnt = m_cur_opr_param_type.size();
    if (param_cnt > 1) {
        additional_params_type = m_builder.CreateVectorScalarCast<uint8_t>(
                m_cur_opr_param_type.data() + 1, param_cnt - 1);
        additional_params = m_builder.CreateVector(m_cur_opr_param.data() + 1,
                                                   param_cnt - 1);
    }

    fbs::OperatorBuilder builder(m_builder);
    builder.add_type_id(registry->unversioned_type_id);
    builder.add_inputs(inputs);
    if (m_config.keep_opr_priority) {
        builder.add_priority(opr->node_prop().attribute().priority);
    }
    builder.add_comp_node(comp_node);
    builder.add_output_name(output_names);
    builder.add_output_dtype(output_dtype);
    if (param_cnt > 0) {
        builder.add_param_type(m_cur_opr_param_type[0]);
        builder.add_param(m_cur_opr_param[0]);
    }
    if (param_cnt > 1) {
        builder.add_additional_params_type(additional_params_type);
        builder.add_additional_params(additional_params);
    }
    builder.add_tensors(tensors);
    builder.add_blobs(blobs);
    m_cur_opr = nullptr;
    return builder.Finish();
}

GraphDumper::DumpResult GraphDumperOSS::dump(
        const SymbolVarArray& output_vars, const DumpConfig& config) {
    mgb_throw_if(output_vars.empty(), SerializationError,
                 "Can't dump empty graph");

    auto begin_pos = m_file->tell();
    m_config = config;
    m_builder.Reset();

    m_output_vars.clear();
    m_cur_rst = {};
    m_used_input_names.clear();
    m_used_param_names.clear();
    m_nr_shared_tensor = 0;

    // process output vars
    bool keep_output_var_name = m_config.keep_var_name >= 1;
    std::unordered_set<std::string> output_var_names;
    for (auto i : output_vars) {
        mgb_assert(!i.node()->contain_flag(VarNode::Flag::VOLATILE_CONTENT),
                   "can not dump var with VOLATILE_CONTENT flag: %s",
                   cg::dump_var_info({i.node()}).c_str());
        if (m_output_vars.insert(i.node()).second && keep_output_var_name) {
            auto name_ins = output_var_names.insert(i.node()->name()).second;
            mgb_assert(name_ins, "duplicated output var name: %s",
                       i.node()->cname());
        }
    }

    // Write magic
    uint32_t magic = MGB_MAGIC;
    m_file->write(&magic, sizeof(magic));

    // Padding
    uint32_t reserved = 0;
    m_file->write(&reserved, sizeof(reserved));

    // Write placeholder for offset_to_fbs
    auto offset_pos = m_file->tell();
    uint64_t offset_to_fbs = 0;
    m_file->write(&offset_to_fbs, sizeof(offset_to_fbs));

    // Dump operators
    init_oprs_to_dump(output_vars);
    std::vector<flatbuffers::Offset<fbs::Operator>> oprs;
    for (auto&& i : m_oprs_to_dump) {
        oprs.emplace_back(build_single_opr(i.first, i.second));
    }
    auto fb_oprs = m_builder.CreateVector(oprs);

    // Dump output vars
    std::vector<fbs::OutputVar> output_vars_idx;
    output_vars_idx.reserve(output_vars.size());
    for (auto i : output_vars) {
        output_vars_idx.emplace_back(m_var2id.at(i.node()), i.node()->id());
    }
    auto fb_output_vars = m_builder.CreateVectorOfStructs(output_vars_idx);

    XXHash content_hash;
    content_hash.update(m_builder.GetCurrentBufferPointer(),
                        m_builder.GetSize());
    auto graph_hash = content_hash.digest();

    fbs::GraphBuilder graph(m_builder);
    graph.add_mgb_version(MGB_VERSION);
    graph.add_hash(graph_hash);
    graph.add_oprs(fb_oprs);
    graph.add_output_vars_idx(fb_output_vars);
    graph.add_nr_shared_tensor(m_nr_shared_tensor);
    m_builder.FinishSizePrefixed(graph.Finish(), fbs::GraphIdentifier());

    // Write actual offset_to_fbs
    auto cur = m_file->tell();
    mgb_assert(cur >= offset_pos && cur - offset_pos >= sizeof(offset_to_fbs));
    offset_to_fbs = cur - offset_pos - sizeof(offset_to_fbs);
    m_file->seek(offset_pos);
    m_file->write(&offset_to_fbs, sizeof(offset_to_fbs));
    m_file->seek(cur);

    // Write serialized fbs::Graph
    m_file->write(m_builder.GetBufferPointer(), m_builder.GetSize());

    // Finalize DumpResult
    auto&& ret = m_cur_rst;
    for (size_t i = 0; i < output_vars.size(); i++) {
        ret.outputs.emplace_back(keep_output_var_name
                                         ? output_vars[i].node()->cname()
                                         : ssprintf("unnamed%zu", i));
    }
    ret.content_hash = graph_hash;
    std::sort(ret.inputs.begin(), ret.inputs.end());
    mgb_assert(ret.nr_opr == m_oprs_to_dump.size());
    ret.tot_bytes = m_file->tell() - begin_pos;
    return ret;
}

void GraphDumperOSS::dump_tensor(const std::string& name,
                                       const HostTensorND& tensor,
                                       TensorWriteMethod method) {
    using namespace flatbuffers;
    using Meth = TensorWriteMethod;
    mgb_assert((method == Meth::VALUE_ANONYMOUS) ^ (!name.empty()),
               "name must be non-empty for non Meth::VALUE_ANONYMOUS tensors");

    bool has_value = method != Meth::META_INPUT;
    bool should_keep_name = true;
    switch (method) {
        case Meth::VALUE_ANONYMOUS:
            should_keep_name = false;
            break;
        case Meth::VALUE_SHARED:
            should_keep_name = m_config.keep_param_name;
            ++m_nr_shared_tensor;
            if (m_config.keep_param_name) {
                mgb_assert(m_used_param_names.insert(name).second,
                           "duplicated VALUE_SHARED tensor name: %s",
                           name.c_str());
                m_cur_rst.params.emplace_back(name);
            }
            break;
        case Meth::META_INPUT:
        case Meth::VALUE_INPUT:
            mgb_assert(!name.empty(), "empty input tensor name");
            mgb_assert(m_used_input_names.insert(name).second,
                       "duplicated input tensor name: %s", name.c_str());
            m_cur_rst.inputs.emplace_back(name);
            break;
    }

    size_t value_size = 0;
    if (has_value) {
        check_tensor_value_valid(name, tensor);
        auto begin = m_file->tell();
        auto&& dumper = m_config.tensor_value_dumper;
        if (dumper) {
            dumper(*m_file, *m_cur_opr, tensor);
        } else {
            m_file->write(tensor.raw_ptr(), tensor.layout().span().high_byte);
        }
        value_size = m_file->tell() - begin;
        m_cur_rst.tensor_value_bytes += value_size;
    }

    auto fbname = should_keep_name ? m_builder.CreateSharedString(name) : 0;
    auto shape = m_builder.CreateVectorScalarCast<uint32_t>(
            tensor.shape().shape, tensor.shape().ndim);
    auto comp_node = fbs::CreateCompNode(
            m_builder, m_builder.CreateSharedString(
                               tensor.comp_node().to_string_logical()));
    auto dtype = build_dtype(tensor.dtype());
    auto serialized_tensor = fbs::CreateTensor(m_builder, fbname, shape,
                                               comp_node, dtype, value_size);
    m_cur_opr_tensor.emplace_back(serialized_tensor);
}

void GraphDumperOSS::dump_buf_with_len(const void* data, uint32_t size) {
    auto blob = fbs::CreateBlob(
            m_builder,
            m_builder.CreateVector(static_cast<const uint8_t*>(data), size));
    m_blobs.emplace_back(blob);
}

// ----------------------------- Loader --------------------------------------

class GraphLoaderOSS final : public GraphLoader {
    const LoadConfig* m_cur_load_config = nullptr;
    std::unique_ptr<InputFile> m_file;
    SharedBuffer m_graph_buf{{}, 0};
    const fbs::Graph* m_graph;
    SharedTensorIDMap m_shared_tensor_map;
    uint32_t m_mgb_version = 0;
    uint64_t m_graph_hash = 0;

    class OprLoadContextImpl;
    friend class OprLoadContextImpl;

    void verify();

public:
    GraphLoaderOSS(std::unique_ptr<InputFile> input_file)
            : m_file{std::move(input_file)} {}

    std::unique_ptr<InputFile> reset_file(
            std::unique_ptr<InputFile> file) override {
        file.swap(m_file);
        return file;
    }

    LoadResult load(const LoadConfig& config, bool rewind) override;

    const SharedTensorIDMap& shared_tensor_id_map() const override {
        mgb_assert(m_graph_hash, "graph not loaded yet");
        return m_shared_tensor_map;
    }

    GraphDumpFormat format() const override {
        return GraphDumpFormat::FLATBUFFERS;
    }
};

class GraphLoaderOSS::OprLoadContextImpl final
        : public OprLoadContextFlatBuffers {
    GraphLoaderOSS* const m_loader;
    size_t m_cur_shared_tensor_idx = 0;
    std::shared_ptr<ComputingGraph> m_graph;
    LoadResult::TensorMap m_tensor_map;
    VarNodeArray m_id2varnode;
    BatchedDeviceValueLoader m_device_value_loader;
    const fbs::Operator* m_current_opr;
    size_t m_cur_opr_tensor_cnt;
    size_t m_cur_opr_blob_cnt;
    size_t m_cur_opr_param_cnt;

    ComputingGraph& graph() override { return *m_graph; }

    const GraphLoadConfig& config() const override {
        return *m_loader->m_cur_load_config;
    }

    void load_tensor_value(HostTensorND* dest, const TensorLayout& layout,
                           const fbs::Tensor* tensor);

    std::shared_ptr<HostTensorND> load_tensor() override;

    std::shared_ptr<DeviceTensorND> load_tensor_shared() override;

    void load_single_opr(const fbs::Operator* opr);

public:
    OprLoadContextImpl(GraphLoaderOSS* loader, uint32_t version)
            : OprLoadContextFlatBuffers(version), m_loader{loader} {
        m_graph = loader->m_cur_load_config->comp_graph;
        if (!m_graph) {
            m_graph = ComputingGraph::make();
        }
        auto maker = [this]() {
            return std::shared_ptr<OprLoadContext>{
                    std::shared_ptr<OprLoadContext>{}, this};
        };
        auto got = m_graph->options()
                           .user_data.get_user_data_or_create<OprLoadContext>(
                                   maker);
        mgb_assert(got == this);
    }

    ~OprLoadContextImpl() noexcept {
        auto nr = m_graph->options().user_data.pop_user_data<OprLoadContext>();
        mgb_assert(nr == 1);
    }

    LoadResult load_oprs();
    CompNode load_comp_node(const fbs::CompNode* comp_node);

    const void* get_next_param(uint32_t enumv) override {
        auto type = static_cast<fbs::OperatorParam>(enumv);
        if (m_cur_opr_param_cnt == 0) {
            m_cur_opr_param_cnt++;
            if (m_current_opr->param_type() == type) {
                return m_current_opr->param();
            }
        } else {
            mgb_assert(m_current_opr->additional_params() &&
                       m_cur_opr_param_cnt - 1 <
                               m_current_opr->additional_params()->size());
            auto i = m_cur_opr_param_cnt++ - 1;
            if (m_current_opr->additional_params_type()->Get(i) == type) {
                return m_current_opr->additional_params()->Get(i);
            }
        }
        return nullptr;
    }

    std::string load_buf_with_len() override {
        mgb_assert(m_current_opr->blobs() &&
                   m_cur_opr_blob_cnt < m_current_opr->blobs()->size());
        auto blob = m_current_opr->blobs()->Get(m_cur_opr_blob_cnt++);
        mgb_assert(blob && blob->data());
        auto data = blob->data()->data();
        return {reinterpret_cast<const char*>(data), blob->data()->size()};
    }
    SharedBuffer load_shared_buf_with_len() override {
        mgb_assert(m_current_opr->blobs() &&
                   m_cur_opr_blob_cnt < m_current_opr->blobs()->size());
        auto blob = m_current_opr->blobs()->Get(m_cur_opr_blob_cnt++);
        mgb_assert(blob && blob->data());
        auto size = blob->data()->size();
        std::shared_ptr<uint8_t> shptr{new uint8_t[size],
                                       [](uint8_t* p) { delete[] p; }};
        memcpy(shptr.get(), blob->data()->data(), size);
        return {std::move(shptr), size};
    }
};

CompNode GraphLoaderOSS::OprLoadContextImpl::load_comp_node(
        const fbs::CompNode* comp_node) {
    mgb_assert(comp_node);
    if (!comp_node->logical_locator())
        return {};
    auto loc = CompNode::Locator::parse(comp_node->logical_locator()->str());
    m_loader->m_cur_load_config->comp_node_mapper(loc);
    return CompNode::load(loc);
}

TensorLayout load_tensor_layout(const fbs::Tensor* tensor) {
    TensorLayout layout;
    if (tensor->shape()) {
        layout.ndim = tensor->shape()->size();
        std::copy(tensor->shape()->begin(), tensor->shape()->end(),
                  layout.shape);
        layout.init_contiguous_stride();
    }
    if (tensor->dtype()) {
        layout.dtype = fbs::intl::load_dtype(tensor->dtype());
    }
    return layout;
}

void GraphLoaderOSS::OprLoadContextImpl::load_tensor_value(
        HostTensorND* dest, const TensorLayout& layout,
        const fbs::Tensor* tensor) {
    auto&& loader = m_loader->m_cur_load_config->tensor_value_loader;
    auto&& file = m_loader->m_file;
    auto begin_pos = file->tell();
    file->skip(tensor->offset());
    if (loader) {
        // call custom loader
        void* dest_ptr = nullptr;
        if (dest) {
            dest->dtype(layout.dtype).resize(layout);
            dest_ptr = dest->raw_ptr();
        }
        loader(dest_ptr, layout, *file);
    } else {
        if (dest) {
            file->read_into_tensor(*dest, layout);
        } else {
            file->skip(layout.span().high_byte);
        }
    }
    mgb_throw_if(file->tell() < begin_pos, SerializationError,
                 "Custom tensor value loader accessed out of range data before "
                 "start of data blob");
    auto data_size = tensor->data_size();
    auto consumed_size = file->tell() - begin_pos;
    mgb_throw_if(consumed_size > data_size, SerializationError,
                 "Custom tensor value loader consumed more data than "
                 "available: consumed %lu, has %u",
                 consumed_size, data_size);
    if (consumed_size < data_size) {
        mgb_log_warn(
                "Tensor value loader consumed less data than available: "
                "consumed %lu bytes, has %u bytes",
                consumed_size, data_size);
        file->skip(data_size - consumed_size);
    }
}

std::shared_ptr<HostTensorND>
GraphLoaderOSS::OprLoadContextImpl::load_tensor() {
    mgb_assert(m_current_opr->tensors() &&
               m_cur_opr_tensor_cnt < m_current_opr->tensors()->size());
    auto tensor = m_current_opr->tensors()->Get(m_cur_opr_tensor_cnt++);
    auto comp_node = load_comp_node(tensor->comp_node());
    auto layout = load_tensor_layout(tensor);
    auto ret = std::make_shared<HostTensorND>(comp_node, layout);
    if (tensor->data_size()) {
        load_tensor_value(ret.get(), layout, tensor);
    }
    if (tensor->name()) {
        m_tensor_map[tensor->name()->str()] = ret;
    }
    if (auto&& mod = m_loader->m_cur_load_config->tensor_modifier) {
        mod(tensor->name() ? tensor->name()->str() : "",
            tensor->data_size() != 0, *ret);
    }
    return ret;
}

std::shared_ptr<DeviceTensorND>
GraphLoaderOSS::OprLoadContextImpl::load_tensor_shared() {
    mgb_assert(m_current_opr->tensors() &&
               m_cur_opr_tensor_cnt < m_current_opr->tensors()->size());
    auto tensor = m_current_opr->tensors()->Get(m_cur_opr_tensor_cnt++);
    auto comp_node = load_comp_node(tensor->comp_node());
    auto layout = load_tensor_layout(tensor);
    mgb_assert(tensor->data_size());
    auto&& sh_reg = m_loader->m_shared_tensor_map.at(m_cur_shared_tensor_idx++);
    auto&& sh_ptr_ref = sh_reg.second[comp_node.mem_node()];
    if (sh_ptr_ref) {
        // cached tensor value is valid so we can reuse it
        load_tensor_value(nullptr, layout, tensor);
        if (sh_ptr_ref->comp_node() == comp_node)
            return sh_ptr_ref;
        // same mem node but different comp node, change comp node and share
        // value
        auto ret = std::make_shared<DeviceTensorND>(*sh_ptr_ref);
        ret->comp_node(comp_node);
        return ret;
    }
    if (tensor->name()) {
        sh_reg.first = tensor->name()->str();
    }

    if (comp_node.mem_node() == CompNode::default_cpu().mem_node()) {
        // directly forward CPU memory
        HostTensorND hv{comp_node};
        load_tensor_value(&hv, layout, tensor);
        sh_ptr_ref = std::make_shared<DeviceTensorND>();
        *sh_ptr_ref = DeviceTensorND::make_proxy(hv);
    } else {
        // use lazy load for non-CPU devices
        HostTensorND hv{CompNode::default_cpu()};
        load_tensor_value(&hv, layout, tensor);
        sh_ptr_ref = m_device_value_loader.make(comp_node, std::move(hv));
    }
    return sh_ptr_ref;
}

void GraphLoaderOSS::OprLoadContextImpl::load_single_opr(
        const fbs::Operator* fbopr) {
    m_cur_opr_tensor_cnt = 0;
    m_cur_opr_blob_cnt = 0;
    m_cur_opr_param_cnt = 0;

    OperatorNodeConfig config;
    if (fbopr->output_dtype()) {
        config.output_dtype(fbs::intl::load_dtype(fbopr->output_dtype()));
    }
    if (fbopr->comp_node()) {
        auto cnt = fbopr->comp_node()->size();
        cg::OperatorNodeConfig::CompNodeArray comp_node_arr(cnt);
        for (size_t i = 0; i < cnt; i++) {
            CompNode cn{};
            auto node = fbopr->comp_node()->Get(i);
            if (node) {
                cn = load_comp_node(node);
            }
            comp_node_arr[i] = cn;
        }
        config.comp_node_arr(comp_node_arr);
    }

    auto registry = OprRegistry::find_by_unversioned_id(fbopr->type_id());
    mgb_throw_if(!registry, SerializationError,
                 "failed to find opr with type %s, use "
                 "mgb.config.dump_registered_oprs() "
                 "to get a dict that maps from opr id to opr name",
                 std::to_string(fbopr->type_id()).c_str());

    // load inputs
    VarNodeArray inputs;
    if (fbopr->inputs()) {
        inputs.resize(fbopr->inputs()->size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = m_id2varnode.at(fbopr->inputs()->Get(i));
        }
    }

    // call loader
    auto opr = registry->loader(*this, inputs, config);

    // check opr type; note that:
    // 1. registry->type may be empty for dynamic opr loaders or legacy oprs
    // 2. due to some optimization, an opr may be replaced by ImmutableTensor
    mgb_assert(
            opr && (opr->dyn_typeinfo() == registry->type || !registry->type ||
                    opr->same_type<opr::ImmutableTensor>()),
            "got_type=%s expected_type=%s",
            opr ? opr->dyn_typeinfo()->name : nullptr, registry->type->name);
    // record output vars; read output names
    size_t i = 0;
    for (auto ovar : opr->output()) {
        if (!ovar->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            m_id2varnode.push_back(ovar);
            if (fbopr->output_name()) {
                ovar->name(fbopr->output_name()->Get(i++)->str());
            }
        }
    }

    opr->node_prop().attribute().priority = fbopr->priority();
}

GraphLoader::LoadResult GraphLoaderOSS::OprLoadContextImpl::load_oprs() {
    // load oprs
    const auto* oprs = m_loader->m_graph->oprs();
    for (flatbuffers::uoffset_t i = 0; i < oprs->size(); ++i) {
        m_current_opr = oprs->Get(i);
        load_single_opr(m_current_opr);
    }

    // batched loading device values
    m_device_value_loader.apply();

    LoadResult ret;
    ret.graph = m_graph;
    ret.tensor_map = m_tensor_map;

    const auto* outputs = m_loader->m_graph->output_vars_idx();
    ret.output_var_list.resize(outputs->size());
    for (flatbuffers::uoffset_t i = 0; i < outputs->size(); i++) {
        auto out = outputs->Get(i);
        auto var = m_id2varnode.at(out->compact_id());
        ret.output_var_map[var->name()] = var;
        ret.output_var_map_id[out->original_id()] = var;
        ret.output_var_list[i] = var;
    }
    mgb_assert(m_cur_shared_tensor_idx == m_loader->m_shared_tensor_map.size());
    return ret;
}

GraphLoader::LoadResult GraphLoaderOSS::load(const LoadConfig& config,
                                                   bool rewind) {
    mgb_assert(m_file);
    m_cur_load_config = &config;
    if (rewind) {
        m_file->rewind();
    }
    uint32_t magic;
    m_file->read(&magic, sizeof(magic));
    mgb_throw_if(magic != MGB_MAGIC, SerializationError,
                 "wrong magic: wanted %#08x, actual %#08x (not a MegBrain fbs "
                 "model?)",
                 MGB_MAGIC, magic);
    m_file->skip(4);

    uint64_t offset_to_fbs;
    m_file->read(&offset_to_fbs, sizeof(offset_to_fbs));
    auto tensor_begin = m_file->tell();
    // Skip tensor data
    m_file->skip(offset_to_fbs);

    // Read fbs::Graph
    uint32_t size;
    m_file->read(&size, sizeof(size));
    m_graph_buf = m_file->read_shared(size);

    // Rewind back to tensor data
    m_file->rewind();
    m_file->skip(tensor_begin);

    mgb_throw_if(!fbs::GraphBufferHasIdentifier(m_graph_buf.data()),
                 SerializationError, "not a MegBrain fbs model");

    {
        flatbuffers::Verifier verifier(
                static_cast<const uint8_t*>(m_graph_buf.data()),
                m_graph_buf.size());
        mgb_throw_if(!fbs::VerifyGraphBuffer(verifier), SerializationError,
                     "model verification failed (invalid or corrupted model?)");
    }

    m_graph = fbs::GetGraph(m_graph_buf.data());
    m_mgb_version = m_graph->mgb_version();
    if (m_graph->mgb_version() > MGB_VERSION) {
        mgb_log_warn(
                "loading model from future MegBrain: version=%u "
                "model_version=%u",
                MGB_VERSION, m_graph->mgb_version());
    }
    if (!m_graph_hash) {
        m_graph_hash = m_graph->hash();
        mgb_assert(m_graph_hash,
                   "invalid graph hash; maybe error "
                   "occurred during graph dump");
    } else {
        mgb_assert(m_graph_hash == m_graph->hash(),
                   "A GraphLoader instance can be used to load only one graph,"
                   " since the tensor values are shared. Previous graph hash "
                   "is 0x%llx, current graph hash is 0x%llx.",
                   static_cast<unsigned long long>(m_graph_hash),
                   static_cast<unsigned long long>(m_graph->hash()));
    }

    if (m_shared_tensor_map.empty()) {
        m_shared_tensor_map.resize(m_graph->nr_shared_tensor());
    } else {
        mgb_assert(m_shared_tensor_map.size() == m_graph->nr_shared_tensor());
    }

    OprLoadContextImpl ctx{this, m_graph->mgb_version()};
    auto result = ctx.load_oprs();

    auto fbs_end = tensor_begin + offset_to_fbs + sizeof(size) + size;
    auto cur = m_file->tell();
    mgb_assert(fbs_end > cur);
    // Skip to Graph end
    m_file->skip(fbs_end - cur);
    return result;
}

std::unique_ptr<GraphDumper> make_fbs_dumper(std::unique_ptr<OutputFile> file) {
    return std::make_unique<GraphDumperOSS>(std::move(file));
}

std::unique_ptr<GraphLoader> make_fbs_loader(std::unique_ptr<InputFile> file) {
    return std::make_unique<GraphLoaderOSS>(std::move(file));
}

bool is_fbs_file(InputFile& file) {
    uint64_t magic_with_reserved = 0;
    file.read(&magic_with_reserved, sizeof(magic_with_reserved));
    file.skip(-sizeof(magic_with_reserved));
    return magic_with_reserved == MGB_MAGIC;
}

}  // namespace serialization
}  // namespace mgb

#endif
