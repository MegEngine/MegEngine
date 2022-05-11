#pragma once

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/serialization/batched_device_value_loader.h"
#include "megbrain/serialization/internal/schema_v2_generated.h"
#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/serializer.h"

#define CAST_TO_FBS_V2_CTX(cvt) static_cast<GraphLoaderOSSV2::OprLoadContextImpl&>(ctx)

namespace mgb {
namespace serialization {

class GraphDumperOSSV2 final : public GraphDumper, OprDumpContextFlatBuffers {
    const std::unique_ptr<OutputFile> m_file;
    flatbuffers::FlatBufferBuilder m_builder;

    DumpConfig m_config;
    DumpResult m_cur_rst;

    size_t m_nr_shared_tensor;

    std::vector<std::pair<cg::OperatorNodeBase*, const OprRegistryV2*>> m_oprs_to_dump;
    ThinHashMap<VarNode*, VarNode*> m_var_remove_in_dump;

    //! set of output vars specified by user
    ThinHashSet<VarNode*> m_output_vars;
    std::unordered_set<std::string> m_used_input_names, m_used_param_names;

    //! current opr to be dumped
    cg::OperatorNodeBase* m_cur_opr = nullptr;
    // Will be filled in dump_tensor
    std::vector<flatbuffers::Offset<fbs::v2::Tensor>> m_cur_opr_tensor;
    std::vector<flatbuffers::Offset<fbs::v2::Blob>> m_blobs;
    std::vector<fbs::v2::OperatorParam> m_cur_opr_param_type;
    std::vector<flatbuffers::Offset<void>> m_cur_opr_param;

    std::vector<flatbuffers::Offset<fbs::v2::MiddleTensor>> m_model_middle_tensors;
    ThinHashMap<VarNode*, size_t> m_var2midtensor_id;

    SymbolVarArray converter_all_opr_to_compatiable(const SymbolVarArray& output_vars);

    void init_oprs_to_dump(const SymbolVarArray& endpoints);

    flatbuffers::Offset<fbs::v2::Metadata> build_metadata(const Metadata& metadata);
    flatbuffers::Offset<fbs::v2::Operator> build_single_opr(
            cg::OperatorNodeBase* opr, const OprRegistryV2* registry);

    flatbuffers::Offset<fbs::DType> build_dtype(DType dtype);

public:
    GraphDumperOSSV2(std::unique_ptr<OutputFile> file) : m_file{std::move(file)} {}

    DumpResult dump(
            const SymbolVarArray& output_vars, const DumpConfig& config = {},
            const Metadata& metadata = {}) override;

    const GraphDumpConfig& config() const override { return m_config; }

    void dump_tensor(
            const std::string& name, const HostTensorND& tensor,
            TensorWriteMethod method) override;

    void append_param(uint32_t type, uint32_t value) override {
        static_assert(
                std::is_same<uint32_t, flatbuffers::uoffset_t>::value,
                "append_param depends on uoffset_t being uint32_t");
        static_assert(
                std::is_standard_layout<flatbuffers::Offset<void>>::value,
                "append_param depends on flatbuffers::Offset having "
                "standard memory layout");
        mgb_assert(type != fbs::v2::OperatorParam_NONE);
        m_cur_opr_param_type.emplace_back(static_cast<fbs::v2::OperatorParam>(type));
        m_cur_opr_param.emplace_back(value);
    }

    flatbuffers::FlatBufferBuilder& builder() override { return m_builder; }
    void dump_buf_with_len(const void* data, uint32_t size) override;

    GraphDumpFormat format() const override { return GraphDumpFormat::FLATBUFFERS_V2; }
    flatbuffers::Offset<fbs::v2::MiddleTensor> build_middle_tensor(const SymbolVar var);
    flatbuffers::Offset<fbs::v2::OutputVar> build_output_var(const SymbolVar var);
    flatbuffers::Offset<void> build_tensor_format(const TensorLayout::Format& format);

    void set_current_opr(cg::OperatorNodeBase* cur_opr) { m_cur_opr = cur_opr; }
};

// ----------------------------- Loader --------------------------------------
class GraphLoaderOSSV2 final : public GraphLoader {
    const LoadConfig* m_cur_load_config = nullptr;
    std::unique_ptr<InputFile> m_file;
    SharedBuffer m_model_buf{{}, 0};
    const fbs::v2::Model* m_model;
    SharedTensorIDMap m_shared_tensor_map;
    uint32_t m_mgb_version = 0;
    bool m_model_loaded = false;

    void verify();

public:
    class OprLoadContextImpl;
    friend class OprLoadContextImpl;

    GraphLoaderOSSV2(std::unique_ptr<InputFile> input_file)
            : m_file{std::move(input_file)} {}

    std::unique_ptr<InputFile> reset_file(std::unique_ptr<InputFile> file) override {
        file.swap(m_file);
        return file;
    }

    LoadResult load(const LoadConfig& config, bool rewind) override;

    const SharedTensorIDMap& shared_tensor_id_map() const override {
        mgb_assert(m_model_loaded, "graph not loaded yet");
        return m_shared_tensor_map;
    }

    GraphDumpFormat format() const override { return GraphDumpFormat::FLATBUFFERS_V2; }
};

class GraphLoaderOSSV2::OprLoadContextImpl final : public OprLoadContextFlatBuffers {
    GraphLoaderOSSV2* const m_loader;
    size_t m_cur_shared_tensor_idx = 0;
    std::shared_ptr<ComputingGraph> m_graph;
    LoadResult::TensorMap m_tensor_map;
    VarNodeArray m_id2varnode;
    std::vector<const fbs::v2::MiddleTensor*> m_middle_tensors;
    BatchedDeviceValueLoader m_device_value_loader;
    const fbs::v2::Operator* m_current_opr;
    size_t m_cur_opr_tensor_cnt;
    size_t m_cur_opr_blob_cnt;
    size_t m_cur_opr_param_cnt;

public:
    ComputingGraph& graph() override { return *m_graph; }

    const GraphLoadConfig& config() const override {
        return *m_loader->m_cur_load_config;
    }

    std::shared_ptr<HostTensorND> load_tensor() override;

    std::shared_ptr<DeviceTensorND> load_tensor_shared() override;

    void load_single_opr(const fbs::v2::Operator* opr);

    OprLoadContextImpl(GraphLoaderOSSV2* loader, uint32_t version)
            : OprLoadContextFlatBuffers(version), m_loader{loader} {
        m_graph = loader->m_cur_load_config->comp_graph;
        if (!m_graph) {
            m_graph = ComputingGraph::make();
        }
        auto maker = [this]() {
            return std::shared_ptr<OprLoadContext>{
                    std::shared_ptr<OprLoadContext>{}, this};
        };
        auto got = m_graph->options().user_data.get_user_data_or_create<OprLoadContext>(
                maker);
        mgb_assert(got == this);
    }

    ~OprLoadContextImpl() noexcept {
        auto nr = m_graph->options().user_data.pop_user_data<OprLoadContext>();
        mgb_assert(nr == 1);
    }

    Metadata load_metadata();
    LoadResult load_oprs();
    CompNode load_comp_node(const fbs::v2::CompNode* comp_node);

    void load_middle_tensor();

    const void* get_next_param(uint32_t enumv) override {
        auto type = static_cast<fbs::v2::OperatorParam>(enumv);
        if (m_cur_opr_param_cnt == 0) {
            m_cur_opr_param_cnt++;
            if (m_current_opr->param_type() == type) {
                return m_current_opr->param();
            } else {
                mgb_throw(
                        SerializationError,
                        "The param type is not match when load the opr.");
            }
        }
        mgb_throw(
                SerializationError,
                "When load multi param in one Operator, please use read_param(index) "
                "interface. ");
    }

    std::string load_buf_with_len() override {
        mgb_assert(
                m_current_opr->custom_data() &&
                m_cur_opr_blob_cnt < m_current_opr->custom_data()->size());
        auto blob = m_current_opr->custom_data()->Get(m_cur_opr_blob_cnt++);
        mgb_assert(blob && blob->data());
        auto data = blob->data()->data();
        return {reinterpret_cast<const char*>(data), blob->data()->size()};
    }

    SharedBuffer load_shared_buf_with_len() override {
        mgb_assert(
                m_current_opr->custom_data() &&
                m_cur_opr_blob_cnt < m_current_opr->custom_data()->size());
        auto blob = m_current_opr->custom_data()->Get(m_cur_opr_blob_cnt++);
        mgb_assert(blob && blob->data());
        auto size = blob->data()->size();
        std::shared_ptr<uint8_t> shptr{
                new uint8_t[size], [](uint8_t* p) { delete[] p; }};
        memcpy(shptr.get(), blob->data()->data(), size);
        return {std::move(shptr), size};
    }

    const void* get_current_opr_data() override {
        return reinterpret_cast<const void*>(m_current_opr);
    }

    template <class T>
    T read_param(int index) {
        using SourceType = typename fbs::ParamConverter<T>::FlatBufferType;
        auto enumv = fbs::OperatorParamTraits<SourceType>::enum_value;
        auto type = static_cast<fbs::v2::OperatorParam>(enumv);
        if (index == 0) {
            mgb_assert(
                    m_current_opr->param_type() == type,
                    "Load param error, the param type is not right.");
            return fbs::ParamConverter<T>::to_param(
                    static_cast<const SourceType*>(m_current_opr->param()));
        } else {
            int addition_index = index - 1;
            if (addition_index >=
                static_cast<int>(m_current_opr->additional_params()->size())) {
                mgb_log_warn(
                        "Model has no addition param of index %d, just construct a "
                        "default one.",
                        addition_index);
            } else {
                mgb_assert(
                        m_current_opr->additional_params_type()->Get(addition_index) ==
                                type,
                        "Load param error, the addition param type is not right.");
                return fbs::ParamConverter<T>::to_param(static_cast<const SourceType*>(
                        m_current_opr->additional_params()->Get(addition_index)));
            }
        }
    }
};

}  // namespace serialization
}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
