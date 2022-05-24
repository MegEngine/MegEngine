#include "lite/pack_model.h"
#include "../misc.h"
#if LITE_BUILD_WITH_MGE
#include "megbrain/utils/infile_persistent_cache.h"
#endif

#include <flatbuffers/flatbuffers.h>
#include "nlohmann/json.hpp"
#include "pack_model_generated.h"

namespace lite {

class FbsHelper {
public:
    FbsHelper() = default;
    FbsHelper(ModelPacker* packer, std::string model_path);
    FbsHelper(ModelPacker* packer, std::vector<uint8_t>& model_data);
    flatbuffers::Offset<model_parse::ModelHeader> build_header();
    flatbuffers::Offset<model_parse::ModelInfo> build_info();
    flatbuffers::Offset<model_parse::ModelData> build_data();
    flatbuffers::FlatBufferBuilder& builder() { return m_builder; }

private:
    ModelPacker* m_packer;
    flatbuffers::FlatBufferBuilder m_builder;
    std::vector<uint8_t> m_model_buffer;

    const model_parse::ModelHeader* m_model_header = nullptr;
    const model_parse::ModelInfo* m_model_info = nullptr;
    const model_parse::ModelData* m_model_data = nullptr;
};

}  // namespace lite

using namespace lite;
using namespace model_parse;

std::vector<uint8_t> read_file(std::string path) {
    FILE* fin = fopen(path.c_str(), "rb");
    LITE_ASSERT(fin, "failed to open %s: %s", path.c_str(), strerror(errno));
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    std::vector<uint8_t> buf;
    buf.resize(size);
    auto nr = fread(buf.data(), size, 1, fin);
    LITE_ASSERT(nr == 1);
    fclose(fin);
    return buf;
}
FbsHelper::FbsHelper(ModelPacker* packer, std::vector<uint8_t>& model_data)
        : m_packer(packer), m_model_buffer(model_data) {
    const char* model_ptr =
            static_cast<const char*>(static_cast<void*>(m_model_buffer.data()));
    std::string tag(model_ptr, 12);
    if (tag == "packed_model") {
        uint8_t* buffer = m_model_buffer.data() + 12;
        auto model = GetPackModel(buffer)->models()->Get(0);
        m_model_header = model->header();
        m_model_info = model->info();
        m_model_data = model->data();
    }
}

FbsHelper::FbsHelper(ModelPacker* packer, std::string model_path) : m_packer(packer) {
    m_model_buffer = read_file(model_path);

    const char* model_ptr =
            static_cast<const char*>(static_cast<void*>(m_model_buffer.data()));
    std::string tag(model_ptr, 12);
    if (tag == "packed_model") {
        uint8_t* buffer = m_model_buffer.data() + 12;
        auto model = GetPackModel(buffer)->models()->Get(0);
        m_model_header = model->header();
        m_model_info = model->info();
        m_model_data = model->data();
    }
}

flatbuffers::Offset<ModelHeader> FbsHelper::build_header() {
    flatbuffers::Offset<flatbuffers::String> name, info_decryption_method,
            info_parse_method, model_decryption_method, info_cache_parse_method;
    bool is_fast_run_cache;
    if (m_model_header) {
        auto&& header = m_model_header;
        name = m_builder.CreateSharedString(header->name());
        info_decryption_method =
                m_builder.CreateSharedString(header->info_decryption_method());
        info_parse_method = m_builder.CreateSharedString(header->info_parse_method());
        model_decryption_method =
                m_builder.CreateSharedString(header->model_decryption_method());
        info_cache_parse_method =
                m_builder.CreateSharedString(header->info_cache_parse_method());
        is_fast_run_cache = header->is_fast_run_cache();
    } else {
        auto&& header = m_packer->m_header;
        name = m_builder.CreateSharedString(header.name);
        info_decryption_method =
                m_builder.CreateSharedString(header.info_decryption_method);
        info_parse_method = m_builder.CreateSharedString(header.info_parse_method);
        model_decryption_method =
                m_builder.CreateSharedString(header.model_decryption_method);
        info_cache_parse_method =
                m_builder.CreateSharedString(header.info_cache_parse_method);
        is_fast_run_cache = header.fb32.is_fast_run_cache;
    }
    return CreateModelHeader(
            m_builder, name, info_decryption_method, info_parse_method,
            model_decryption_method, info_cache_parse_method, is_fast_run_cache);
}

flatbuffers::Offset<ModelData> FbsHelper::build_data() {
    if (m_model_data) {
        auto data = m_model_data->data()->Data();
        auto size = m_model_data->data()->size();
        return CreateModelData(m_builder, m_builder.CreateVector(data, size));
    } else {
        return CreateModelData(m_builder, m_builder.CreateVector(m_model_buffer));
    }
}

flatbuffers::Offset<ModelInfo> FbsHelper::build_info() {
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_data;
    if (m_model_info && m_model_info->data() && m_packer->m_info_data.empty()) {
        auto data = m_model_info->data()->Data();
        auto size = m_model_info->data()->size();
        fb_data = m_builder.CreateVector(data, size);
    } else if (!m_packer->m_info_data.empty()) {
        fb_data = m_builder.CreateVector(m_packer->m_info_data);
    }

    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_algo_policy;
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_binary_cache;
    if (m_packer->m_header.fb32.is_fast_run_cache) {
        std::vector<uint8_t> info_algo_policy;
        if (!m_packer->m_algo_policy_data.empty()) {
            info_algo_policy = m_packer->m_algo_policy_data;
            if (m_model_info && m_model_info->algo_policy()) {
                auto cache = m_model_info->algo_policy()->Data();
                auto size = m_model_info->algo_policy()->size();

                uint32_t nr_category_1, nr_category_2, nr_category;
                memcpy(&nr_category_1, cache, sizeof(uint32_t));
                memcpy(&nr_category_2, info_algo_policy.data(), sizeof(uint32_t));
                nr_category = nr_category_1 + nr_category_2;

                std::vector<uint8_t> cache_append;
                cache_append.resize(sizeof(nr_category));
                memcpy(cache_append.data(), &nr_category, sizeof(nr_category));
                cache_append.insert(
                        cache_append.end(), cache + sizeof(nr_category), cache + size);
                cache_append.insert(
                        cache_append.end(),
                        info_algo_policy.begin() + sizeof(nr_category),
                        info_algo_policy.end());

                fb_algo_policy = m_builder.CreateVector(cache_append);
            } else {
                fb_algo_policy = m_builder.CreateVector(info_algo_policy);
            }
        }
#if LITE_BUILD_WITH_MGE
        else {
            info_algo_policy = static_cast<mgb::InFilePersistentCache&>(
                                       mgb::PersistentCache::inst())
                                       .dump_cache();
            fb_algo_policy = m_builder.CreateVector(info_algo_policy);
        }
#endif
    }

    ModelInfoBuilder builder(m_builder);
    builder.add_data(fb_data);
    builder.add_algo_policy(fb_algo_policy);
    builder.add_binary_cache(fb_binary_cache);
    return builder.Finish();
}

ModelPacker::ModelPacker(
        std::string model_path, std::string packed_model_path,
        std::string info_data_path, std::string info_algo_policy_path,
        std::string info_binary_cache_path)
        : m_packed_model_path(packed_model_path) {
    m_fbs_helper = std::make_shared<FbsHelper>(this, model_path);
    std::vector<uint8_t> empty_vec;
    m_info_data = info_data_path.empty() ? empty_vec : read_file(info_data_path);
    m_algo_policy_data = info_algo_policy_path.empty()
                               ? empty_vec
                               : read_file(info_algo_policy_path);
    m_binary_cache_data = info_binary_cache_path.empty()
                                ? empty_vec
                                : read_file(info_binary_cache_path);
}

ModelPacker::ModelPacker(
        std::vector<uint8_t> model_data, std::string packed_model_path,
        std::vector<uint8_t> info_data, std::vector<uint8_t> info_algo_policy_data,
        std::vector<uint8_t> info_binary_cache_data) {
    m_fbs_helper = std::make_shared<FbsHelper>(this, model_data);
    m_packed_model_path = packed_model_path;
    m_info_data = info_data;
    m_algo_policy_data = info_algo_policy_data;
    m_binary_cache_data = info_binary_cache_data;
}

void ModelPacker::set_header(
        std::string model_decryption_method, std::string info_decryption_method,
        bool is_fast_run_cache) {
    m_header.model_decryption_method = model_decryption_method;
    m_header.info_decryption_method = info_decryption_method;
    memset(&m_header.fb32, 0, sizeof(m_header.fb32));
    m_header.fb32.is_fast_run_cache = is_fast_run_cache;
    if (!m_info_data.empty()) {
        std::string json_string(
                static_cast<const char*>(static_cast<void*>(m_info_data.data())),
                m_info_data.size());
        auto info = nlohmann::json::parse(json_string);
        m_header.name = info["name"];
    }
}

void ModelPacker::pack_model() {
    auto fb_header = m_fbs_helper->build_header();
    auto fb_info = m_fbs_helper->build_info();
    auto fb_data = m_fbs_helper->build_data();

    ModelBuilder model_builder(m_fbs_helper->builder());
    model_builder.add_header(fb_header);
    model_builder.add_info(fb_info);
    model_builder.add_data(fb_data);

    auto model = model_builder.Finish();
    std::vector<flatbuffers::Offset<Model>> models;
    models.emplace_back(model);
    auto fb_models = m_fbs_helper->builder().CreateVector(models);

    PackModelBuilder pack_model_builder(m_fbs_helper->builder());
    pack_model_builder.add_models(fb_models);
    m_fbs_helper->builder().Finish(pack_model_builder.Finish());

    FILE* fptr = fopen(m_packed_model_path.c_str(), "wb");
    LITE_ASSERT(
            fptr, "failed to open %s: %s", m_packed_model_path.c_str(),
            strerror(errno));
    std::string packed_model_tag = "packed_model";
    auto nr_tag = fwrite(packed_model_tag.c_str(), 1, packed_model_tag.size(), fptr);
    LITE_ASSERT(nr_tag == packed_model_tag.size());

    auto fb_size = m_fbs_helper->builder().GetSize();
    auto nr_fb = fwrite(m_fbs_helper->builder().GetBufferPointer(), 1, fb_size, fptr);
    LITE_ASSERT(nr_fb == fb_size);
    fclose(fptr);
}