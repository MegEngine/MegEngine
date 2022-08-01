#include "model_lite.h"
#include <gflags/gflags.h>
#include <cstring>
#include <map>
#include "misc.h"

DECLARE_bool(share_param_mem);

using namespace lar;
ModelLite::ModelLite(const std::string& path) : model_path(path) {
    LITE_LOG("creat lite model use CPU as default comp node");
};

void ModelLite::create_network() {
    m_network = std::make_shared<lite::Network>(config, IO);
}

void ModelLite::load_model() {
    if (share_model_mem) {
        //! WARNNING:maybe not right to share param memmory for this
        LITE_LOG("enable share model memory");

        FILE* fin = fopen(model_path.c_str(), "rb");
        LITE_ASSERT(fin, "failed to open %s: %s", model_path.c_str(), strerror(errno));
        fseek(fin, 0, SEEK_END);
        size_t size = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        void* ptr = malloc(size);
        std::shared_ptr<void> buf{ptr, free};
        auto nr = fread(buf.get(), 1, size, fin);
        LITE_ASSERT(nr == size, "read model file failed");
        fclose(fin);

        m_network->load_model(buf.get(), size);
    } else {
        m_network->load_model(model_path);
    }
}

void ModelLite::run_model() {
    m_network->forward();
}

void ModelLite::wait() {
    m_network->wait();
}
#if MGB_ENABLE_JSON
std::shared_ptr<mgb::json::Object> ModelLite::get_io_info() {
    std::shared_ptr<mgb::json::Array> inputs = mgb::json::Array::make();
    std::shared_ptr<mgb::json::Array> outputs = mgb::json::Array::make();

    auto get_dtype = [&](lite::Layout& layout) {
        std::map<LiteDataType, std::string> type_map = {
                {LiteDataType::LITE_FLOAT, "float32"},
                {LiteDataType::LITE_HALF, "float16"},
                {LiteDataType::LITE_INT64, "int64"},
                {LiteDataType::LITE_INT, "int32"},
                {LiteDataType::LITE_UINT, "uint32"},
                {LiteDataType::LITE_INT16, "int16"},
                {LiteDataType::LITE_UINT16, "uint16"},
                {LiteDataType::LITE_INT8, "int8"},
                {LiteDataType::LITE_UINT8, "uint8"}};
        return type_map[layout.data_type];
    };
    auto make_shape = [](lite::Layout& layout) {
        std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
                shape;
        for (size_t i = 0; i < layout.ndim; ++i) {
            std::string lable = "dim";
            lable += std::to_string(layout.ndim - i - 1);
            shape.push_back(
                    {mgb::json::String(lable),
                     mgb::json::NumberInt::make(layout.shapes[layout.ndim - i - 1])});
        }
        return shape;
    };
    auto input_name = m_network->get_all_input_name();
    for (auto& i : input_name) {
        std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
                json_inp;
        auto layout = m_network->get_io_tensor(i)->get_layout();
        json_inp.push_back(
                {mgb::json::String("shape"),
                 mgb::json::Object::make(make_shape(layout))});
        json_inp.push_back(
                {mgb::json::String("dtype"),
                 mgb::json::String::make(get_dtype(layout))});
        json_inp.push_back({mgb::json::String("name"), mgb::json::String::make(i)});
        inputs->add(mgb::json::Object::make(json_inp));
    }

    auto output_name = m_network->get_all_output_name();
    for (auto& i : output_name) {
        std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
                json_out;
        auto layout = m_network->get_io_tensor(i)->get_layout();
        json_out.push_back(
                {mgb::json::String("shape"),
                 mgb::json::Object::make(make_shape(layout))});
        json_out.push_back(
                {mgb::json::String("dtype"),
                 mgb::json::String::make(get_dtype(layout))});
        json_out.push_back({mgb::json::String("name"), mgb::json::String::make(i)});
        inputs->add(mgb::json::Object::make(json_out));
    }

    return mgb::json::Object::make(
            {{"IO",
              mgb::json::Object::make({{"outputs", outputs}, {"inputs", inputs}})}});
}
#endif

std::vector<uint8_t> ModelLite::get_model_data() {
    std::vector<uint8_t> out_data;
    LITE_THROW("unsupported interface: ModelLite::get_model_data() \n");

    return out_data;
}
