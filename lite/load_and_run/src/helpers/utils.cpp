#include "utils.h"
using namespace lar;

/////////////////// JsonOptionsCoder ///////////////////
#if MGB_ENABLE_JSON
//! encode option

void encode_single_options(
        std::pair<std::string, std::shared_ptr<lar::Value>> item,
        std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>&
                list,
        bool encode_all) {
    auto type = item.second->get_type();
    if (type == JsonValueType::Bool) {
        auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
        if (!encode_all && val_ptr->get_value() == val_ptr->get_default()) {
            return;
        }
        list.push_back(
                {mgb::json::String(item.first),
                 mgb::json::Bool::make(val_ptr->get_value())});
    } else if (type == JsonValueType::NumberInt32) {
        auto val_ptr = std::static_pointer_cast<lar::NumberInt32>(item.second);
        if (!encode_all && val_ptr->get_value() == val_ptr->get_default()) {
            return;
        }
        list.push_back(
                {mgb::json::String(item.first),
                 mgb::json::NumberInt::make(
                         static_cast<int64_t>(val_ptr->get_value()))});
    } else if (type == JsonValueType::NumberUint64) {
        auto val_ptr = std::static_pointer_cast<lar::NumberUint64>(item.second);
        list.push_back(
                {mgb::json::String(item.first),
                 mgb::json::NumberInt::make(
                         static_cast<int64_t>(val_ptr->get_value()))});
    } else if (type == JsonValueType::Number) {
        auto val_ptr = std::static_pointer_cast<lar::Number>(item.second);
        list.push_back(
                {mgb::json::String(item.first),
                 mgb::json::Number::make(val_ptr->get_value())});
    } else if (type == JsonValueType::String) {
        auto val_ptr = std::static_pointer_cast<lar::String>(item.second);
        if (!encode_all && val_ptr->get_value() == val_ptr->get_default()) {
            return;
        }
        list.push_back(
                {mgb::json::String(item.first),
                 mgb::json::String::make(val_ptr->get_value())});
    } else {
        mgb_log_error(
                "unsupport JsonValueType:%s for lar::Value",
                item.second->type_string().c_str());
    }
}
std::string JsonOptionsCoder::encode(OptionValMap& option_val_map, bool encode_all) {
    std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
            json_options;
    for (auto& item : option_val_map) {
        encode_single_options(item, json_options, encode_all);
    }

    auto json_obj = mgb::json::Object::make(
            {{"options", mgb::json::Object::make(json_options)}});

    return json_obj->to_string(1);
}

//! encode device
std::vector<std::shared_ptr<mgb::json::Object>> JsonOptionsCoder::encode(
        OptionValMap& option_val_map) {
    std::vector<std::shared_ptr<mgb::json::Object>> info;
    std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
            json_device;
    std::vector<std::pair<mgb::json::String, std::shared_ptr<mgb::json::Value>>>
            json_options;
    for (auto& item : option_val_map) {
        if ((item.first == "cpu" || item.first == "cpu_default" ||
             item.first == "multithread" || item.first == "multithread_default")) {
            auto type = item.second->get_type();
            if (type == JsonValueType::Bool) {
                auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
                if (val_ptr->get_value() == val_ptr->get_default())
                    continue;
            }
            if (type == JsonValueType::NumberInt32) {
                auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
                if (val_ptr->get_value() == val_ptr->get_default())
                    continue;
            }
            json_device.push_back(
                    {mgb::json::String("type"), mgb::json::String::make("CPU")});

            if (item.first == "cpu_default" || item.first == "multithread_default") {
                json_device.push_back(
                        {mgb::json::String("enable_inplace_model"),
                         mgb::json::Bool::make(true)});
            }

            if (item.first == "multithread" || item.first == "multithread_default") {
                json_device.push_back(
                        {mgb::json::String("number_threads"),
                         mgb::json::NumberInt::make(
                                 std::static_pointer_cast<lar::NumberInt32>(item.second)
                                         ->get_value())});
                if (item.first == "multithread") {
                    json_device.push_back(
                            {mgb::json::String("device_id"),
                             mgb::json::NumberInt::make(0)});
                }
            }

        } else if (item.first == "cuda") {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (val_ptr->get_value() == val_ptr->get_default())
                continue;
            json_device.push_back(
                    {mgb::json::String("type"), mgb::json::String::make("CUDA")});
            json_device.push_back(
                    {mgb::json::String("device_id"), mgb::json::NumberInt::make(0)});
        } else if (item.first == "opencl") {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (val_ptr->get_value() == val_ptr->get_default())
                continue;
            json_device.push_back(
                    {mgb::json::String("type"), mgb::json::String::make("OPENCL")});
        } else if (
                item.first == "record_comp_seq" || item.first == "record_comp_seq2") {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (val_ptr->get_value() == val_ptr->get_default())
                continue;
            int comp_node_seq_record_level = item.first == "record_comp_seq" ? 1 : 2;
            json_options.push_back(
                    {mgb::json::String("comp_node_seq_record_level"),
                     mgb::json::NumberInt::make(comp_node_seq_record_level)});
        } else if (item.first == "fake_first") {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (val_ptr->get_value() == val_ptr->get_default())
                continue;
            json_options.push_back(
                    {mgb::json::String("fake_next_exec"),
                     mgb::json::Bool::make(val_ptr->get_value())});
        } else if (item.first == "no_sanity_check") {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (val_ptr->get_value() == val_ptr->get_default())
                continue;
            json_options.push_back(
                    {mgb::json::String("var_sanity_check_first_run"),
                     mgb::json::Bool::make(!val_ptr->get_value())});
        } else if (item.first == "weight_preprocess") {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (val_ptr->get_value() == val_ptr->get_default())
                continue;
            json_options.push_back(
                    {mgb::json::String("weight_preprocess"),
                     mgb::json::Bool::make(val_ptr->get_value())});
        }
    }
    info.push_back(mgb::json::Object::make(
            {{"options", mgb::json::Object::make(json_options)}}));
    info.push_back(mgb::json::Object::make(
            {{"device", mgb::json::Object::make(json_device)}}));
    return info;
}
//! decode options note string into option map
OptionValMap& JsonOptionsCoder::decode(
        const std::string& code, OptionValMap& option_val_map) {
    std::shared_ptr<mgb::JsonLoader::Value> root =
            m_json_loader.load(code.c_str(), code.size());
    for (auto& item : root->objects()) {
        auto& value = *item.second;
        //! get all keys in json object
        auto keys = value.keys();

        //! set the json format options into internal options
        for (auto& val : keys) {
            if (value[val]->is_bool()) {
                auto val_ptr = std::static_pointer_cast<lar::Bool>(option_val_map[val]);
                val_ptr->set_value(value[val]->Bool());
            } else if (value[val]->is_number()) {
                auto type = option_val_map[val]->get_type();
                if (type == JsonValueType::Number) {
                    auto val_ptr =
                            std::static_pointer_cast<lar::Number>(option_val_map[val]);
                    val_ptr->set_value(value[val]->number());
                } else if (type == JsonValueType::NumberInt32) {
                    auto val_ptr = std::static_pointer_cast<lar::NumberInt32>(
                            option_val_map[val]);
                    val_ptr->set_value(static_cast<int32_t>(value[val]->number()));
                } else if (type == JsonValueType::NumberUint64) {
                    auto val_ptr = std::static_pointer_cast<lar::NumberUint64>(
                            option_val_map[val]);
                    val_ptr->set_value(static_cast<uint64_t>(value[val]->number()));
                } else {
                    mgb_log_error(
                            "invalid number type:%s to set",
                            option_val_map[val]->type_string().c_str());
                }
            } else if (value[val]->is_str()) {
                auto val_ptr =
                        std::static_pointer_cast<lar::String>(option_val_map[val]);
                val_ptr->set_value(value[val]->str());
            } else {
                mgb_log_error("invalid value type for JsonLoader");
            }
        }
    }
    return option_val_map;
}

#endif

std::string GflagsOptionsCoder::encode(OptionValMap& option_val_map, bool encode_all) {
    std::vector<std::string> gflags_options;
    for (auto& item : option_val_map) {
        auto type = item.second->get_type();
        std::string val = "--";
        if (type == JsonValueType::Bool) {
            auto val_ptr = std::static_pointer_cast<lar::Bool>(item.second);
            if (!encode_all && val_ptr->get_value() == val_ptr->get_default()) {
                continue;
            }
            val += item.first;
            val += "=";
            val += val_ptr->get_value() ? "true" : "false";
            gflags_options.push_back(val);
        } else if (type == JsonValueType::NumberInt32) {
            auto val_ptr = std::static_pointer_cast<lar::NumberInt32>(item.second);
            if (!encode_all && val_ptr->get_value() == val_ptr->get_default()) {
                continue;
            }
            val += item.first;
            val += "=";
            val += std::to_string(val_ptr->get_value());
            gflags_options.push_back(val);
        } else if (type == JsonValueType::NumberUint64) {
            auto val_ptr = std::static_pointer_cast<lar::NumberUint64>(item.second);
            val += item.first;
            val += "=";
            val += std::to_string(val_ptr->get_value());
            gflags_options.push_back(val);
        } else if (type == JsonValueType::Number) {
            auto val_ptr = std::static_pointer_cast<lar::Number>(item.second);
            val += item.first;
            val += "=";
            val += std::to_string(val_ptr->get_value());
            gflags_options.push_back(val);
        } else if (type == JsonValueType::String) {
            auto val_ptr = std::static_pointer_cast<lar::String>(item.second);
            if (!encode_all && val_ptr->get_value() == val_ptr->get_default()) {
                continue;
            }
            val += item.first;
            val += "=\"";
            val += val_ptr->get_value();
            val += "\"";
            gflags_options.push_back(val);
        } else {
            mgb_log_error(
                    "unsupport JsonValueType:%s for lar::Value",
                    item.second->type_string().c_str());
        }
    }
    std::string ret;
    for (auto& item : gflags_options) {
        ret += item;
        ret += "\n";
    }

    return ret;
}

//! decode options note string into option map
OptionValMap& GflagsOptionsCoder::decode(
        const std::string& code, OptionValMap& option_val_map) {
    std::unordered_map<std::string, std::string> gflags_map;
    auto to_raw_string = [](const std::string& str) {
        auto size = str.size();
        std::string ret;
        if ('\"' == str[0] && '\"' == str[size - 1]) {
            ret = str.substr(1, size - 2);
        } else {
            ret = str;
        }
        return ret;
    };

    size_t start = 0;
    size_t end = code.find("\n", start);

    while (end != std::string::npos) {
        auto str = code.substr(start, end - start);
        if (str.substr(0, 2) == "--") {
            size_t idx = str.find("=", 0);
            gflags_map.insert(
                    {str.substr(2, idx - 2), to_raw_string(str.substr(idx + 1))});
        } else {
            mgb_log_error("invaid gflags argument %s", str.c_str());
        }
        start = end + 1;
        end = code.find("\n", start);
    }
    for (auto& item : gflags_map) {
        if (option_val_map.count(item.first) != 0) {
            auto& option_val = option_val_map[item.first];
            auto type = option_val->get_type();
            if (type == JsonValueType::Bool) {
                auto val_ptr = std::static_pointer_cast<lar::Bool>(option_val);
                if (item.second == "true" || item.second == "false") {
                    auto val = item.second == "true";
                    val_ptr->set_value(val);
                }
            } else if (type == JsonValueType::NumberInt32) {
                auto val_ptr = std::static_pointer_cast<lar::NumberInt32>(option_val);
                MGB_TRY {
                    int32_t val = std::stoi(item.second);
                    val_ptr->set_value(val);
                }
                MGB_CATCH(std::exception & exc, {
                    mgb_log_error(
                            "invaid value: %s for %s", item.second.c_str(),
                            item.first.c_str());
                });
            } else if (type == JsonValueType::NumberUint64) {
                auto val_ptr = std::static_pointer_cast<lar::NumberUint64>(option_val);
                MGB_TRY {
                    uint64_t val = std::stoull(item.second);
                    val_ptr->set_value(val);
                }
                MGB_CATCH(std::exception & exc, {
                    mgb_log_error(
                            "invaid value: %s for %s", item.second.c_str(),
                            item.first.c_str());
                });

            } else if (type == JsonValueType::Number) {
                auto val_ptr = std::static_pointer_cast<lar::Number>(option_val);
                MGB_TRY {
                    double val = std::stod(item.second);
                    val_ptr->set_value(val);
                }
                MGB_CATCH(std::exception & exc, {
                    mgb_log_error(
                            "invaid value: %s for %s", item.second.c_str(),
                            item.first.c_str());
                });
            } else if (type == JsonValueType::String) {
                auto val_ptr = std::static_pointer_cast<lar::String>(option_val);
                val_ptr->set_value(item.second);

            } else {
                mgb_log_error(
                        "unsupport JsonValueType:%s for lar::Value",
                        option_val->type_string().c_str());
            }
        } else {
            mgb_log_error("invalid gflags when set runtime options in fitting mode");
        }
    }

    return option_val_map;
}
