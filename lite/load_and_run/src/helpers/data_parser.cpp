#include "data_parser.h"
#include <sstream>
#include "json_loader.h"
#include "npy.h"

using namespace lar;

/*!
 * \brief feed different data to diffferent parser
 * \param path data file path or data string
 */
void DataParser::feed(const std::string& path) {
    std::string blob_name = "data", blob_string = path;
    size_t sep = path.find(":");
    if (sep != std::string::npos) {
        blob_name = path.substr(0, sep);
        blob_string = path.substr(sep + 1);
    }

    auto endWith = [blob_string](std::string suffix) -> bool {
        const auto index = blob_string.rfind(suffix);
        if (index != std::string::npos and
            index == blob_string.length() - suffix.length()) {
            return true;
        }
        return false;
    };

    if (endWith(".ppm") || endWith(".pgm")) {
        parse_image(blob_name, blob_string);
    } else if (endWith(".json")) {
        parse_json(blob_string);
    } else if (endWith(".npy")) {
        parse_npy(blob_name, blob_string);
    } else {
        parse_string(blob_name, blob_string);
    }
}

void DataParser::parse_json(const std::string& path) {
    mgb::JsonLoader json;
    std::shared_ptr<mgb::JsonLoader::Value> root = json.load(path.c_str());

    mgb_assert(root != nullptr, "parse json %s fail", path.c_str());
    // parse json to data map
    const std::string SHAPE = "shape", TYPE = "type", RAW = "raw";
    for (auto& item : root->objects()) {
        auto&& value = *item.second;
        auto&& shape = value[SHAPE];
        mgb_assert(shape->is_array());

        auto&& type = value[TYPE];
        mgb_assert(type->is_str());

        auto&& raw = value[RAW];
        mgb_assert(raw->is_array());

        megdnn::SmallVector<size_t> data_shape;
        for (auto&& shape_ptr : shape->array()) {
            data_shape.append({static_cast<size_t>(std::round(shape_ptr->number()))});
        }

        // get type
        const std::map<std::string, megdnn::DType> type_map = {
                {"float32", mgb::dtype::Float32()}, {"float", mgb::dtype::Float32()},
                {"int32", mgb::dtype::Int32()},     {"int", mgb::dtype::Int32()},
                {"int8", mgb::dtype::Int8()},       {"uint8", mgb::dtype::Uint8()}};

        const std::string& type_str = type->str();
        mgb_assert(
                type_map.find(type_str) != type_map.end(),
                "unknown json data type for --input");

        mgb::DType datatype = type_map.at(type_str);
        mgb::HostTensorND hv;
        hv.comp_node(mgb::CompNode::default_cpu(), true)
                .dtype(datatype)
                .resize(data_shape);
        mgb::dt_byte* raw_ptr = hv.raw_ptr();
        size_t elem_size = datatype.size();

        // get raw
        const size_t array_size = raw->len();
        for (size_t idx = 0; idx < array_size; ++idx) {
            double tmp = (*raw)[idx]->number();

            switch (datatype.enumv()) {
                case megdnn::DTypeEnum::Int32: {
                    int32_t ival = std::round(tmp);
                    memcpy(((char*)raw_ptr) + idx * elem_size, &ival, elem_size);
                } break;
                case megdnn::DTypeEnum::Uint8:
                case megdnn::DTypeEnum::Int8: {
                    int8_t cval = std::round(tmp);
                    memcpy(((char*)raw_ptr) + idx, &cval, sizeof(int8_t));
                } break;
                case megdnn::DTypeEnum::Float32: {
                    float fval = tmp;
                    memcpy(((char*)raw_ptr) + idx * elem_size, &fval, elem_size);
                } break;
                default:
                    break;
            }
        }

        inputs.insert(std::make_pair(item.first, std::move(hv)));
    }
}

void DataParser::parse_image(const std::string& name, const std::string& path) {
    // load binary ppm/pgm
    std::ifstream fin;
    fin.open(path, std::ifstream::binary | std::ifstream::in);
    mgb_assert(fin.is_open(), "open file %s failed for --input", path.c_str());

    size_t w = 0, h = 0, channel = 0;
    char buf[128] = {0};

    fin.getline(buf, 128);
    if ('5' == buf[1]) {
        channel = 1;
    } else if ('6' == buf[1]) {
        channel = 3;
    } else {
        mgb_assert(0, "not a formal ppm/pgm");
    }

    while (fin.getline(buf, 128)) {
        if (buf[0] == '#') {
            continue;
        }
        break;
    }
    std::stringstream ss;
    ss << std::string(buf);
    ss >> w;
    ss >> h;

    mgb_assert(w > 0 and h > 0);

    mgb::HostTensorND hv;
    hv.comp_node(mgb::CompNode::default_cpu(), true)
            .dtype(mgb::dtype::Uint8())
            .resize({1, h, w, channel});

    fin.read((char*)(hv.raw_ptr()), hv.layout().total_nr_elems());
    fin.close();
    inputs.insert(std::make_pair(name, std::move(hv)));
}

void DataParser::parse_npy(const std::string& name, const std::string& path) {
    std::string type_str;
    std::vector<npy::ndarray_len_t> stl_shape;
    std::vector<int8_t> raw;
    npy::LoadArrayFromNumpy(path, type_str, stl_shape, raw);

    megdnn::SmallVector<size_t> shape;
    for (auto val : stl_shape) {
        shape.append({static_cast<size_t>(val)});
    }

    const std::map<std::string, megdnn::DType> type_map = {
            {"f4", mgb::dtype::Float32()}, {"i4", mgb::dtype::Int32()},
            {"i2", mgb::dtype::Int16()},   {"u2", mgb::dtype::Uint16()},
            {"i1", mgb::dtype::Int8()},    {"u1", mgb::dtype::Uint8()}};

    megdnn::DType hv_type;
    for (auto& item : type_map) {
        if (type_str.find(item.first) != std::string::npos) {
            hv_type = item.second;
            break;
        }
    }

    mgb::HostTensorND hv;
    hv.comp_node(mgb::CompNode::default_cpu(), true).dtype(hv_type).resize(shape);
    mgb::dt_byte* raw_ptr = hv.raw_ptr();
    memcpy(raw_ptr, raw.data(), raw.size());

    inputs.insert(std::make_pair(name, std::move(hv)));
}

void DataParser::parse_string(const std::string name, const std::string& str) {
    // data type
    megdnn::DType data_type = mgb::dtype::Int32();
    if (str.find(".") != std::string::npos or str.find(".") != std::string::npos) {
        data_type = mgb::dtype::Float32();
    }
    // shape
    size_t number_cnt = 0;

    std::shared_ptr<Brace> brace_root = std::make_shared<Brace>();
    std::shared_ptr<Brace> cur = brace_root;
    for (size_t i = 0; i < str.size(); ++i) {
        char c = str[i];
        if (c == '[') {
            std::shared_ptr<Brace> child = std::make_shared<Brace>();
            child->parent = cur;
            cur->chidren.emplace_back(child);
            cur = child;
        } else if (c == ']') {
            cur = cur->parent.lock();
        } else if (c == ',') {
            number_cnt++;
        }
        continue;
    }
    ++number_cnt;

    mgb_assert(cur == brace_root, "braces not closed for --input");
    megdnn::SmallVector<size_t> shape;
    cur = brace_root;
    while (not cur->chidren.empty()) {
        shape.append({cur->chidren.size()});
        number_cnt /= cur->chidren.size();
        cur = cur->chidren[0];
    }
    mgb_assert(number_cnt > 0);
    shape.append({number_cnt});

    // data
    std::string json_arr;
    for (size_t i = 0; i < str.size(); ++i) {
        char c = str[i];
        if (c != '[' and c != ']') {
            json_arr += c;
        }
    }
    json_arr = "[" + json_arr + "]";

    // reuse json parser to resolve raw data
    mgb::JsonLoader json;
    std::shared_ptr<mgb::JsonLoader::Value> json_root =
            json.load(json_arr.data(), json_arr.size());
    mgb_assert(json_root != nullptr, "parse json fail in parse_string");

    mgb::HostTensorND hv;
    hv.comp_node(mgb::CompNode::default_cpu(), true).dtype(data_type).resize(shape);
    mgb::dt_byte* raw_ptr = hv.raw_ptr();

    const size_t array_len = json_root->len();
    const size_t elem_size = data_type.size();
    for (size_t idx = 0; idx < array_len; ++idx) {
        double tmp = json_root->array()[idx]->number();
        switch (data_type.enumv()) {
            case megdnn::DTypeEnum::Int32: {
                int32_t ival = std::round(tmp);
                memcpy(((char*)raw_ptr) + idx * elem_size, &ival, elem_size);
            } break;
            case megdnn::DTypeEnum::Float32: {
                float fval = tmp;
                memcpy(((char*)raw_ptr) + idx * elem_size, &fval, elem_size);
            } break;
            default:
                break;
        }
    }
    inputs.insert(std::make_pair(name, std::move(hv)));
}
