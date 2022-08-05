#include "json_loader.h"

using namespace mgb;

template <typename T>
T* JsonLoader::Value::safe_cast() {
    T* ptr = (T*)(this);
    mgb_assert(nullptr != ptr, "cast ptr is null\n");
    return ptr;
}

std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::operator[](
        const std::string& key) {
    mgb_assert(Type::OBJECT == m_type);
    auto t = safe_cast<JsonLoader::ObjectValue>();
    return t->m_obj.at(key);
}

std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::operator[](const size_t index) {
    mgb_assert(Type::ARRAY == m_type);
    auto t = safe_cast<JsonLoader::ArrayValue>();
    return t->m_obj[index];
}

std::map<std::string, std::unique_ptr<JsonLoader::Value>>& JsonLoader::Value::
        objects() {
    mgb_assert(Type::OBJECT == m_type);
    auto t = safe_cast<JsonLoader::ObjectValue>();
    return t->m_obj;
}

std::vector<std::string>& JsonLoader::Value::keys() {
    mgb_assert(Type::OBJECT == m_type);
    auto t = safe_cast<JsonLoader::ObjectValue>();
    return t->m_keys;
}

size_t JsonLoader::Value::len() {
    if (Type::ARRAY == m_type) {
        auto t = safe_cast<JsonLoader::ArrayValue>();
        return t->m_obj.size();
    } else if (Type::OBJECT == m_type) {
        auto t = safe_cast<JsonLoader::ObjectValue>();
        return t->m_obj.size();
    }
    return 0;
}

megdnn::SmallVector<std::unique_ptr<JsonLoader::Value>>& JsonLoader::Value::array() {
    mgb_assert(Type::ARRAY == m_type);
    auto t = safe_cast<JsonLoader::ArrayValue>();
    return t->m_obj;
}

double JsonLoader::Value::number() {
    mgb_assert(Type::NUMBER == m_type);
    auto t = safe_cast<JsonLoader::NumberValue>();
    return t->value();
}

bool JsonLoader::Value::Bool() {
    mgb_assert(Type::BOOL == m_type);
    auto t = safe_cast<JsonLoader::BoolValue>();
    return t->value();
}

std::string JsonLoader::Value::str() {
    if (Type::STRING == m_type) {
        auto t = safe_cast<StringValue>();
        return t->value();
    }
    return std::string();
}

void JsonLoader::expect(char c) {
    mgb_assert(c == (*m_buf));
    m_buf++;
}

void JsonLoader::skip_whitespace() {
    const char* p = m_buf;
    while (' ' == *p || '\t' == *p || '\n' == *p || '\r' == *p) {
        ++p;
    }
    m_buf = p;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_object() {
    expect('{');
    skip_whitespace();

    std::unique_ptr<JsonLoader::Value> ret;
    std::unique_ptr<JsonLoader::ObjectValue> pObject =
            std::make_unique<JsonLoader::ObjectValue>();

    if ('}' == *m_buf) {
        m_buf = m_buf + 1;
        ret = std::move(pObject);
        return ret;
    }

    while (true) {
        std::unique_ptr<JsonLoader::Value> key = parse_string();
        if (m_state != State::OK) {
            return ret;
        }

        skip_whitespace();
        if (':' != (*m_buf)) {
            m_state = State::MISS_COLON;
            return ret;
        }
        m_buf++;
        skip_whitespace();

        std::unique_ptr<JsonLoader::Value> pVal = parse_value();
        if (m_state != State::OK) {
            return ret;
        }

        if (pObject->m_obj.find(pVal->str()) != pObject->m_obj.end()) {
            m_state = State::KEY_NOT_UNIQUE;
            return ret;
        }

        pObject->m_obj.insert(std::make_pair(key->str(), std::move(pVal)));
        pObject->m_keys.push_back(key->str());

        skip_whitespace();
        if (',' == (*m_buf)) {
            m_buf++;
            skip_whitespace();
        } else if ('}' == (*m_buf)) {
            m_buf++;
            break;
        } else {
            m_state = State::MISS_BRACE;
            break;
        }
    }
    ret = std::move(pObject);
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_array() {
    expect('[');
    skip_whitespace();
    std::unique_ptr<JsonLoader::Value> ret;
    std::unique_ptr<JsonLoader::ArrayValue> pArray =
            std::make_unique<JsonLoader::ArrayValue>();

    if (']' == *m_buf) {
        m_buf = m_buf + 1;

        ret = std::move(pArray);
        return ret;
    }

    while (true) {
        std::unique_ptr<JsonLoader::Value> pVal = parse_value();
        if (m_state != State::OK) {
            mgb_assert(0, "parse value failed during pase array");
            return ret;
        }

        pArray->m_obj.emplace_back(pVal.get());
        pVal.release();

        skip_whitespace();
        if (',' == *m_buf) {
            m_buf++;
            skip_whitespace();
        } else if (']' == *m_buf) {
            m_buf++;
            break;
        } else {
            m_state = State::BAD_ARRAY;
            return ret;
        }
    }

    ret = std::move(pArray);
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_string() {
    expect('\"');
    std::unique_ptr<JsonLoader::StringValue> pStr =
            std::make_unique<JsonLoader::StringValue>();

    const char* p = m_buf;
    while (true) {
        if (*p == '\"') {
            p++;
            break;
        } else {
            pStr->m_value += (*p);
            p++;
        }
    }
    m_buf = p;
    std::unique_ptr<JsonLoader::Value> ret = std::move(pStr);
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_number() {
    const char* p = m_buf;

    auto loop_digit = [this](const char*& p) {
        if (not std::isdigit(*p)) {
            m_state = State::BAD_DIGIT;
            return;
        }
        while (std::isdigit(*p)) {
            p++;
        }
        return;
    };

    if ('-' == *p)
        p++;
    if ('0' == *p)
        p++;
    else {
        loop_digit(std::ref(p));
    }
    if ('.' == *p) {
        p++;
        loop_digit(std::ref(p));
    }

    if ('e' == *p || 'E' == *p) {
        p++;
        if ('+' == *p || '-' == *p)
            p++;
        loop_digit(std::ref(p));
    }
    std::unique_ptr<JsonLoader::NumberValue> pNum =
            std::make_unique<JsonLoader::NumberValue>();
    pNum->m_value = strtod(m_buf, nullptr);

    m_buf = p;

    std::unique_ptr<JsonLoader::Value> ret = std::move(pNum);
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_value() {
    switch (*m_buf) {
        case '[':
            return parse_array();
        case '{':
            return parse_object();
        case '\"':
            return parse_string();
        case 't':
            return parse_bool();
        case 'f':
            return parse_bool();
        case '\0':
            m_state = State::BAD_TYPE;
            break;
        default:
            return parse_number();
    }
    return nullptr;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_bool() {
    const char* p = m_buf;
    std::string value;
    if ('t' == *p) {
        value = "";
        for (size_t idx = 0; idx < 4; ++idx) {
            value += *p++;
        }
    } else if ('f' == *p) {
        value = "";
        for (size_t idx = 0; idx < 5; ++idx) {
            value += *p++;
        }
    }
    bool val = false;
    if ("true" == value) {
        val = true;
    } else if ("false" == value) {
        val = false;
    } else {
        mgb_log_error("invalid value: %s for possible bool value", value.c_str());
    }

    std::unique_ptr<JsonLoader::BoolValue> pBool =
            std::make_unique<JsonLoader::BoolValue>();
    pBool->m_value = val;
    m_buf = p;
    std::unique_ptr<JsonLoader::Value> ret = std::move(pBool);
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::load(
        const char* content, const size_t size) {
    m_buf = content;
    skip_whitespace();
    std::unique_ptr<JsonLoader::Value> value = parse_value();
    skip_whitespace();

    if (m_state != State::OK) {
        return nullptr;
    }
    mgb_assert(size == static_cast<size_t>(m_buf - content));

    return value;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::load(const char* path) {
    std::unique_ptr<std::FILE, void (*)(std::FILE*)> fin(
            std::fopen(path, "rb"), [](std::FILE* fp) { std::fclose(fp); });

    mgb_assert(fin.get(), "failed to open %s: %s", path, strerror(errno));
    std::fseek(fin.get(), 0, SEEK_END);
    const size_t size = ftell(fin.get());
    std::fseek(fin.get(), 0, SEEK_SET);

    std::vector<char> buf(size + 1);

    auto nr = std::fread(buf.data(), 1, size, fin.get());
    mgb_assert(nr == size);

    return load(buf.data(), size);
}
