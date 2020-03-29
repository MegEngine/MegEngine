/**
 * \file sdk/load-and-run/src/json_loader.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "json_loader.h"

using namespace mgb;

template <typename T>
T* JsonLoader::Value::safe_cast() {
    T* ptr = (T*)(this);
    if (nullptr == ptr) {
        fprintf(stderr, "cast ptr is null\n");
    }
    return ptr;
}

std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::operator[](
        const std::string& key) {
    mgb_assert(Type::OBJECT == m_type);
    auto t = safe_cast<JsonLoader::ObjectValue>();
    return t->m_obj.at(key);
}

std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::operator[](
        const size_t index) {
    mgb_assert(Type::ARRAY == m_type);
    auto t = safe_cast<JsonLoader::ArrayValue>();
    return t->m_obj[index];
}

std::map<std::string, std::unique_ptr<JsonLoader::Value>>&
JsonLoader::Value::objects() {
    mgb_assert(Type::OBJECT == m_type);
    auto t = safe_cast<JsonLoader::ObjectValue>();
    return t->m_obj;
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

megdnn::SmallVector<std::unique_ptr<JsonLoader::Value>>&
JsonLoader::Value::array() {
    mgb_assert(Type::ARRAY == m_type);
    auto t = safe_cast<JsonLoader::ArrayValue>();
    return t->m_obj;
}

double JsonLoader::Value::number() {
    mgb_assert(Type::NUMBER == m_type);
    auto t = safe_cast<JsonLoader::NumberValue>();
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
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
        ++p;
    }
    m_buf = p;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_object() {
    expect('{');
    skip_whitespace();

    std::unique_ptr<JsonLoader::Value> ret;
    JsonLoader::ObjectValue* pObject = new JsonLoader::ObjectValue();

    if ('}' == *m_buf) {
        m_buf = m_buf + 1;
        ret.reset((JsonLoader::Value*)(pObject));
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

    ret.reset((JsonLoader::Value*)(pObject));
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_array() {
    expect('[');
    skip_whitespace();

    std::unique_ptr<JsonLoader::Value> ret;
    JsonLoader::ArrayValue* pArray = new JsonLoader::ArrayValue();

    if (']' == *m_buf) {
        m_buf = m_buf + 1;

        ret.reset((JsonLoader::Value*)(pArray));
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

    ret.reset((JsonLoader::Value*)(pArray));
    return ret;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::parse_string() {
    expect('\"');

    std::unique_ptr<JsonLoader::Value> ret;
    JsonLoader::StringValue* pStr = new JsonLoader::StringValue();

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
    ret.reset((JsonLoader::Value*)(pStr));
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

    if (*p == '-')
        p++;
    if (*p == '0')
        p++;
    else {
        loop_digit(std::ref(p));
    }
    if (*p == '.') {
        p++;
        loop_digit(std::ref(p));
    }

    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-')
            p++;
        loop_digit(std::ref(p));
    }
    JsonLoader::NumberValue* pNum = new JsonLoader::NumberValue();
    pNum->m_value = strtod(m_buf, nullptr);

    m_buf = p;

    std::unique_ptr<JsonLoader::Value> ret;
    ret.reset((JsonLoader::Value*)(pNum));
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
        case '\0':
            m_state = State::BAD_TYPE;
            break;
        default:
            return parse_number();
    }
    return nullptr;
}

std::unique_ptr<JsonLoader::Value> JsonLoader::load(const char* content,
                                                    const size_t size) {
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

    std::unique_ptr<char> buf(static_cast<char*>(malloc(size)));

    auto nr = std::fread(buf.get(), 1, size, fin.get());
    mgb_assert(nr == size);

    return load(buf.get(), size);
}
