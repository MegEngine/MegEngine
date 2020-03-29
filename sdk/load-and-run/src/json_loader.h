/**
 * \file sdk/load-and-run/src/json_loader.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include "megbrain/common.h"
#include "megdnn/thin/small_vector.h"

namespace mgb {

class JsonLoader {
public:
    class Value {
    protected:
        enum struct Type : uint8_t { UNKNOWN, NUMBER, STRING, OBJECT, ARRAY };
        Type m_type;

    public:
        template <typename T>
        T* safe_cast();

        Value() { m_type = Type::UNKNOWN; }

        Value(Type type) : m_type(type) {}

        virtual ~Value() {}

        bool is_array() { return Type::ARRAY == m_type; }

        bool is_object() { return Type::OBJECT == m_type; }

        bool is_number() { return Type::NUMBER == m_type; }

        bool is_str() { return Type::STRING == m_type; }

        std::unique_ptr<Value>& operator[](const std::string& key);

        std::unique_ptr<Value>& operator[](const size_t index);

        std::map<std::string, std::unique_ptr<Value>>& objects();

        size_t len();

        megdnn::SmallVector<std::unique_ptr<Value>>& array();

        double number();

        std::string str();
    };

    void expect(char c);

    void skip_whitespace();

    std::unique_ptr<Value> parse_object();

    std::unique_ptr<Value> parse_array();

    std::unique_ptr<Value> parse_string();

    std::unique_ptr<Value> parse_number();

    std::unique_ptr<Value> parse_value();

    enum struct State : uint8_t {
        OK = 0,
        BAD_TYPE,
        BAD_DIGIT,
        BAD_ARRAY,
        MISS_COLON,
        MISS_BRACE,
        KEY_NOT_UNIQUE
    };

    JsonLoader() { m_state = State::OK; }

    std::unique_ptr<Value> load(const char* content, const size_t size);

    std::unique_ptr<Value> load(const char* path);

    class NumberValue final : public Value {
        friend std::unique_ptr<Value> JsonLoader::parse_number();
        double m_value;

    public:
        NumberValue() : Value(Type::NUMBER) {}

        double value() { return m_value; }
    };

    class StringValue final : public Value {
        std::string m_value;

    public:
        StringValue() : Value(Type::STRING) {}

        std::string value() { return m_value; }

        friend std::unique_ptr<Value> JsonLoader::parse_string();
    };

    class ArrayValue final : public Value {
        megdnn::SmallVector<std::unique_ptr<Value>> m_obj;

    public:
        ArrayValue() : Value(Type::ARRAY) {}

        ArrayValue(ArrayValue& arr) : Value(arr) {
            m_obj.clear();
            for (auto& item : arr.m_obj) {
                m_obj.emplace_back(item.get());
                item.release();
            }
        }

        ArrayValue(ArrayValue&& arr) : Value(arr) {
            m_obj.clear();
            for (auto& item : arr.m_obj) {
                m_obj.emplace_back(item.get());
                item.release();
            }
        }

        friend std::unique_ptr<Value> JsonLoader::parse_array();
        friend std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::
        operator[](const size_t index);
        friend megdnn::SmallVector<std::unique_ptr<JsonLoader::Value>>&
        JsonLoader::Value::array();
        friend size_t JsonLoader::Value::len();
    };

    class ObjectValue final : public Value {
        std::map<std::string, std::unique_ptr<Value>> m_obj;

    public:
        ObjectValue() : Value(Type::OBJECT) {}

        ObjectValue(ObjectValue& arr) : Value(arr) {
            m_obj.clear();
            for (auto itra = arr.m_obj.begin(); itra != arr.m_obj.end();
                 ++itra) {
                m_obj.emplace(
                        std::make_pair(itra->first, std::move(itra->second)));
            }
        }

        ObjectValue(ObjectValue&& arr) : Value(arr) {
            m_obj.clear();
            for (auto itra = arr.m_obj.begin(); itra != arr.m_obj.end();
                 ++itra) {
                m_obj.emplace(
                        std::make_pair(itra->first, std::move(itra->second)));
            }
        }

        friend std::unique_ptr<Value> JsonLoader::parse_object();
        friend std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::
        operator[](const std::string&);
        friend std::map<std::string, std::unique_ptr<JsonLoader::Value>>&
        JsonLoader::Value::objects();
        friend size_t JsonLoader::Value::len();
    };

private:
    const char* m_buf;
    State m_state;
};

}  // namespace mgb
