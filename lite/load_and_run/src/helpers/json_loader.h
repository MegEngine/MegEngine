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
/*!
 * \brief JSON format data loader for --input
 */
class JsonLoader {
public:
    // base class for different value format
    class Value {
    protected:
        enum struct Type : uint8_t { UNKNOWN, NUMBER, STRING, OBJECT, ARRAY, BOOL };
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

        bool is_bool() { return Type::BOOL == m_type; }

        std::unique_ptr<Value>& operator[](const std::string& key);

        std::unique_ptr<Value>& operator[](const size_t index);

        std::map<std::string, std::unique_ptr<Value>>& objects();

        std::vector<std::string>& keys();

        size_t len();

        megdnn::SmallVector<std::unique_ptr<Value>>& array();

        double number();

        std::string str();

        bool Bool();
    };

    void expect(char c);

    void skip_whitespace();

    std::unique_ptr<Value> parse_object();

    std::unique_ptr<Value> parse_array();

    std::unique_ptr<Value> parse_string();

    std::unique_ptr<Value> parse_number();

    std::unique_ptr<Value> parse_value();

    std::unique_ptr<Value> parse_bool();

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
        friend std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::operator[](
                const size_t index);
        friend megdnn::SmallVector<std::unique_ptr<JsonLoader::Value>>& JsonLoader::
                Value::array();
        friend size_t JsonLoader::Value::len();
    };

    class ObjectValue final : public Value {
        std::map<std::string, std::unique_ptr<Value>> m_obj;
        std::vector<std::string> m_keys;

    public:
        ObjectValue() : Value(Type::OBJECT) {}

        ObjectValue(ObjectValue& arr) : Value(arr) {
            m_obj.clear();
            m_keys.clear();
            for (auto itra = arr.m_obj.begin(); itra != arr.m_obj.end(); ++itra) {
                m_obj.emplace(std::make_pair(itra->first, std::move(itra->second)));
                m_keys.push_back(itra->first);
            }
        }

        ObjectValue(ObjectValue&& arr) : Value(arr) {
            m_obj.clear();
            m_keys.clear();
            for (auto itra = arr.m_obj.begin(); itra != arr.m_obj.end(); ++itra) {
                m_obj.emplace(std::make_pair(itra->first, std::move(itra->second)));
                m_keys.push_back(itra->first);
            }
        }

        friend std::unique_ptr<Value> JsonLoader::parse_object();
        friend std::unique_ptr<JsonLoader::Value>& JsonLoader::Value::operator[](
                const std::string&);
        friend std::map<std::string, std::unique_ptr<JsonLoader::Value>>& JsonLoader::
                Value::objects();
        friend std::vector<std::string>& JsonLoader::Value::keys();
        friend size_t JsonLoader::Value::len();
    };

    class BoolValue final : public Value {
        bool m_value;

    public:
        BoolValue() : Value(Type::BOOL) {}
        bool value() { return m_value; }
        friend std::unique_ptr<Value> JsonLoader::parse_bool();
    };

private:
    const char* m_buf;
    State m_state;
};

}  // namespace mgb
