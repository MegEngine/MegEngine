#include <string>
#include <memory>
#include "megbrain/utils/json.h"

namespace mgb {
namespace imperative {

class ChromeTraceEvent {
public:
    ChromeTraceEvent& name(std::string name) {
        m_name = std::move(name);
        return *this;
    }
    ChromeTraceEvent& tid(uint64_t tid) {
        m_tid = std::move(tid);
        return *this;
    }
    ChromeTraceEvent& cat(std::string cat) {
        m_cat = std::move(cat);
        return *this;
    }
    ChromeTraceEvent& pid(uint64_t pid) {
        m_pid = pid;
        return *this;
    }
    ChromeTraceEvent& id(uint64_t id) {
        m_id = id;
        return *this;
    }
    ChromeTraceEvent& idx(uint64_t idx) {
        m_idx = idx;
        return *this;
    }
    ChromeTraceEvent& ts(double ts) {
        m_ts = ts;
        return *this;
    }
    ChromeTraceEvent& dur(double dur) {
        m_dur = dur;
        return *this;
    }
    ChromeTraceEvent& ph(char ph) {
        m_ph = ph;
        return *this;
    }
    ChromeTraceEvent& bp(char bp) {
        m_bp = bp;
        return *this;
    }
    ChromeTraceEvent& args(std::shared_ptr<json::Object> args) {
        m_args = std::move(args);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, std::string value) {
        if (!m_args) {
            m_args = json::Object::make();
        }
        (*m_args)[key] = json::String::make(value);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, double value) {
        if (!m_args) {
            m_args = json::Object::make();
        }
        (*m_args)[key] = json::Number::make(value);
        return *this;
    }
    ChromeTraceEvent& arg(std::string key, std::shared_ptr<json::Value> value) {
        if (!m_args) {
            m_args = json::Object::make();
        }
        (*m_args)[key] = value;
        return *this;
    }

    std::shared_ptr<json::Object> to_json() const {
        auto result = json::Object::make();
        auto prop_str = [&](auto key, auto value) {
            if (value.empty()) {
                return;
            }
            (*result)[key] = json::String::make(value);
        };
        auto prop_num = [&](auto key, auto value) {
            if (!value) {
                return;
            }
            (*result)[key] = json::Number::make(value.value());
        };
        auto prop_char = [&](auto key, auto value) {
            if (!value) {
                return;
            }
            (*result)[key] = json::String::make(std::string{} + value.value());
        };
        prop_str("name", m_name);
        prop_num("tid", m_tid);
        prop_str("cat", m_cat);
        prop_num("pid", m_pid);
        prop_num("id", m_id);
        prop_num("idx", m_idx);
        prop_num("ts", m_ts);
        prop_num("dur", m_dur);
        prop_char("ph", m_ph);
        prop_char("bp", m_bp);
        if (m_args) {
            (*result)["args"] = m_args;
        }
        return result;
    }
private:
    std::string m_name;
    std::string m_cat;

    std::optional<uint64_t> m_tid;
    std::optional<uint64_t> m_pid;
    std::optional<uint64_t> m_id;
    std::optional<uint64_t> m_idx;
    std::optional<double> m_ts;
    std::optional<double> m_dur;
    std::optional<char> m_ph;
    std::optional<char> m_bp;
    std::shared_ptr<json::Object> m_args;
};

class ChromeTraceEventList {
public:
    ChromeTraceEvent& new_event() {
        m_content.emplace_back();
        return m_content.back();
    }

    std::shared_ptr<json::Array> to_json() const {
        auto result = json::Array::make();
        for (auto&& event: m_content) {
            result->add(event.to_json());
        }
        return result;
    }
private:
    std::vector<ChromeTraceEvent> m_content;
};

}  // namespace imperative
}  // namespace mgb
