/**
 * \file imperative/src/impl/persistent_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <fstream>
#include <string>
#include <vector>

#include "cpp_redis/cpp_redis"

#include "megbrain/imperative/persistent_cache.h"
#include "megbrain/imperative/utils/base64.h"
#include "megbrain/utils/infile_persistent_cache.h"

namespace mgb::imperative::persistent_cache {

class RedisCache final : public ExtendedPersistentCache {
public:
    RedisCache(std::string prefix, uint64_t timeout) : m_prefix(prefix) {
        m_local = std::make_shared<mgb::InMemoryPersistentCache>();
    }

    void connect(std::string ip, size_t port, std::optional<std::string> password) {
        if (password) {
            m_client.auth(*password);
        }
        m_client.connect(
                ip, port,
                [](const std::string& host, std::size_t port,
                   cpp_redis::connect_state status) {
                    if (status == cpp_redis::connect_state::dropped) {
                        mgb_log("client disconnected from %s.", host.c_str());
                        mgb_log("Redis server connect to %s :%zu failed.", host.c_str(),
                                port);
                    }
                },
                std::uint32_t(200));
        mgb_assert(m_client.is_connected(), "connect failed");
        auto flag = m_client.get("mgb-cache-flag");
        sync();
        auto is_valid = [](const cpp_redis::reply& reply) {
            switch (reply.get_type()) {
                case cpp_redis::reply::type::error:
                case cpp_redis::reply::type::null:
                    return false;
                case cpp_redis::reply::type::integer:
                    return reply.as_integer() != 0;
                case cpp_redis::reply::type::simple_string:
                case cpp_redis::reply::type::bulk_string:
                    return !reply.as_string().empty();
                case cpp_redis::reply::type::array:
                    return !reply.as_array().empty();
                default:
                    mgb_assert(false, "unknown reply type %d", (int)reply.get_type());
            }
        };
        mgb_assert(is_valid(flag.get()), "invalid mgb-cache-flag");
    }

    bool valid() const override { return m_client.is_connected(); }

    void flush() override {}

    mgb::Maybe<Blob> get(const std::string& category, const Blob& key) override {
        MGB_LOCK_GUARD(m_mtx);
        auto mem_result = m_local->get(category, key);
        if (mem_result.valid()) {
            return mem_result;
        }
        std::string key_str(static_cast<const char*>(key.ptr), key.size);
        std::string redis_key_str;
        encode(category + '@' + key_str, redis_key_str, 24);
        auto result = m_client.get(m_prefix + redis_key_str);
        sync();
        auto content = result.get();
        if (content.is_null()) {
            return None;
        }
        std::string decode_content;
        decode(content.as_string(), decode_content);
        m_local->put(category, key, {decode_content.data(), decode_content.length()});
        return m_local->get(category, key);
    }

    void put(const std::string& category, const Blob& key, const Blob& value) override {
        MGB_LOCK_GUARD(m_mtx);
        std::string key_str(static_cast<const char*>(key.ptr), key.size);
        std::string redis_key_str;
        encode(category + '@' + key_str, redis_key_str, 24);
        std::string value_str(static_cast<const char*>(value.ptr), value.size);
        std::string redis_value_str;
        encode(value_str, redis_value_str);
        auto result = m_client.set(m_prefix + redis_key_str, redis_value_str);
        m_local->put(category, key, value);
        sync();
    }

    std::optional<size_t> clear() override {
        size_t cursor = 0, nr_deleted = 0;
        std::string pattern = m_prefix + "@*";
        do {
            auto reply = m_client.scan(cursor, pattern).share();
            sync();
            auto keys = reply.get().as_array();
            std::vector<std::string> string_keys;
            for (auto&& key : keys) {
                string_keys.push_back(key.as_string());
            }
            m_client.del(string_keys);
            nr_deleted += string_keys.size();
            cursor = reply.get().as_array()[0].as_integer();
        } while (cursor != 0);
        return nr_deleted;
    }

private:
    std::shared_ptr<mgb::PersistentCache> m_local;
    std::mutex m_mtx;
    cpp_redis::client m_client;
    std::string m_prefix;
    uint64_t m_timeout;

    void sync() {
        m_client.sync_commit<double, std::milli>(std::chrono::milliseconds(m_timeout));
        mgb_assert(valid());
    }
};

class ExtendedInFilePersistentCache final : public ExtendedPersistentCache {
private:
    std::optional<std::string> m_path;
    std::unique_ptr<mgb::InFilePersistentCache> m_impl;

public:
    ExtendedInFilePersistentCache() = default;

    void open(std::string path) {
        std::fstream file;
        file.open(path, std::ios::in | std::ios::binary);
        mgb_assert(file.is_open(), "can't open file in %s", path.c_str());
        std::vector<char> bytes = {
                std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        if (bytes.size()) {
            m_impl = std::make_unique<mgb::InFilePersistentCache>(
                    (const uint8_t*)bytes.data(), bytes.size());
        } else {
            m_impl = std::make_unique<mgb::InFilePersistentCache>();
        }
        m_path = path;
    }

    void open() { m_impl = std::make_unique<mgb::InFilePersistentCache>(); }

    ~ExtendedInFilePersistentCache() { flush(); }

    mgb::Maybe<Blob> get(const std::string& category, const Blob& key) override {
        return m_impl->get(category, key);
    }

    void put(const std::string& category, const Blob& key, const Blob& value) override {
        return m_impl->put(category, key, value);
    }

    std::optional<size_t> clear() override {
        if (m_impl) {
            m_impl = std::make_unique<mgb::InFilePersistentCache>();
            if (m_path) {
                m_impl->dump_cache(m_path->c_str());
            }
        }
        return {};
    }

    bool valid() const override { return m_impl != nullptr; }

    void flush() override {
        if (m_impl && m_path) {
            m_impl->dump_cache(m_path->c_str());
        }
    }
};

std::shared_ptr<ExtendedPersistentCache> ExtendedPersistentCache::make_from_config(
        std::string type, std::unordered_map<std::string, std::string> args,
        std::string& err_msg) {
    try {
        if (type == "redis") {
            std::string prefix = args.at("prefix");
            std::optional<std::string> password = args.count("password")
                                                        ? args.at("password")
                                                        : std::optional<std::string>();
            auto cache = std::make_shared<RedisCache>(prefix, 100);
            if (args.count("unixsocket")) {
                std::string unixsocket = args.at("unixsocket");
                cache->connect(unixsocket, 0, password);
            } else {
                std::string ip = args.at("hostname");
                int port = atoi(args.at("port").c_str());
                std::optional<std::string> password =
                        args.count("password") ? args.at("password")
                                               : std::optional<std::string>();
                cache->connect(ip, port, password);
            }
            return cache;
        } else if (type == "in-file") {
            std::string path = args.at("path");
            auto cache = std::make_shared<ExtendedInFilePersistentCache>();
            cache->open(path);
            return cache;
        } else if (type == "in-memory") {
            auto cache = std::make_shared<ExtendedInFilePersistentCache>();
            cache->open();
            return cache;
        } else {
            mgb_assert(false, "persistent cache type %s unsupported", type.c_str());
        }
    } catch (const std::exception& exc) {
        err_msg = exc.what();
    } catch (...) {
        err_msg = "unknown exception";
    }
    return nullptr;
}

}  // namespace mgb::imperative::persistent_cache

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
