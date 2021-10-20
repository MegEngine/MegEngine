/**
 * \file imperative/src/impl/interpreter/stack_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/small_vector.h"

namespace mgb::imperative::interpreter::intl {

class StackSnapshot;

class StackManager : public NonCopyableObj {
public:
    class Node;
    class Guard;
    struct Frame;
    class Trace;

private:
    std::unique_ptr<Node> m_root = nullptr;
    Node* m_current = nullptr;
    SmallVector<uint64_t> m_trace_id_stack;
    uint64_t m_last_trace_id = 0;

public:
    StackManager();
    std::pair<Node*, uint64_t> enter(std::string name);
    void exit(std::string name);
    Trace dump();
    Node* current();
};

class StackManager::Node : public NonCopyableObj {
private:
    std::string m_name;
    std::unordered_map<std::string, std::unique_ptr<Node>> m_children;
    std::unordered_map<std::string, size_t> m_id_table;
    Node* m_parent = nullptr;
    int64_t m_depth = -1;
    uint64_t m_version = 0;
    explicit Node(std::string name, Node* parent) : m_name{name}, m_parent{parent} {
        if (parent) {
            m_depth = parent->m_depth + 1;
        }
    }

public:
    const std::string& name() const { return m_name; }
    Node* operator[](const std::string& name) {
        auto& child = m_children[name];
        if (child == nullptr) {
            child.reset(new Node(name, this));
        }
        return child.get();
    }
    Node* parent() { return m_parent; }
    bool is_root() { return m_parent == nullptr; }
    uint64_t version() const { return m_version; }
    void update_version() {
        ++m_version;
        for (auto&& [key, child] : m_children) {
            child->reset_version();
        }
        m_id_table.clear();
    }
    void reset_version() {
        m_version = 0;
        m_id_table.clear();
    }
    int64_t depth() const { return m_depth; }
    uint64_t next_id(std::string key) { return m_id_table[key]++; }
    static std::unique_ptr<Node> make() {
        return std::unique_ptr<Node>(new Node("", nullptr));
    }
};

class StackManager::Guard {
private:
    std::string m_name;
    StackManager* m_manager;

public:
    Guard(std::string name, StackManager* manager) : m_name{name}, m_manager{manager} {
        if (m_manager) {
            m_manager->enter(name);
        }
    }
    ~Guard() { release(); }
    void release() {
        if (m_manager) {
            m_manager->exit(m_name);
            m_manager = nullptr;
        }
    }
};

struct StackManager::Frame {
    StackManager::Node* node;
    uint64_t version;
};

class StackManager::Trace {
private:
    SmallVector<StackManager::Frame> m_frames;
    uint64_t m_id = 0;

public:
    explicit Trace(StackManager::Node* top, uint64_t id) : m_id{id} {
        int64_t nr_frames = top->depth() + 1;
        m_frames = SmallVector<StackManager::Frame>(nr_frames);
        StackManager::Node* node = top;
        for (int64_t i = 0; i < nr_frames; ++i) {
            m_frames[m_frames.size() - 1 - i] = {node, node->version()};
            node = node->parent();
        }
        mgb_assert(node->is_root(), "");
    }
    Trace() = default;
    std::string to_string() const {
        std::string buffer;
        for (auto&& [node, version] : m_frames) {
            if (!buffer.empty()) {
                buffer.append(".");
            }
            buffer.append(node->name());
            if (version != 0) {
                buffer.append(ssprintf("[%zu]", version));
            }
        }
        return buffer;
    }
    const SmallVector<StackManager::Frame>& frames() const { return m_frames; }
    uint64_t id() const { return m_id; }
};

inline StackManager::StackManager() {
    m_root = Node::make();
    m_current = m_root.get();
}

inline std::pair<StackManager::Node*, uint64_t> StackManager::enter(std::string name) {
    m_current = (*m_current)[name];
    m_trace_id_stack.push_back(++m_last_trace_id);
    return {m_current, m_current->version()};
}

inline void StackManager::exit(std::string name) {
    mgb_assert(m_current->name() == name, "scope name mismatch");
    m_current = m_current->parent();
    m_trace_id_stack.pop_back();
    m_current->update_version();
}

inline StackManager::Trace StackManager::dump() {
    return Trace(m_current, m_trace_id_stack.empty() ? 0 : m_trace_id_stack.back());
}

inline StackManager::Node* StackManager::current() {
    return m_current;
}

}  // namespace mgb::imperative::interpreter::intl
