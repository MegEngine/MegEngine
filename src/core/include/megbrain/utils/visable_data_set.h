/**
 * \file src/core/include/megbrain/utils/visable_data_set.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megbrain/utils/json.h"
#if MGB_ENABLE_JSON
#include <set>
#include "megbrain/plugin/base.h"
#include "megbrain/plugin/static_mem_record.h"

namespace mgb {
class VisableDataSet : public NonCopyableObj {
private:
    const std::string m_logdir;
    std::unordered_map<std::string, std::set<std::string>> m_file2content;
    std::unordered_map<std::string, std::vector<std::shared_ptr<json::Value>>>
            m_filecontent2value;

public:
    class Content {
    private:
        std::string m_file_name;
        std::string m_content_name;
        std::string m_id;

    public:
        Content(std::string file_name, std::string content_name, std::string id)
                : m_file_name(file_name), m_content_name(content_name), m_id(id) {}
        const std::string& file_name() const { return m_file_name; }
        const std::string& content_name() const { return m_content_name; }
        const std::string& id() const { return m_id; }
        virtual std::shared_ptr<json::Value> to_json() const = 0;
        virtual ~Content() = default;
    };
    VisableDataSet(std::string logdir) : m_logdir(logdir) {}

    void draw_graph(std::shared_ptr<json::Value> graph_json);

    void dump_info(Content& c);

    void write_to_file();
};

class Chunk : public VisableDataSet::Content {
private:
    const char* enum_str[2] = {"static_mem", "weight_mem"};
    std::string m_type, m_time_begin, m_time_end, m_logic_addr_begin, m_logic_addr_end,
            m_overwrite_dest_id;  // m_overwriter_dest_id = "-1" means no
                                  // overwrite dest
public:
    enum chunk_type { static_mem, weight_mem };

    Chunk(std::string id, chunk_type type, std::string time_begin, std::string time_end,
          std::string logic_addr_begin, std::string logic_addr_end,
          std::string overwrite_dest_id)
            : Content("StaticMemoryInfo.json", "chunk", id),
              m_type(enum_str[type]),
              m_time_begin(time_begin),
              m_time_end(time_end),
              m_logic_addr_begin(logic_addr_begin),
              m_logic_addr_end(logic_addr_end),
              m_overwrite_dest_id(overwrite_dest_id) {}
    std::shared_ptr<json::Value> to_json() const override;
};

class OprSeq : public VisableDataSet::Content {
private:
    std::string m_id, m_name;

public:
    OprSeq(std::string id, std::string opr_name)
            : Content("StaticMemoryInfo.json", "opr", id), m_name(opr_name) {}
    std::shared_ptr<json::Value> to_json() const override;
};
}  // namespace mgb
#endif