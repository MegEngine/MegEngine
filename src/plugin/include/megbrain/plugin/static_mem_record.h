/**
 * \file src/plugin/include/megbrain/plugin/static_mem_record.h
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
#include "megbrain/utils/metahelper.h"
#ifndef __IN_TEE_ENV__
namespace mgb {
namespace cg {

class StaticMemRecorder : public NonCopyableObj {
public:
    static StaticMemRecorder& Instance() {
        static StaticMemRecorder StaticMemRecorder;
        return StaticMemRecorder;
    }

    struct opr_record {
        size_t id, size;
        std::string name;
    };
    struct memory_chunk_record {
        size_t id, size_orig, time_begin, time_end, addr_begin,
                addr_end, overwrite_dest_id;
        bool is_overwrite;
        std::string owner_var_name;
    };

    void active() { m_is_record = true; }

    bool valid() { return m_is_record; }

    void clear_opr_seq() { m_opr_seq_recorder.clear(); }

    void regist_opr_seq(opr_record opr) { m_opr_seq_recorder.push_back(opr); }

    void clear_memory_chunk() { m_memory_chunk_recorder.clear(); }

    void regist_memory_chunk(memory_chunk_record mcr) {
        m_memory_chunk_recorder.push_back(mcr);
    }

    void regist_memory_chunk_owner_var_name(size_t id, std::string name) {
        m_memory_chunk_recorder.at(id).owner_var_name = name;
    }

    void regist_peak_mem_size(size_t size) { m_peak_mem_size = size; }

    const size_t& peak_mem_size() { return m_peak_mem_size; }

    void set_sum_mem_size(size_t size) { m_sum_mem_size = size; }

    const size_t& sum_mem_size() { return m_sum_mem_size; }

    const size_t& set_weight_chunk_id() {
        m_weight_chunk_id = m_memory_chunk_recorder.size();
        return m_weight_chunk_id;
    }

    const size_t& weight_chunk_id() { return m_weight_chunk_id; }

    void dump_svg(std::string svg_name);

    void show(std::string svg_name);

private:
    bool m_is_record = false;
    // All chunks after m_memory_chunk_recorder.at(m_weight_chunk_id) are
    // weights memory chunks
    size_t m_peak_mem_size, m_sum_mem_size, m_weight_chunk_id;
    std::vector<opr_record> m_opr_seq_recorder;
    std::vector<memory_chunk_record> m_memory_chunk_recorder;
    std::vector<std::vector<size_t>> get_chunk_construct(
            std::vector<size_t> opr_ids);
};
}  // namespace cg
}  // namespace mgb
#endif
