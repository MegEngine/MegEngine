/**
 * \file src/plugin/impl/static_mem_record.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/plugin/static_mem_record.h"
#include "megbrain/utils/visable_data_set.h"
#ifndef __IN_TEE_ENV__
#if MGB_ENABLE_JSON
#include <fstream>
#include <iostream>

using namespace mgb;
using namespace cg;

void StaticMemRecorder::dump_to_json() {
    VisableDataSet writer(m_log_dir);
    for (auto&& i : m_memory_chunk_recorder) {
        // static mem chunk
        if (i.id < m_weight_chunk_id) {
            std::string overwrite_dest_id =
                    i.is_overwrite ? std::to_string(i.overwrite_dest_id) : "-1";
            Chunk c(std::to_string(i.id), Chunk::static_mem,
                    std::to_string(i.time_begin), std::to_string(i.time_end),
                    std::to_string(i.addr_begin), std::to_string(i.addr_end),
                    overwrite_dest_id);
            writer.dump_info(c);
        } else {
            // weight mem chunk
            Chunk c(std::to_string(i.id), Chunk::weight_mem,
                    std::to_string(i.time_begin), std::to_string(i.time_end),
                    std::to_string(i.addr_begin), std::to_string(i.addr_end), "-1");
            writer.dump_info(c);
        }
    }
    for (auto&& i : m_opr_seq_recorder) {
        OprSeq o(std::to_string(i.id), i.name);
        writer.dump_info(o);
    }
    writer.write_to_file();
}
#endif
#endif