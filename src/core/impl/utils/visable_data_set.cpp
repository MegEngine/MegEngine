/**
 * \file src/core/impl/utils/tensorboard.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/utils/visable_data_set.h"
#if MGB_ENABLE_JSON
#include <fstream>
#include <iostream>

using namespace mgb;

#if WIN32
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#define getcwd _getcwd
namespace {

auto mkdir(const char* path, int) {
    return _mkdir(path);
}

}  // namespace
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {
#if defined(IOS)
#pragma message "build test on iOS; need ios_get_mgb_output_dir() to be defined"
extern "C" void ios_get_mgb_output_dir(char** dir);
#endif

std::string output_file(std::string dir_name, const std::string& fname) {
    static std::string cwd;
    static std::mutex cwd_mtx;
    MGB_LOCK_GUARD(cwd_mtx);
    if (cwd.empty()) {
#if defined(IOS)
        char* buf = nullptr;
        ios_get_mgb_output_dir(&buf);
#else
        auto buf = getcwd(nullptr, 0);
#endif
        mgb_assert(buf);
        cwd = buf;
        free(buf);
        dir_name = dir_name + "/";
        for (size_t i = 0; i < dir_name.size(); i++) {
            size_t pos = dir_name.find("/", i);
            if (pos < dir_name.size() && pos - i > 1) {
                cwd.append("/" + dir_name.substr(i, pos - i));
                mkdir(cwd.c_str(), 0755);
                i = pos;
            }
        }
    }
    if (fname.empty())
        return cwd;
    auto ret = cwd + "/" + fname;
    FILE* fout = fopen(ret.c_str(), "w");
    mgb_assert(fout, "failed to open %s: %s", ret.c_str(), strerror(errno));
    fclose(fout);

    return ret;
}
}  // namespace

void VisableDataSet::draw_graph(std::shared_ptr<json::Value> graph_json) {
    graph_json->writeto_fpath(output_file(m_logdir, "graph.json"));
}

void VisableDataSet::dump_info(Content& c) {
    auto&& content_set = m_file2content[c.file_name()];
    content_set.insert(c.content_name());
    auto&& value_list =
            m_filecontent2value[c.file_name() + "/" + c.content_name()];
    value_list.push_back(c.to_json());
}

void VisableDataSet::write_to_file() {
    for (auto& i : m_file2content) {
        auto f_objptr = json::Object::make();
        auto&& f_obj = *f_objptr;
        for (auto& c : i.second) {
            auto c_objptr = json::Object::make();
            auto&& c_obj = *c_objptr;
            for (auto& j : m_filecontent2value[i.first + "/" + c]) {
                c_obj[(*j).cast_final_safe<json::Object>()["id"]
                              ->cast_final_safe<json::String>()
                              .get_impl()] = j;
            }
            f_obj[c] = c_objptr;
        }
        f_objptr->writeto_fpath(output_file(m_logdir, i.first));
    }
}

std::shared_ptr<json::Value> Chunk::to_json() const {
    auto objptr = json::Object::make();
    auto&& obj = *objptr;
    obj["id"] = json::String::make(id());
    obj["type"] = json::String::make(m_type);
    obj["time_begin"] = json::String::make(m_time_begin);
    obj["time_end"] = json::String::make(m_time_end);
    obj["logic_addr_begin"] = json::String::make(m_logic_addr_begin);
    obj["logic_addr_end"] = json::String::make(m_logic_addr_end);
    obj["overwrite_dest_id"] = json::String::make(m_overwrite_dest_id);
    return objptr;
}

std::shared_ptr<json::Value> OprSeq::to_json() const {
    auto objptr = json::Object::make();
    auto&& obj = *objptr;
    obj["id"] = json::String::make(id());
    obj["name"] = json::String::make(m_name);
    return objptr;
}
#endif