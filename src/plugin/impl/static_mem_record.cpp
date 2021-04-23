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
#include <fstream>
#include <iostream>

using namespace mgb;
using namespace cg;

namespace {
#define SVG_WIDTH 20000.0
#define SVG_HEIGHT 15000.0
#define OPR_RECT_WIDTH 40.0
#define OPR_RECT_HEIGHT 20.0

const std::string rect =
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" "
        " {}></rect>";
const std::string text = "<text x=\"{}\" y=\"{}\" font-size=\"{}\">{}</text>";
const std::string polyline =
        "<polyline points=\"{}\" style=\"fill:none;stroke:{};stroke-width:{}\" "
        "/>";
const std::string opr_info =
        "mge:type=\"opr\" mge:id=\"{}\" mge:size=\"{}\" mge:name=\"{}\"";
const std::string chunk_info =
        "mge:type=\"chunk\" mge:id=\"{}\" mge:time=\"{}\" mge:addr=\"{}\" "
        "mge:size=\"{}\" mge:owner_var_name=\"{}\"";
const std::string animate =
        "<animate attributeName=\"opacity\" from=\"0\" to=\"1\" "
        "begin=\"{}.mouseover\" fill=\"freeze\" dur=\"1s\"/>\n<animate "
        "attributeName=\"opacity\" from=\"1\" to=\"0\" begin=\"{}.mouseout\" "
        "fill=\"freeze\" dur=\"1s\"/>";

std::string& replace_by_parameter(std::string& original_str, size_t index) {
    return original_str;
}

template <typename... Args>
std::string& replace_by_parameter(std::string& original_str, size_t index,
                                  const std::string& parameter,
                                  const Args&... args) {
    index = original_str.find("{}", index);
    original_str.replace(index, 2, parameter);
    index += parameter.length();
    replace_by_parameter(original_str, index, args...);
    return original_str;
}

std::string set_opr_info(std::string id, std::string size, std::string name,
                         std::string info = opr_info) {
    return replace_by_parameter(info, 0, id, size, name);
}

std::string set_chunk_info(std::string id, std::string time, std::string addr,
                           std::string size, std::string owner_var_name,
                           std::string info = chunk_info) {
    return replace_by_parameter(info, 0, id, time, addr, size, owner_var_name);
}

std::string draw_rect(std::string x, std::string y, std::string widith,
                      std::string height, std::string color, std::string info,
                      std::string r = rect) {
    return replace_by_parameter(r, 0, x, y, widith, height, color, info);
}

std::string draw_text(std::string x, std::string y, std::string font_size,
                      std::string txt, std::string t = text) {
    return replace_by_parameter(t, 0, x, y, font_size, txt);
}

std::string draw_polyline(std::string point_seq, std::string color,
                          std::string width, std::string p = polyline) {
    return replace_by_parameter(p, 0, point_seq, color, width);
}
}  // namespace

void StaticMemRecorder::dump_svg(std::string svg_name) {
    float svg_width = SVG_WIDTH, svg_height = SVG_HEIGHT,
          opr_rect_width = OPR_RECT_WIDTH, opr_rect_height = OPR_RECT_HEIGHT;
    float address_scale = 1;
    size_t opr_nr = m_opr_seq_recorder.size();
    if (opr_nr * OPR_RECT_WIDTH > SVG_WIDTH) {
        svg_width = SVG_WIDTH;
        opr_rect_width = svg_width / opr_nr;
        opr_rect_height = opr_rect_width / 2;
    } else {
        opr_rect_width = OPR_RECT_WIDTH;
        svg_width = opr_nr * opr_rect_width;
    }
    if (m_sum_mem_size > SVG_HEIGHT) {
        svg_height = SVG_HEIGHT;
        address_scale = svg_height / m_sum_mem_size;
    } else {
        svg_height = m_sum_mem_size;
    }

    // Rescale
    float aspect_ratio = SVG_WIDTH / SVG_HEIGHT;
    if (svg_width / svg_height < 1) {
        svg_width = svg_height * aspect_ratio;
        opr_rect_width = svg_width / opr_nr;
        opr_rect_height = opr_rect_width / 2;
    } else if (svg_width / svg_height > aspect_ratio) {
        svg_height = svg_width / aspect_ratio;
        address_scale = svg_height / m_sum_mem_size;
    }

    svg_height = svg_height + opr_rect_height * 2;

    std::ofstream outfile;
    outfile.open(svg_name);
    outfile << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl;
    outfile << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN/\" "
               "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">"
            << std::endl;
    outfile << "<svg width=\"" + std::to_string(svg_width) + "\" height=\"" +
                       std::to_string(svg_height) +
                       "\" version=\"1.1\" "
                       "xmlns=\"http://www.w3.org/2000/svg\">"
            << std::endl;

    float base_height = svg_height - opr_rect_height;
    std::string peak_mem_polyline =
            "0," +
            std::to_string(base_height - m_peak_mem_size * address_scale) +
            " " + std::to_string(m_opr_seq_recorder.size() * opr_rect_width) +
            "," + std::to_string(base_height - m_peak_mem_size * address_scale);
    std::string sum_mem_polyline =
            "0," +
            std::to_string(base_height - m_sum_mem_size * address_scale) + " " +
            std::to_string(m_opr_seq_recorder.size() * opr_rect_width) + "," +
            std::to_string(base_height - m_sum_mem_size * address_scale);
    std::string memory_polyline = "";
    for (size_t i = 0; i < m_opr_seq_recorder.size(); i++) {
        auto&& opr = m_opr_seq_recorder.at(i);
        memory_polyline +=
                std::to_string((i + 0.5) * opr_rect_width) + "," +
                std::to_string(base_height - opr.size * address_scale) + " ";

        outfile << draw_text(std::to_string(i * opr_rect_width),
                             std::to_string(svg_height - opr_rect_height * 0.5),
                             std::to_string(opr_rect_height * 0.5),
                             "opr" + std::to_string(i))
                << std::endl;
        std::string opr_info =
                set_opr_info(
                        std::to_string(opr.id),
                        std::to_string(opr.size) + "B(" +
                                std::to_string(opr.size / 1024.0 / 1024.0) +
                                "MiB)",
                        opr.name) +
                " opacity=\"0\"";
        outfile << draw_rect(std::to_string(i * opr_rect_width),
                             std::to_string(base_height),
                             std::to_string(opr_rect_width),
                             std::to_string(opr_rect_height), "white", opr_info)
                << std::endl;
    }

    for (size_t i = 0; i < m_memory_chunk_recorder.size(); i++) {
        auto&& chunk = m_memory_chunk_recorder.at(i);
        std::string chunk_info = set_chunk_info(
                std::to_string(chunk.id),
                "[" + std::to_string(chunk.time_begin) + "," +
                        std::to_string(chunk.time_end) + ")",
                "[" + std::to_string(chunk.addr_begin) + "," +
                        std::to_string(chunk.addr_end) + ")",
                std::to_string(chunk.addr_end - chunk.addr_begin) + "B(" +
                        std::to_string((chunk.addr_end - chunk.addr_begin) /
                                       1024.0 / 1024.0) +
                        "MiB)",
                chunk.owner_var_name);

        outfile << draw_rect(
                           std::to_string(chunk.time_begin * opr_rect_width),
                           std::to_string(base_height -
                                          chunk.addr_end * address_scale),
                           std::to_string((chunk.time_end - chunk.time_begin) *
                                          opr_rect_width),
                           std::to_string((chunk.addr_end - chunk.addr_begin) *
                                          address_scale),
                           "gray", chunk_info)
                << std::endl;
        outfile << draw_text(std::to_string(chunk.time_begin * opr_rect_width),
                             std::to_string(base_height -
                                            chunk.addr_end * address_scale + 9),
                             std::to_string(9),
                             "chunk" + std::to_string(chunk.id))
                << std::endl;
    }

    outfile << draw_text("0",
                         std::to_string(base_height -
                                        m_peak_mem_size * address_scale +
                                        opr_rect_height * 0.5),
                         std::to_string(opr_rect_height * 0.5),
                         "peak_memory_size:" + std::to_string(m_peak_mem_size) +
                                 "B(" +
                                 std::to_string(m_peak_mem_size / 1024.0 /
                                                1024.0) +
                                 "MiB)")
            << std::endl;
    outfile << draw_text("0",
                         std::to_string(base_height -
                                        m_sum_mem_size * address_scale +
                                        opr_rect_height * 0.5),
                         std::to_string(opr_rect_height * 0.5),
                         "sum_memory_size:" + std::to_string(m_sum_mem_size) +
                                 "B(" +
                                 std::to_string(m_sum_mem_size / 1024.0 /
                                                1024.0) +
                                 "MiB)")
            << std::endl;
    outfile << draw_polyline(memory_polyline, "blue",
                             std::to_string(opr_rect_height * 0.1))
            << std::endl;
    outfile << draw_polyline(peak_mem_polyline, "green",
                             std::to_string(opr_rect_height * 0.1))
            << std::endl;
    outfile << draw_polyline(sum_mem_polyline, "red",
                             std::to_string(opr_rect_height * 0.1))
            << std::endl;
    outfile << "<text svg:info=\"The abscissa represents the opr sequence, the "
               "ordinate represents the logical address.\" "
               "svg:chunk_time=\"[opra,oprb) means the chunk is created when "
               "opra execute and is freed before oprb\" "
               "svg:chunk_oner_var_name=\"var that first creates this "
               "chunk\"></text>"
            << std::endl;
    outfile << "</svg>" << std::endl;
    outfile.close();
}

void StaticMemRecorder::show(std::string svg_name) {
    for (auto&& i : m_memory_chunk_recorder) {
        if (i.id >= m_weight_chunk_id) {
            break;
        }
        size_t begin = i.time_begin, end = i.time_end;
        if (i.is_overwrite) {
            begin++;
        }
        for (size_t j = begin; j < end; j++) {
            m_opr_seq_recorder.at(j).size += i.size_orig;
        }
    }

    // log peak memory size, where it is reached and which chunks constitute it.
    mgb_log("peak_mem_size = %zu\n", m_peak_mem_size);
    size_t max_size = 0;
    std::vector<size_t> opr_ids;
    for (auto&& i : m_opr_seq_recorder) {
        if (i.size == max_size) {
            opr_ids.push_back(i.id);
        } else if (i.size > max_size) {
            max_size = i.size;
            opr_ids.clear();
            opr_ids.push_back(i.id);
        }
    }

    auto opr2chunk = get_chunk_construct(opr_ids);
    mgb_log("oprs reach the peak memory:\n");
    for (auto&& i : opr_ids) {
        mgb_log("opr id = %zu\n", i);
    }
    mgb_log("More details:\n");
    for (size_t i = 0; i < opr2chunk.size(); i++) {
        mgb_log("opr id = %zu\n", opr_ids.at(i));
        if (i + 1 < opr2chunk.size() &&
            opr2chunk.at(i) == opr2chunk.at(i + 1)) {
            continue;
        }
        for (size_t j = 0; j < opr2chunk.at(i).size(); j++) {
            auto&& chunk = m_memory_chunk_recorder.at(opr2chunk.at(i).at(j));
            mgb_log("[memory_chunk_id=%zu, size=%zu B,  "
                    "[life_begin=%zu,life_end=%zu),  owner_opr_name=%s]\n",
                    chunk.id, chunk.size_orig, chunk.time_begin, chunk.time_end,
                    m_opr_seq_recorder.at(chunk.time_begin).name.c_str());
        }
    }
    dump_svg(svg_name);
}

std::vector<std::vector<size_t>> StaticMemRecorder::get_chunk_construct(
        std::vector<size_t> opr_ids) {
    std::vector<std::vector<size_t>> chunk_ids;
    chunk_ids.resize(opr_ids.size());
    for (auto&& i : m_memory_chunk_recorder) {
        if (i.id >= m_weight_chunk_id) {
            break;
        }
        size_t begin = i.time_begin, end = i.time_end;
        if (i.is_overwrite) {
            begin = begin + 1;
        }
        if (opr_ids.front() >= end || opr_ids.back() < begin) {
            continue;
        }
        for (size_t k = 0; k < opr_ids.size(); k++) {
            if (opr_ids.at(k) >= end) {
                break;
            } else if (opr_ids.at(k) >= begin) {
                chunk_ids.at(k).push_back(i.id);
            }
        }
    }
    return chunk_ids;
}