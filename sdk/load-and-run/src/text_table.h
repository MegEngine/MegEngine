/**
 * \file sdk/load-and-run/src/text_table.h
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

#include <array>
#include <iomanip>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include "megbrain/common.h"

namespace mgb
{

class TextTable {
public:
    enum Level { Summary, Detail };
    enum class Align : int { Left, Right, Mid };
    explicit TextTable(const std::string& table_name) : m_name(table_name) {}
    TextTable& horizontal(char c) {
        m_row.params.horizontal = c;
        return *this;
    }
    TextTable& vertical(char c) {
        m_row.params.vertical = c;
        return *this;
    }
    TextTable& corner(char c) {
        m_row.params.corner = c;
        return *this;
    }
    TextTable& align(Align v) {
        m_row.params.align = v;
        return *this;
    }
    TextTable& padding(size_t w) {
        m_padding = w;
        return *this;
    }
    TextTable& prefix(const std::string& str) {
        m_prefix = str;
        return *this;
    }

    template <typename T>
    TextTable& add(const T& value) {
        if constexpr (std::is_floating_point<T>::value) {
            std::stringstream ss;
            ss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
            ss << value;
            m_row.values.emplace_back(ss.str());
        } else if constexpr (std::is_integral<T>::value) {
            m_row.values.emplace_back(std::to_string(value));
        } else {
            m_row.values.emplace_back(value);
        }
        if (m_cols_max_w.size() < m_row.values.size()) {
            m_cols_max_w.emplace_back(m_row.values.back().length());
        } else {
            mgb_assert(m_row.values.size() >= 1);
            size_t i = m_row.values.size() - 1;
            m_cols_max_w[i] =
                std::max(m_cols_max_w[i], m_row.values.back().length());
        }
        return *this;
    }

    void eor() {
        m_rows.emplace_back(m_row);
        adjuster_last_row();
        m_row.values.clear();
    }

    void reset() {
        m_row = {};
        m_cols_max_w.clear();
        m_padding = 0;
        m_rows.clear();
    }

    void show(std::ostream& os);

private:
    void adjuster_last_row();
    std::string m_name;
    std::vector<size_t> m_cols_max_w;
    size_t m_padding = 0;
    std::string m_prefix = "";
    struct Row {
        std::vector<std::string> values;
        struct Params {
            Align align = Align::Left;
            char horizontal = '-', vertical = '|', corner = '+';
        } params;
    };
    std::vector<Row> m_rows;
    Row m_row;
};   

inline std::ostream& operator<<(std::ostream& stream, TextTable& table) {
    table.show(stream);
    return stream;
}

} // namespace mgb