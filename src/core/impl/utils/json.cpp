/**
 * \file src/core/impl/utils/json.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/json.h"
#include "megbrain/utils/thread.h"
#include "megbrain/utils/debug.h"

#if MGB_ENABLE_JSON

#include <limits>
#include <cstring>
#include <cerrno>

using namespace mgb::json;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Number);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(NumberInt);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Bool);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(String);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Object);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Array);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Null);

namespace {

void write_indent(std::string& fout, int indent) {
    mgb_assert(0 <= indent, "indent must be greater or equal to zero");
    while (indent--) {
        fout.append(1, ' ');
    }
}

} // anonymous namespace

std::string Value::to_string(int indent) const {
    std::string ostr;
    writeto(ostr, indent);
    return ostr;
}

void Value::writeto_fpath(const char* fout_path, int indent) const {
    auto str = to_string(indent);
    debug::write_to_file(fout_path, str);
}

void Number::writeto(std::string& fout, int) const {
    static char fmt[16];
    static Spinlock fmt_mtx;
    if (!fmt[sizeof(fmt) - 1]) {
        MGB_LOCK_GUARD(fmt_mtx);
        if (!fmt[sizeof(fmt) - 1]) {
            snprintf(fmt, sizeof(fmt) - 1, "%%.%dg", static_cast<int>(
                        std::numeric_limits<decltype(m_val)>::digits10));
            fmt[sizeof(fmt) - 1] = 1;
        }
    }
    char val[64];
    snprintf(val, sizeof(val), fmt, m_val);
    fout += val;
}

void NumberInt::writeto(std::string& fout, int) const {
    fout += std::to_string(m_val);
}

void Bool::writeto(std::string &fout, int) const {
    fout += (m_val ? "true" : "false");
}

std::shared_ptr<Bool> Bool::make(bool v) {
    static auto vtrue = std::make_shared<Bool>(true),
                vfalse = std::make_shared<Bool>(false);
    return v ? vtrue : vfalse;
}

void String::writeto(std::string &fout, int) const  {
    fout += '"';
    for (char ch: m_val) {
        switch (ch) {
            case '"':
                fout += "\\\"";
                break;
            case '\\':
                fout += "\\\\";
                break;
            case '/':
                fout += "\\/";
                break;
            case '\b':
                fout += "\\b";
                break;
            case '\f':
                fout += "\\f";
                break;
            case '\n':
                fout += "\\n";
                break;
            case '\r':
                fout += "\\r";
                break;
            case '\t':
                fout += "\\t";
                break;
            default:
                mgb_assert(ch >= 1);
                fout += ch;
        }
    }
    fout += '"';
}

void Object::writeto(std::string &fout, int indent) const {
    char linebreak;
    if (indent) {
        linebreak = '\n';
        ++ indent;
    } else {
        linebreak = ' ';
    }
    fout.append("{").append(1, linebreak);
    bool first = true;
    for (auto &&i: m_val) {
        if (first) {
            first = false;
        } else {
            fout.append(",").append(1, linebreak);
        }
        write_indent(fout, indent);
        i.first.writeto(fout, indent);
        fout += ": ";
        i.second->writeto(fout, indent);
    }
    if (indent) {
        fout.append(1, linebreak);
        write_indent(fout, --indent);
    }
    fout += '}';
}

void Array::writeto(std::string &fout, int indent) const {
    char linebreak;
    if (indent) {
        linebreak = '\n';
        ++ indent;
    } else {
        linebreak = ' ';
    }
    fout += "[";
    bool first = true;
    for (auto &&i: m_val) {
        if (first) {
            first = false;
        } else {
            fout += ",";
        }
        fout.append(1, linebreak);
        write_indent(fout, indent);
        i->writeto(fout, indent);
    }
    if (!first && indent) {
        fout.append(1, linebreak);
        write_indent(fout, --indent);
    }
    fout += ']';
}

void Null::writeto(std::string &fout, int /*indent*/) const {
    fout += "null";
}

#endif // MGB_ENABLE_JSON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

