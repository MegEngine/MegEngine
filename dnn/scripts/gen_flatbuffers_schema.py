#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import hashlib
import io
import os
import struct
import textwrap

from gen_param_defs import IndentWriterBase, ParamDef, member_defs


def _cname_to_fbname(cname):
    return {
        "uint32_t": "uint",
        "uint64_t": "ulong",
        "int32_t": "int",
        "float": "float",
        "double": "double",
        "DTypeEnum": "DTypeEnum",
        "bool": "bool",
    }[cname]


def scramble_enum_member_name(name):
    s = name.find("<<")
    if s != -1:
        name = name[0 : name.find("=") + 1] + " " + name[s + 2 :]
    if name in ("MIN", "MAX"):
        return name + "_"
    o_name = name.split(" ")[0].split("=")[0]
    if o_name in ("MIN", "MAX"):
        return name.replace(o_name, o_name + "_")
    return name


class FlatBuffersWriter(IndentWriterBase):
    _skip_current_param = False
    _last_param = None
    _enums = None
    _used_enum = None
    _cur_const_val = {}

    def __call__(self, fout, defs):
        param_io = io.StringIO()
        super().__call__(param_io)
        self._used_enum = set()
        self._enums = {}
        self._process(defs)
        super().__call__(fout)
        self._write("// %s", self._get_header())
        self._write('include "dtype.fbs";')
        self._write("namespace mgb.serialization.fbs.param;\n")
        self._write_enums()
        self._write(param_io.getvalue())

    def _write_enums(self):
        for (p, e) in sorted(self._used_enum):
            name = p + e
            e = self._enums[(p, e)]
            self._write_doc(e.name)
            attribute = "(bit_flags)" if e.combined else ""
            self._write("enum %s%s : uint %s {", p, e.name, attribute, indent=1)
            for idx, member in enumerate(e.members):
                self._write_doc(member)
                self._write("%s,", scramble_enum_member_name(str(member)))
            self._write("}\n", indent=-1)

    def _write_doc(self, doc):
        if not isinstance(doc, member_defs.Doc) or not doc.doc:
            return
        doc_lines = []
        if doc.no_reformat:
            doc_lines = doc.raw_lines
        else:
            doc = doc.doc.replace("\n", " ")
            text_width = 80 - len(self._cur_indent) - 4
            doc_lines = textwrap.wrap(doc, text_width)
        for line in doc_lines:
            self._write("/// " + line)

    def _on_param_begin(self, p):
        self._last_param = p
        self._cur_const_val = {}
        self._write_doc(p.name)
        self._write("table %s {", p.name, indent=1)

    def _on_param_end(self, p):
        if self._skip_current_param:
            self._skip_current_param = False
            return
        self._write("}\n", indent=-1)

    def _on_member_enum(self, e):
        p = self._last_param
        key = str(p.name), str(e.name)
        self._enums[key] = e
        if self._skip_current_param:
            return
        self._write_doc(e.name)
        self._used_enum.add(key)
        if e.combined:
            default = e.compose_combined_enum(e.default)
        else:
            default = scramble_enum_member_name(
                str(e.members[e.default]).split(" ")[0].split("=")[0]
            )
        self._write("%s:%s%s = %s;", e.name_field, p.name, e.name, default)

    def _resolve_const(self, v):
        while v in self._cur_const_val:
            v = self._cur_const_val[v]
        return v

    def _on_member_field(self, f):
        if self._skip_current_param:
            return
        self._write_doc(f.name)
        self._write(
            "%s:%s = %s;",
            f.name,
            _cname_to_fbname(f.dtype.cname),
            self._get_fb_default(self._resolve_const(f.default)),
        )

    def _on_const_field(self, f):
        self._cur_const_val[str(f.name)] = str(f.default)

    def _on_member_enum_alias(self, e):
        if self._skip_current_param:
            return
        self._used_enum.add((e.src_class, e.src_name))
        enum_name = e.src_class + e.src_name
        s = e.src_enum
        if s.combined:
            default = s.compose_combined_enum(e.get_default())
        else:
            default = scramble_enum_member_name(
                str(s.members[e.get_default()]).split(" ")[0].split("=")[0]
            )
        self._write("%s:%s = %s;", e.name_field, enum_name, default)

    def _get_fb_default(self, cppdefault):
        if not isinstance(cppdefault, str):
            return cppdefault

        d = cppdefault
        if d.endswith("f"):  # 1.f
            return d[:-1]
        if d.endswith("ull"):
            return d[:-3]
        if d.startswith("DTypeEnum::"):
            return d[11:]
        return d


def main():
    parser = argparse.ArgumentParser(
        "generate FlatBuffers schema of operator param from description file"
    )
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.input) as fin:
        inputs = fin.read()
        exec(inputs, {"pdef": ParamDef, "Doc": member_defs.Doc})
        input_hash = hashlib.sha256()
        input_hash.update(inputs.encode(encoding="UTF-8"))
        input_hash = input_hash.hexdigest()

    writer = FlatBuffersWriter()
    with open(args.output, "w") as fout:
        writer.set_input_hash(input_hash)(fout, ParamDef.all_param_defs)


if __name__ == "__main__":
    main()
