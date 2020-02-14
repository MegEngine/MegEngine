#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import textwrap
import os
import hashlib
import struct
import io

from gen_param_defs import member_defs, ParamDef, IndentWriterBase

class ConverterWriter(IndentWriterBase):
    _skip_current_param = False
    _last_param = None
    _param_fields = None
    _fb_fields = []

    def __call__(self, fout, defs):
        super().__call__(fout)
        self._write("// %s", self._get_header())
        self._write('#include <flatbuffers/flatbuffers.h>')
        self._write("namespace mgb {")
        self._write("namespace serialization {")
        self._write("namespace fbs {")
        self._process(defs)
        self._write("}  // namespace fbs")
        self._write("}  // namespace serialization")
        self._write("}  // namespace mgb")

    def _on_param_begin(self, p):
        self._last_param = p
        self._param_fields = []
        self._fb_fields = ["builder"]
        if p.is_legacy:
            self._skip_current_param = True
            return
        self._write("template<>\nstruct ParamConverter<megdnn::param::%s> {",
                    p.name, indent=1)
        self._write("using MegDNNType = megdnn::param::%s;", p.name)
        self._write("using FlatBufferType = fbs::param::%s;\n", p.name)

    def _on_param_end(self, p):
        if self._skip_current_param:
            self._skip_current_param = False
            return
        self._write("static MegDNNType to_param(const FlatBufferType* fb) {",
                    indent=1)
        line = 'return {'
        line += ', '.join(self._param_fields)
        line += '};'
        self._write(line)
        self._write("}\n", indent=-1)

        self._write(
            "static flatbuffers::Offset<FlatBufferType> to_flatbuffer(flatbuffers::FlatBufferBuilder& builder, const MegDNNType& param) {",
            indent=1)
        line = 'return fbs::param::Create{}('.format(str(p.name))
        line += ', '.join(self._fb_fields)
        line += ');'
        self._write(line)
        self._write('}', indent=-1)

        self._write("};\n", indent=-1)

    def _on_member_enum(self, e):
        p = self._last_param
        key = str(p.name) + str(e.name)
        if self._skip_current_param:
            return
        self._param_fields.append(
            "static_cast<megdnn::param::{}::{}>(fb->{}())".format(
                str(p.name), str(e.name), e.name_field))
        self._fb_fields.append("static_cast<fbs::param::{}>(param.{})".format(
            key, e.name_field))

    def _on_member_field(self, f):
        if self._skip_current_param:
            return
        if f.dtype.cname == 'DTypeEnum':
            self._param_fields.append(
                "intl::convert_dtype_to_megdnn(fb->{}())".format(f.name))
            self._fb_fields.append(
                "intl::convert_dtype_to_fbs(param.{})".format(f.name))
        else:
            self._param_fields.append("fb->{}()".format(f.name))
            self._fb_fields.append("param.{}".format(f.name))

    def _on_const_field(self, f):
        pass

    def _on_member_enum_alias(self, e):
        if self._skip_current_param:
            return
        enum_name = e.src_class + e.src_name
        self._param_fields.append(
            "static_cast<megdnn::param::{}::{}>(fb->{}())".format(
                e.src_class, e.src_name, e.name_field))
        self._fb_fields.append("static_cast<fbs::param::{}>(param.{})".format(
            enum_name, e.name_field))


def main():
    parser = argparse.ArgumentParser(
        'generate convert functions between FlatBuffers type and MegBrain type')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(args.input) as fin:
        inputs = fin.read()
        exec(inputs, {'pdef': ParamDef, 'Doc': member_defs.Doc})
        input_hash = hashlib.sha256()
        input_hash.update(inputs.encode(encoding='UTF-8'))
        input_hash = input_hash.hexdigest()

    writer = ConverterWriter()
    with open(args.output, 'w') as fout:
        writer.set_input_hash(input_hash)(fout, ParamDef.all_param_defs)

if __name__ == "__main__":
    main()
