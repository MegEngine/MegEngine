#!/usr/bin/python
# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import json
import codecs

def generate_code_file():
    # read config
    config = {}
    with codecs.open("config.json","rb","UTF-8") as f:
        config = json.loads(f.read())
    if not config:
        return

    # read template file
    s = ""
    template = config.get("templateFile")
    with codecs.open(template, "rb", "UTF-8") as f:
        s = f.read()
    if not s:
        return
    s = s % config

    # save to file
    fn = config["fileNamePrefix"]
    with codecs.open(fn, "wb", "UTF-8") as f:
        f.write(s)
        f.flush()

def generate_a_batch_of_code_file():
    # read config
    config = {}
    with codecs.open("config.json","rb","UTF-8") as f:
        config = json.loads(f.read())
    if not config:
        return

    # read template file
    s_template = ""
    template = config.get("templateFile")
    with codecs.open(template, "rb", "UTF-8") as f:
        s_template = f.read()
    if not s_template:
        return

    for i in range(1, 8):
        config["kernelSize"] = str(i)
        s = s_template % config

        # save to file
        fn = config["fileNamePrefix"] + "_" +\
             config["nonlineTypeLower"] +\
             "_ksize" + str(i) + ".cu"

        print('generating {}...'.format(fn))

        with codecs.open(fn, "wb", "UTF-8") as f:
            f.write(s)
            f.flush()
if __name__ == '__main__':
    generate_a_batch_of_code_file()
    try:
        generate_code_file()
    except Exception, ex:
        print(ex)
