# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MLIR_TEST'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = config.mlir_test_dir

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.environ['RUNFILES_DIR']

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = config.mlir_mgb_tools_dirs + [config.mlir_tools_dir, config.llvm_tools_dir]
tool_names = [
    'mgb-opt',
    'mlir-tblgen',
    'mlir-translate',
    'mgb-file-check',
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)
