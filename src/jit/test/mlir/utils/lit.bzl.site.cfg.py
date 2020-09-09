# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Lit runner site configuration."""
import os
import lit.llvm

config.llvm_tools_dir = os.path.join(os.environ['TEST_SRCDIR'], 'llvm-project', 'llvm')
config.mlir_obj_root = os.path.join(os.environ['TEST_SRCDIR'])
config.mlir_tools_dir = os.path.join(os.environ['TEST_SRCDIR'], 'llvm-project', 'mlir')
config.suffixes = ['.td', '.mlir', '.pbtxt']

mlir_mgb_tools_dirs = [
    'brain/megbrain/tools/mlir',
]
config.mlir_mgb_tools_dirs = [
    os.path.join(os.environ['TEST_SRCDIR'], os.environ['TEST_WORKSPACE'], s)
    for s in mlir_mgb_tools_dirs
]
test_dir = os.environ['TEST_TARGET']
test_dir = test_dir.strip('/').rsplit(':', 1)[0]
config.mlir_test_dir = os.path.join(
    os.environ['TEST_SRCDIR'],
    os.environ['TEST_WORKSPACE'],
    test_dir,
)
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(
    config,
    os.path.join(
        os.path.join(
            os.environ['TEST_SRCDIR'],
            os.environ['TEST_WORKSPACE'],
            'brain/megbrain/src/jit/test/mlir/utils/lit.bzl.cfg.py',
        )))
