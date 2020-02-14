# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pytest
from helpers import modified_environ

from megengine.hub import hub


@pytest.mark.internet
def test_hub_http_basic(tmp_path):
    # Override XDG_CACHE_HOME to make sure test won't have side effect for system.
    with modified_environ(XDG_CACHE_HOME=str(tmp_path)):
        # Use pytorch's URL due to we don't have public address now.
        repo_info, entry = "pytorch/vision:v0.4.2", "alexnet"

        assert len(hub.list(repo_info)) > 0

        assert entry in hub.list(repo_info)

        assert hub.help(repo_info, entry)

        assert isinstance(hub.load(repo_info, entry), object)


@pytest.mark.internet
def test_github_load_with_commit_id(tmp_path):
    # Override XDG_CACHE_HOME to make sure test won't have side effect for system.
    with modified_environ(XDG_CACHE_HOME=str(tmp_path)):
        # Use pytorch's URL due to we don't have public address now.
        repo_info, commit, entry = "pytorch/vision", "d2c763e1", "alexnet"

        assert isinstance(hub.load(repo_info, entry, commit=commit), object)
