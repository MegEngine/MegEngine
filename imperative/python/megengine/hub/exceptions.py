# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
class FetcherError(Exception):
    r"""Base class for fetch related error."""


class InvalidRepo(FetcherError):
    r"""The repo provided was somehow invalid."""


class InvalidGitHost(FetcherError):
    r"""The git host provided was somehow invalid."""


class GitPullError(FetcherError):
    r"""A git pull error occurred."""


class GitCheckoutError(FetcherError):
    r"""A git checkout error occurred."""


class InvalidProtocol(FetcherError):
    r"""The protocol provided was somehow invalid."""
