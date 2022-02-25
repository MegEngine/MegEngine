# -*- coding: utf-8 -*-
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
