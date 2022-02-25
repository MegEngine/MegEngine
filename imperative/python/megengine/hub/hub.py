# -*- coding: utf-8 -*-
import functools
import hashlib
import os
import sys
import types
from typing import Any, List
from urllib.parse import urlparse

from megengine.utils.http_download import download_from_url

from ..distributed import is_distributed
from ..logger import get_logger
from ..serialization import load as _mge_load_serialized
from .const import (
    DEFAULT_CACHE_DIR,
    DEFAULT_GIT_HOST,
    DEFAULT_PROTOCOL,
    ENV_MGE_HOME,
    ENV_XDG_CACHE_HOME,
    HUBCONF,
    HUBDEPENDENCY,
)
from .exceptions import InvalidProtocol
from .fetcher import GitHTTPSFetcher, GitSSHFetcher
from .tools import cd, check_module_exists, load_module

logger = get_logger(__name__)


PROTOCOLS = {
    "HTTPS": GitHTTPSFetcher,
    "SSH": GitSSHFetcher,
}


def _get_megengine_home() -> str:
    r"""MGE_HOME setting complies with the XDG Base Directory Specification"""
    megengine_home = os.path.expanduser(
        os.getenv(
            ENV_MGE_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "megengine"),
        )
    )
    return megengine_home


def _get_repo(
    git_host: str,
    repo_info: str,
    use_cache: bool = False,
    commit: str = None,
    protocol: str = DEFAULT_PROTOCOL,
) -> str:
    if protocol not in PROTOCOLS:
        raise InvalidProtocol(
            "Invalid protocol, the value should be one of {}.".format(
                ", ".join(PROTOCOLS.keys())
            )
        )
    cache_dir = os.path.expanduser(os.path.join(_get_megengine_home(), "hub"))
    with cd(cache_dir):
        fetcher = PROTOCOLS[protocol]
        repo_dir = fetcher.fetch(git_host, repo_info, use_cache, commit)
        return os.path.join(cache_dir, repo_dir)


def _check_dependencies(module: types.ModuleType) -> None:
    if not hasattr(module, HUBDEPENDENCY):
        return

    dependencies = getattr(module, HUBDEPENDENCY)
    if not dependencies:
        return

    missing_deps = [m for m in dependencies if not check_module_exists(m)]
    if len(missing_deps):
        raise RuntimeError("Missing dependencies: {}".format(", ".join(missing_deps)))


def _init_hub(
    repo_info: str,
    git_host: str,
    use_cache: bool = True,
    commit: str = None,
    protocol: str = DEFAULT_PROTOCOL,
):
    r"""Imports hubmodule like python import.

    Args:
        repo_info: a string with format ``"repo_owner/repo_name[:tag_name/:branch_name]"`` with an optional
            tag/branch. The default branch is ``master`` if not specified. Eg: ``"brain_sdk/MegBrain[:hub]"``
        git_host: host address of git repo. Eg: github.com
        use_cache: whether to use locally cached code or completely re-fetch.
        commit: commit id on github or gitlab.
        protocol: which protocol to use to get the repo, and HTTPS protocol only supports public repo on github.
            The value should be one of HTTPS, SSH.

    Returns:
        a python module.
    """
    cache_dir = os.path.expanduser(os.path.join(_get_megengine_home(), "hub"))
    os.makedirs(cache_dir, exist_ok=True)
    absolute_repo_dir = _get_repo(
        git_host, repo_info, use_cache=use_cache, commit=commit, protocol=protocol
    )
    sys.path.insert(0, absolute_repo_dir)
    hubmodule = load_module(HUBCONF, os.path.join(absolute_repo_dir, HUBCONF))
    sys.path.remove(absolute_repo_dir)

    return hubmodule


@functools.wraps(_init_hub)
def import_module(*args, **kwargs):
    return _init_hub(*args, **kwargs)


def list(
    repo_info: str,
    git_host: str = DEFAULT_GIT_HOST,
    use_cache: bool = True,
    commit: str = None,
    protocol: str = DEFAULT_PROTOCOL,
) -> List[str]:
    r"""Lists all entrypoints available in repo hubconf.

    Args:
        repo_info: a string with format ``"repo_owner/repo_name[:tag_name/:branch_name]"`` with an optional
            tag/branch. The default branch is ``master`` if not specified. Eg: ``"brain_sdk/MegBrain[:hub]"``
        git_host: host address of git repo. Eg: github.com
        use_cache: whether to use locally cached code or completely re-fetch.
        commit: commit id on github or gitlab.
        protocol: which protocol to use to get the repo, and HTTPS protocol only supports public repo on github.
            The value should be one of HTTPS, SSH.

    Returns:
        all entrypoint names of the model.
    """
    hubmodule = _init_hub(repo_info, git_host, use_cache, commit, protocol)

    return [
        _
        for _ in dir(hubmodule)
        if not _.startswith("__") and callable(getattr(hubmodule, _))
    ]


def load(
    repo_info: str,
    entry: str,
    *args,
    git_host: str = DEFAULT_GIT_HOST,
    use_cache: bool = True,
    commit: str = None,
    protocol: str = DEFAULT_PROTOCOL,
    **kwargs
) -> Any:
    r"""Loads model from github or gitlab repo, with pretrained weights.

    Args:
        repo_info: a string with format ``"repo_owner/repo_name[:tag_name/:branch_name]"`` with an optional
            tag/branch. The default branch is ``master`` if not specified. Eg: ``"brain_sdk/MegBrain[:hub]"``
        entry: an entrypoint defined in hubconf.
        git_host: host address of git repo. Eg: github.com
        use_cache: whether to use locally cached code or completely re-fetch.
        commit: commit id on github or gitlab.
        protocol: which protocol to use to get the repo, and HTTPS protocol only supports public repo on github.
            The value should be one of HTTPS, SSH.

    Returns:
        a single model with corresponding pretrained weights.
    """
    hubmodule = _init_hub(repo_info, git_host, use_cache, commit, protocol)

    if not hasattr(hubmodule, entry) or not callable(getattr(hubmodule, entry)):
        raise RuntimeError("Cannot find callable {} in hubconf.py".format(entry))

    _check_dependencies(hubmodule)

    module = getattr(hubmodule, entry)(*args, **kwargs)
    return module


def help(
    repo_info: str,
    entry: str,
    git_host: str = DEFAULT_GIT_HOST,
    use_cache: bool = True,
    commit: str = None,
    protocol: str = DEFAULT_PROTOCOL,
) -> str:
    r"""This function returns docstring of entrypoint ``entry`` by following steps:

    1. Pull the repo code specified by git and repo_info.
    2. Load the entry defined in repo's hubconf.py
    3. Return docstring of function entry.

    Args:
        repo_info: a string with format ``"repo_owner/repo_name[:tag_name/:branch_name]"`` with an optional
            tag/branch. The default branch is ``master`` if not specified. Eg: ``"brain_sdk/MegBrain[:hub]"``
        entry: an entrypoint defined in hubconf.py
        git_host: host address of git repo. Eg: github.com
        use_cache: whether to use locally cached code or completely re-fetch.
        commit: commit id on github or gitlab.
        protocol: which protocol to use to get the repo, and HTTPS protocol only supports public repo on github.
            The value should be one of HTTPS, SSH.

    Returns:
        docstring of entrypoint ``entry``.
    """
    hubmodule = _init_hub(repo_info, git_host, use_cache, commit, protocol)

    if not hasattr(hubmodule, entry) or not callable(getattr(hubmodule, entry)):
        raise RuntimeError("Cannot find callable {} in hubconf.py".format(entry))

    doc = getattr(hubmodule, entry).__doc__
    return doc


def load_serialized_obj_from_url(url: str, model_dir=None) -> Any:
    """Loads MegEngine serialized object from the given URL.

    If the object is already present in ``model_dir``, it's deserialized and
    returned. If no ``model_dir`` is specified, it will be ``MGE_HOME/serialized``.

    Args:
        url: url to serialized object.
        model_dir: dir to cache target serialized file.

    Returns:
        loaded object.
    """
    if model_dir is None:
        model_dir = os.path.join(_get_megengine_home(), "serialized")
    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)

    # use hash as prefix to avoid filename conflict from different urls
    sha256 = hashlib.sha256()
    sha256.update(url.encode())
    digest = sha256.hexdigest()[:6]
    filename = digest + "_" + filename

    cached_file = os.path.join(model_dir, filename)
    logger.info(
        "load_serialized_obj_from_url: download to or using cached %s", cached_file
    )
    if not os.path.exists(cached_file):
        if is_distributed():
            logger.warning(
                "Downloading serialized object in DISTRIBUTED mode\n"
                "    File may be downloaded multiple times. We recommend\n"
                "    users to download in single process first."
            )
        download_from_url(url, cached_file)

    state_dict = _mge_load_serialized(cached_file)
    return state_dict


class pretrained:
    r"""Decorator which helps to download pretrained weights from the given url. Including fs, s3, http(s).

    For example, we can decorate a resnet18 function as follows

    .. code-block::

        @hub.pretrained("https://url/to/pretrained_resnet18.pkl")
        def resnet18(**kwargs):

    Returns:
        When decorated function is called with ``pretrained=True``, MegEngine will automatically
        download and fill the returned model with pretrained weights.
    """

    def __init__(self, url):
        self.url = url

    def __call__(self, func):
        @functools.wraps(func)
        def pretrained_model_func(
            pretrained=False, **kwargs
        ):  # pylint: disable=redefined-outer-name
            model = func(**kwargs)
            if pretrained:
                weights = load_serialized_obj_from_url(self.url)
                model.load_state_dict(weights)
            return model

        return pretrained_model_func


__all__ = [
    "list",
    "load",
    "help",
    "load_serialized_obj_from_url",
    "pretrained",
    "import_module",
]
