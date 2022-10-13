# -*- coding: utf-8 -*-
import hashlib
import os
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import Tuple
from zipfile import ZipFile

import requests
from tqdm import tqdm

from megengine import __version__
from megengine.utils.http_download import (
    CHUNK_SIZE,
    HTTP_CONNECTION_TIMEOUT,
    HTTPDownloadError,
)

from ..distributed import is_distributed, synchronized
from ..logger import get_logger
from .const import DEFAULT_BRANCH_NAME, HTTP_READ_TIMEOUT
from .exceptions import GitCheckoutError, GitPullError, InvalidGitHost, InvalidRepo
from .tools import cd

logger = get_logger(__name__)

HTTP_TIMEOUT = (HTTP_CONNECTION_TIMEOUT, HTTP_READ_TIMEOUT)

pattern = re.compile(
    r"^(?:[a-z0-9]"  # First character of the domain
    r"(?:[a-z0-9-_]{0,61}[a-z0-9])?\.)"  # Sub domain + hostname
    r"+[a-z0-9][a-z0-9-_]{0,61}"  # First 61 characters of the gTLD
    r"[a-z]$"  # Last character of the gTLD
)


class RepoFetcherBase:
    @classmethod
    def fetch(
        cls,
        git_host: str,
        repo_info: str,
        use_cache: bool = False,
        commit: str = None,
        silent: bool = True,
    ) -> str:
        raise NotImplementedError()

    @classmethod
    def _parse_repo_info(cls, repo_info: str) -> Tuple[str, str, str]:
        try:
            branch_info = DEFAULT_BRANCH_NAME
            if ":" in repo_info:
                prefix_info, branch_info = repo_info.split(":")
            else:
                prefix_info = repo_info
            repo_owner, repo_name = prefix_info.split("/")
            return repo_owner, repo_name, branch_info
        except ValueError:
            raise InvalidRepo("repo_info: '{}' is invalid.".format(repo_info))

    @classmethod
    def _check_git_host(cls, git_host):
        return cls._is_valid_domain(git_host) or cls._is_valid_host(git_host)

    @classmethod
    def _is_valid_domain(cls, s):
        try:
            return pattern.match(s.encode("idna").decode("ascii"))
        except UnicodeError:
            return False

    @classmethod
    def _is_valid_host(cls, s):
        nums = s.split(".")
        if len(nums) != 4 or any(not _.isdigit() for _ in nums):
            return False
        return all(0 <= int(_) < 256 for _ in nums)

    @classmethod
    def _gen_repo_dir(cls, repo_dir: str) -> str:
        return hashlib.sha1(repo_dir.encode()).hexdigest()[:16]


class GitSSHFetcher(RepoFetcherBase):
    @classmethod
    @synchronized
    def fetch(
        cls,
        git_host: str,
        repo_info: str,
        use_cache: bool = False,
        commit: str = None,
        silent: bool = True,
    ) -> str:
        """Fetches git repo by SSH protocol

        Args:
            git_host: host address of git repo. Eg: github.com
            repo_info: a string with format ``"repo_owner/repo_name[:tag_name/:branch_name]"`` with an optional
                tag/branch. The default branch is ``master`` if not specified. Eg: ``"brain_sdk/MegBrain[:hub]"``
            use_cache: whether to use locally fetched code or completely re-fetch.
            commit: commit id on github or gitlab.
            silent: whether to accept the stdout and stderr of the subprocess with PIPE, instead of
                displaying on the screen.

        Returns:
            directory where the repo code is stored.
        """
        if not cls._check_git_host(git_host):
            raise InvalidGitHost("git_host: '{}' is malformed.".format(git_host))

        repo_owner, repo_name, branch_info = cls._parse_repo_info(repo_info)
        normalized_branch_info = branch_info.replace("/", "_")
        repo_dir_raw = "{}_{}_{}".format(
            repo_owner, repo_name, normalized_branch_info
        ) + ("_{}".format(commit) if commit else "")
        repo_dir = (
            "_".join(__version__.split(".")) + "_" + cls._gen_repo_dir(repo_dir_raw)
        )
        git_url = "git@{}:{}/{}.git".format(git_host, repo_owner, repo_name)

        if use_cache and os.path.exists(repo_dir):  # use cache
            logger.debug("Cache Found in %s", repo_dir)
            return repo_dir

        if is_distributed():
            logger.warning(
                "When using `hub.load` or `hub.list` to fetch git repositories\n"
                "    in DISTRIBUTED mode for the first time, processes are synchronized to\n"
                "    ensure that target repository is ready to use for each process.\n"
                "    Users are expected to see this warning no more than ONCE, otherwise\n"
                "    (very little chance) you may need to remove corrupt cache\n"
                "    `%s` and fetch again.",
                repo_dir,
            )

        shutil.rmtree(repo_dir, ignore_errors=True)  # ignore and clear cache

        logger.debug(
            "Git Clone from Repo:%s Branch: %s to %s",
            git_url,
            normalized_branch_info,
            repo_dir,
        )

        kwargs = (
            {"stderr": subprocess.PIPE, "stdout": subprocess.PIPE} if silent else {}
        )
        if commit is None:
            # shallow clone repo by branch/tag
            p = subprocess.Popen(
                [
                    "git",
                    "clone",
                    "-b",
                    normalized_branch_info,
                    git_url,
                    repo_dir,
                    "--depth=1",
                ],
                **kwargs,
            )
            cls._check_clone_pipe(p)
        else:
            # clone repo and checkout to commit_id
            p = subprocess.Popen(["git", "clone", git_url, repo_dir], **kwargs)
            cls._check_clone_pipe(p)

            with cd(repo_dir):
                logger.debug("git checkout to %s", commit)
                p = subprocess.Popen(["git", "checkout", commit], **kwargs)
                _, err = p.communicate()
                if p.returncode:
                    shutil.rmtree(repo_dir, ignore_errors=True)
                    raise GitCheckoutError(
                        "Git checkout error, please check the commit id.\n"
                        + err.decode()
                    )
        with cd(repo_dir):
            shutil.rmtree(".git")

        return repo_dir

    @classmethod
    def _check_clone_pipe(cls, p):
        _, err = p.communicate()
        if p.returncode:
            raise GitPullError(
                "Repo pull error, please check repo info.\n" + err.decode()
            )


class GitHTTPSFetcher(RepoFetcherBase):
    @classmethod
    @synchronized
    def fetch(
        cls,
        git_host: str,
        repo_info: str,
        use_cache: bool = False,
        commit: str = None,
        silent: bool = True,
    ) -> str:
        """Fetches git repo by HTTPS protocol.

        Args:
            git_host: host address of git repo. Eg: github.com
            repo_info: a string with format ``"repo_owner/repo_name[:tag_name/:branch_name]"`` with an optional
                tag/branch. The default branch is ``master`` if not specified. Eg: ``"brain_sdk/MegBrain[:hub]"``
            use_cache: whether to use locally cached code or completely re-fetch.
            commit: commit id on github or gitlab.
            silent: whether to accept the stdout and stderr of the subprocess with PIPE, instead of
                displaying on the screen.
 

        Returns:
            directory where the repo code is stored.
        """
        if not cls._check_git_host(git_host):
            raise InvalidGitHost("git_host: '{}' is malformed.".format(git_host))

        repo_owner, repo_name, branch_info = cls._parse_repo_info(repo_info)
        normalized_branch_info = branch_info.replace("/", "_")
        repo_dir_raw = "{}_{}_{}".format(
            repo_owner, repo_name, normalized_branch_info
        ) + ("_{}".format(commit) if commit else "")
        repo_dir = (
            "_".join(__version__.split(".")) + "_" + cls._gen_repo_dir(repo_dir_raw)
        )
        archive_url = cls._git_archive_link(
            git_host, repo_owner, repo_name, branch_info, commit
        )

        if use_cache and os.path.exists(repo_dir):  # use cache
            logger.debug("Cache Found in %s", repo_dir)
            return repo_dir

        if is_distributed():
            logger.warning(
                "When using `hub.load` or `hub.list` to fetch git repositories "
                "in DISTRIBUTED mode for the first time, processes are synchronized to "
                "ensure that target repository is ready to use for each process.\n"
                "Users are expected to see this warning no more than ONCE, otherwise"
                "(very little chance) you may need to remove corrupt hub cache %s and fetch again."
            )

        shutil.rmtree(repo_dir, ignore_errors=True)  # ignore and clear cache

        logger.debug("Downloading from %s to %s", archive_url, repo_dir)
        cls._download_zip_and_extract(archive_url, repo_dir)

        return repo_dir

    @classmethod
    def _download_zip_and_extract(cls, url, target_dir):
        resp = requests.get(url, timeout=HTTP_TIMEOUT, stream=True)
        if resp.status_code != 200:
            raise HTTPDownloadError(
                "An error occured when downloading from {}".format(url)
            )

        total_size = int(resp.headers.get("Content-Length", 0))
        _bar = tqdm(total=total_size, unit="iB", unit_scale=True)

        with NamedTemporaryFile("w+b") as f:
            for chunk in resp.iter_content(CHUNK_SIZE):
                if not chunk:
                    break
                _bar.update(len(chunk))
                f.write(chunk)
            _bar.close()
            f.seek(0)
            with ZipFile(f) as temp_zip_f:
                zip_dir_name = temp_zip_f.namelist()[0].split("/")[0]
                temp_zip_f.extractall(".")
                shutil.move(zip_dir_name, target_dir)

    @classmethod
    def _git_archive_link(cls, git_host, repo_owner, repo_name, branch_info, commit):
        archive_link = "https://{}/{}/{}/archive/{}.zip".format(
            git_host, repo_owner, repo_name, commit or branch_info
        )

        return archive_link
