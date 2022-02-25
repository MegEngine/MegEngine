import collections
import ctypes
import glob
import os
import re
import subprocess
import sys
import time
from typing import List, Optional, Union

from ..core.ops.custom import get_custom_op_abi_tag, load
from ..logger import get_logger


def _get_win_folder_with_ctypes(csidl_name):
    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
    }[csidl_name]

    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # Downgrade to short path name if have highbit chars. See
    # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2

    return buf.value


system = sys.platform
if system == "win32":
    _get_win_folder = _get_win_folder_with_ctypes

PLAT_TO_VCVARS = {
    "win-amd64": "x86_amd64",
}

logger = get_logger()

# environment varible
ev_custom_op_root_dir = "MGE_CUSTOM_OP_DIR"
ev_cuda_root_dir = "CUDA_ROOT_DIR"
ev_cudnn_root_dir = "CUDNN_ROOT_DIR"

# operating system
IS_WINDOWS = system == "win32"
IS_LINUX = system == "linux"
IS_MACOS = system == "darwin"

MGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MGE_INC_PATH = os.path.join(MGE_PATH, "core", "include")
MGE_LIB_PATH = os.path.join(MGE_PATH, "core", "lib")
MGE_ABI_VER = 0


# compile version
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_CLANG_CL_VERSION = (12, 0, 1)

# compile flags
COMMON_MSVC_FLAGS = [
    "/MD",
    "/wd4002",
    "/wd4819",
    "/EHsc",
]

MSVC_IGNORE_CUDAFE_WARNINGS = [
    "field_without_dll_interface",
]

COMMON_NVCC_FLAGS = []

# Finds the CUDA install path
def _find_cuda_root_dir() -> Optional[str]:
    cuda_root_dir = os.environ.get(ev_cuda_root_dir)
    if cuda_root_dir is None:
        try:
            which = "where" if IS_WINDOWS else "which"
            with open(os.devnull, "w") as devnull:
                nvcc = (
                    subprocess.check_output([which, "nvcc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                cuda_root_dir = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            if IS_WINDOWS:
                cuda_root_dir = os.environ.get("CUDA_PATH", None)
                if cuda_root_dir == None:
                    cuda_root_dirs = glob.glob(
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*"
                    )
                    if len(cuda_root_dirs) == 0:
                        cuda_root_dir = ""
                    else:
                        cuda_root_dir = cuda_root_dirs[0]
            else:
                cuda_root_dir = "/usr/local/cuda"
            if not os.path.exists(cuda_root_dir):
                cuda_root_dir = None
    return cuda_root_dir


def _find_cudnn_root_dir() -> Optional[str]:
    cudnn_root_dir = os.environ.get(ev_cudnn_root_dir)
    return cudnn_root_dir


CUDA_ROOT_DIR = _find_cuda_root_dir()
CUDNN_ROOT_DIR = _find_cudnn_root_dir()

#####################################################################
# Phase 1
#####################################################################


def _is_cuda_file(path: str) -> bool:
    valid_ext = [".cu", ".cuh"]
    return os.path.splitext(path)[1] in valid_ext


# Return full path to the user-specific cache dir for this application.
# Typical user cache directories are:
#     Mac OS X:   ~/Library/Caches/<AppName>
#     Unix:       ~/.cache/<AppName> (XDG default)
#     Windows:    C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>\Cache
def _get_user_cache_dir(appname=None, appauthor=None, version=None, opinion=True):
    if system == "win32":
        appauthor = appname if appauthor is None else appauthor
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor)
            else:
                path = os.path.join(path, appname)
            if opinion:
                path = os.path.join(path, "Cache")
    elif system == "darwin":
        path = os.path.expanduser("~/Library/Caches")
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path


# Returns the path to the root folder under which custom op will built.
def _get_default_build_root() -> str:
    return os.path.realpath(_get_user_cache_dir(appname="mge_custom_op"))


def _get_build_dir(name: str) -> str:
    custom_op_root_dir = os.environ.get(ev_custom_op_root_dir)
    if custom_op_root_dir is None:
        custom_op_root_dir = _get_default_build_root()

    build_dir = os.path.join(custom_op_root_dir, name)
    return build_dir


#####################################################################
# Phase 2
#####################################################################


def update_hash(seed, value):
    # using boost::hash_combine
    # https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
    return seed ^ (hash(value) + 0x9E3779B9 + (seed << 6) + (seed >> 2))


def hash_source_files(hash_value, source_files):
    for filename in source_files:
        with open(filename) as file:
            hash_value = update_hash(hash_value, file.read())
    return hash_value


def hash_build_args(hash_value, build_args):
    for group in build_args:
        for arg in group:
            hash_value = update_hash(hash_value, arg)
    return hash_value


Entry = collections.namedtuple("Entry", "version, hash")


class Versioner(object):
    def __init__(self):
        self.entries = {}

    def get_version(self, name):
        entry = self.entries.get(name)
        return None if entry is None else entry.version

    def bump_version_if_changed(
        self, name, sources, build_args, build_dir, with_cuda, with_cudnn, abi_tag
    ):
        hash_value = 0
        hash_value = hash_source_files(hash_value, sources)
        hash_value = hash_build_args(hash_value, build_args)
        hash_value = update_hash(hash_value, build_dir)
        hash_value = update_hash(hash_value, with_cuda)
        hash_value = update_hash(hash_value, with_cudnn)
        hash_value = update_hash(hash_value, abi_tag)

        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        return entry.version


custom_op_versioner = Versioner()


def version_check(
    name, sources, build_args, build_dir, with_cuda, with_cudnn, abi_tag,
):
    old_version = custom_op_versioner.get_version(name)
    version = custom_op_versioner.bump_version_if_changed(
        name, sources, build_args, build_dir, with_cuda, with_cudnn, abi_tag,
    )
    return version, old_version


#####################################################################
# Phase 3
#####################################################################


def _check_ninja_availability():
    try:
        subprocess.check_output("ninja --version".split())
    except Exception:
        raise RuntimeError(
            "Ninja is required to build custom op, please install ninja and update your PATH"
        )


def _mge_is_built_from_src():
    file_path = os.path.abspath(__file__)
    if "site-packages" in file_path:
        return False
    else:
        return True


def _accepted_compilers_for_platform():
    if IS_WINDOWS:
        return ["clang-cl"]
    if IS_MACOS:
        return ["clang++", "clang"]
    if IS_LINUX:
        return ["g++", "gcc", "gnu-c++", "gnu-cc"]


# Verifies that the compiler is the expected one for the current platform.
def _check_compiler_existed_for_platform(compiler: str) -> bool:
    # there is no suitable cmd like `which` on windows, so we assume the compiler is always true on windows
    if IS_WINDOWS:
        try:
            version_string = subprocess.check_output(
                ["clang-cl", "--version"], stderr=subprocess.STDOUT
            ).decode()
            return True
        except Exception:
            return False

    # use os.path.realpath to resolve any symlinks, in particular from "c++" to e.g. "g++".
    which = subprocess.check_output(["which", compiler], stderr=subprocess.STDOUT)
    compiler_path = os.path.realpath(which.decode().strip())
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True

    version_string = subprocess.check_output(
        [compiler, "-v"], stderr=subprocess.STDOUT
    ).decode()
    if sys.platform.startswith("linux"):
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            return False
        compiler_path = os.path.realpath(results[0].strip())
        return any(name in compiler_path for name in _accepted_compilers_for_platform())

    if sys.platform.startswith("darwin"):
        return version_string.startswith("Apple clang")

    return False


# Verifies that the given compiler is ABI-compatible with MegEngine.
def _check_compiler_abi_compatibility(compiler: str):
    # we think if the megengine is built from source, the user will use the same compiler to compile the custom op
    if _mge_is_built_from_src() or os.environ.get("MGE_CHECK_ABI", "1") == "0":
        return True

    # [TODO] There is no particular minimum version we need for clang, so we"re good here.
    if sys.platform.startswith("darwin"):
        return True

    try:
        if sys.platform.startswith("linux"):
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output(
                [compiler, "-dumpfullversion", "-dumpversion"]
            )
            version = versionstr.decode().strip().split(".")
        else:
            minimum_required_version = MINIMUM_CLANG_CL_VERSION
            compiler_info = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT
            )
            match = re.search(r"(\d+)\.(\d+)\.(\d+)", compiler_info.decode().strip())
            version = (0, 0, 0) if match is None else match.groups()
    except Exception:
        _, error, _ = sys.exc_info()
        logger.warning(
            "Error checking compiler version for {}: {}".format(compiler, error)
        )
        return False

    if tuple(map(int, version)) >= minimum_required_version:
        return True

    return False


def _check_compiler_comatibility():
    # we use clang-cl on windows, refer: https://clang.llvm.org/docs/UsersManual.html#clang-cl
    compiler = (
        os.environ.get("CXX", "clang-cl")
        if IS_WINDOWS
        else os.environ.get("CXX", "c++")
    )

    existed = _check_compiler_existed_for_platform(compiler)
    if existed == False:
        log_str = (
            "Cannot find compiler which is compatible with the compiler "
            "MegEngine was built with for this platform, which is {mge_compiler} on "
            "{platform}. Please use {mge_compiler} to to compile your extension. "
            "Alternatively, you may compile MegEngine from source using "
            "{user_compiler}, and then you can also use {user_compiler} to compile "
            "your extension."
        ).format(
            user_compiler=compiler,
            mge_compiler=_accepted_compilers_for_platform()[0],
            platform=sys.platform,
        )

        logger.warning(log_str)
        return False

    compatible = _check_compiler_abi_compatibility(compiler)
    if compatible == False:
        log_str = (
            "Your compiler version may be ABI-incompatible with MegEngine! "
            "Please use a compiler that is ABI-compatible with GCC 5.0 on Linux "
            "and LLVM/Clang 12.0 on Windows ."
        )
        logger.warning(log_str)
    return True


#####################################################################
# Phase 4
#####################################################################


# Quote command-line arguments for DOS/Windows conventions.
def _nt_quote_args(args: Optional[List[str]]) -> List[str]:
    # Cover None-type
    if not args:
        return []
    return ['"{}"'.format(arg) if " " in arg else arg for arg in args]


# Now we need user to specify the arch of GPU
def _get_cuda_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    return []


def _setup_sys_includes(with_cuda: bool, with_cudnn: bool):
    includes = [os.path.join(MGE_INC_PATH)]
    if with_cuda:
        includes.append(os.path.join(CUDA_ROOT_DIR, "include"))
    if with_cudnn:
        includes.append(os.path.join(CUDNN_ROOT_DIR, "include"))
    return includes


def _setup_includes(extra_include_paths: List[str], with_cuda: bool, with_cudnn: bool):
    user_includes = [os.path.abspath(path) for path in extra_include_paths]
    system_includes = _setup_sys_includes(with_cuda, with_cudnn)
    if IS_WINDOWS:
        user_includes += system_includes
        system_includes.clear()
    return user_includes, system_includes


def _setup_common_cflags(user_includes: List[str], system_includes: List[str]):
    common_cflags = []
    common_cflags += ["-I{}".format(include) for include in user_includes]
    common_cflags += ["-isystem {}".format(include) for include in system_includes]
    if not IS_WINDOWS:
        common_cflags += ["-D_GLIBCXX_USE_CXX11_ABI={}".format(MGE_ABI_VER)]
    return common_cflags


def _setup_cuda_cflags(cflags: List[str], extra_cuda_cflags: List[str]):
    cuda_flags = cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags()
    if IS_WINDOWS:
        for flag in COMMON_MSVC_FLAGS:
            cuda_flags = ["-Xcompiler", flag] + cuda_flags
        for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
            cuda_flags = ["-Xcudafe", "--diag_suppress=" + ignore_warning] + cuda_flags
        cuda_flags = _nt_quote_args(cuda_flags)
        cuda_flags += _nt_quote_args(extra_cuda_cflags)
    else:
        cuda_flags += ["--compiler-options", '"-fPIC"']
        cuda_flags += extra_cuda_cflags
        if not any(flag.startswith("-std=") for flag in cuda_flags):
            cuda_flags.append("-std=c++14")
        if os.getenv("CC") is not None:
            cuda_flags = ["-ccbin", os.getenv("CC")] + cuda_flags
    return cuda_flags


def _setup_ldflags(
    extra_ldflags: List[str], with_cuda: bool, with_cudnn: bool
) -> List[str]:
    ldflags = extra_ldflags
    if IS_WINDOWS:
        ldflags.append(os.path.join(MGE_LIB_PATH, "megengine_shared.lib"))
        if with_cuda:
            ldflags.append(os.path.join(CUDA_ROOT_DIR, "lib", "x64", "cudart.lib"))
        if with_cudnn:
            ldflags.append(os.path.join(CUDNN_ROOT_DIR, "lib", "x64", "cudnn.lib"))

    else:
        ldflags.append("-lmegengine_shared -L{}".format(MGE_LIB_PATH))
        ldflags.append("-Wl,-rpath,{}".format(MGE_LIB_PATH))
        if with_cuda:
            ldflags.append("-lcudart")
            ldflags.append("-L{}".format(os.path.join(CUDA_ROOT_DIR, "lib64")))
            ldflags.append("-Wl,-rpath,{}".format(os.path.join(CUDA_ROOT_DIR, "lib64")))
        if with_cudnn:
            ldflags.append("-L{}".format(os.path.join(CUDNN_ROOT_DIR, "lib64")))
            ldflags.append(
                "-Wl,-rpath,{}".format(os.path.join(CUDNN_ROOT_DIR, "lib64"))
            )

    return ldflags


def _add_shared_flag(ldflags: List[str]):
    ldflags += ["/LD" if IS_WINDOWS else "-shared"]
    return ldflags


#####################################################################
# Phase 5
#####################################################################


def _obj_file_path(src_file_path: str):
    file_name = os.path.splitext(os.path.basename(src_file_path))[0]
    if _is_cuda_file(src_file_path):
        target = "{}.cuda.o".format(file_name)
    else:
        target = "{}.o".format(file_name)
    return target


def _dump_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
):
    def sanitize_flags(flags):
        return [] if flags is None else [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    ldflags = sanitize_flags(ldflags)

    assert len(sources) == len(objects)
    assert len(sources) > 0

    if IS_WINDOWS:
        compiler = os.environ.get("CXX", "clang-cl")
    else:
        compiler = os.environ.get("CXX", "c++")

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append("cxx = {}".format(compiler))
    if with_cuda:
        nvcc = os.path.join(CUDA_ROOT_DIR, "bin", "nvcc")
        config.append("nvcc = {}".format(nvcc))

    flags = ["cflags = {}".format(" ".join(cflags))]
    flags.append("post_cflags = {}".format(" ".join(post_cflags)))
    if with_cuda:
        flags.append("cuda_cflags = {}".format(" ".join(cuda_cflags)))
        flags.append("cuda_post_cflags = {}".format(" ".join(cuda_post_cflags)))
    flags.append("ldflags = {}".format(" ".join(ldflags)))

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        compile_rule.append(
            "  command = clang-cl /showIncludes $cflags -c $in /Fo$out $post_cflags"
        )
        compile_rule.append("  deps = msvc")
    else:
        compile_rule.append(
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
        compile_rule.append("  depfile = $out.d")
        compile_rule.append("  deps = gcc")

    if with_cuda:
        cuda_compile_rule = ["rule cuda_compile"]
        nvcc_gendeps = ""
        cuda_compile_rule.append(
            "  command = $nvcc {} $cuda_cflags -c $in -o $out $cuda_post_cflags".format(
                nvcc_gendeps
            )
        )

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = "cuda_compile" if is_cuda_source else "compile"
        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append("build {}: {} {}".format(object_file, rule, source_file))

    if library_target is not None:
        link_rule = ["rule link"]
        if IS_WINDOWS:
            link_rule.append("  command = clang-cl $in /nologo $ldflags /out:$out")
        else:
            link_rule.append("  command = $cxx $in $ldflags -o $out")

        link = ["build {}: link {}".format(library_target, " ".join(objects))]
        default = ["default {}".format(library_target)]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, "w") as build_file:
        for block in blocks:
            lines = "\n".join(block)
            build_file.write("{}\n\n".format(lines))


class FileBaton:
    def __init__(self, lock_file_path, wait_seconds=0.1):
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None

    def try_acquire(self):
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)


#####################################################################
# Phase 6
#####################################################################


def _build_with_ninja(build_dir: str, verbose: bool, error_prefix: str):
    command = ["ninja", "-v"]
    env = os.environ.copy()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fileno = 1
        subprocess.run(
            command,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_dir,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        with open(os.path.join(build_dir, "build.ninja")) as f:
            lines = f.readlines()
            print(lines)
        _, error, _ = sys.exc_info()
        message = error_prefix
        if hasattr(error, "output") and error.output:
            message += ": {}".format(error.output.decode())
        raise RuntimeError(message) from e


def build(
    name: str,
    sources: Union[str, List[str]],
    extra_cflags: Union[str, List[str]] = [],
    extra_cuda_cflags: Union[str, List[str]] = [],
    extra_ldflags: Union[str, List[str]] = [],
    extra_include_paths: Union[str, List[str]] = [],
    with_cuda: Optional[bool] = None,
    build_dir: Optional[bool] = None,
    verbose: bool = False,
    abi_tag: Optional[int] = None,
) -> str:
    r"""Build a Custom Op with ninja in the way of just-in-time (JIT).

    To build the custom op, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/mge_custom_op/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the custom op. This location can be overridden in two ways.
    First, if the ``MGE_CUSTOM_OP_DIR`` environment variable is set, it
    replaces ``<tmp>/mge_custom_op`` and all custom op will be compiled
    into subfolders of this directory. Second, if the ``build_dir``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your custom op
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_ROOT_DIR`` environment variable is the
    safest option. If you use CUDNN, please also setting the ``CUDNN_ROOT_DIR`` 
    environment variable.

    Args:
        name: The name of the custom op to build.
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.cu`` or
            ``.cuh`` in ``sources``. Set it to `True`` to force CUDA headers
            and libraries to be included.
        build_dir: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        abi_tag: Determines the value of MACRO ``_GLIBCXX_USE_CXX11_ABI``
            in gcc compiler, should be ``0`` or ``1``.

    Returns:
        the compiled dynamic library path

    """

    # phase 1: prepare config
    if abi_tag == None:
        abi_tag = get_custom_op_abi_tag()
    global MGE_ABI_VER
    MGE_ABI_VER = abi_tag

    def strlist(args, name):
        assert isinstance(args, str) or isinstance(
            args, list
        ), "{} must be str or list[str]".format(name)
        if isinstance(args, str):
            return [args]
        for arg in args:
            assert isinstance(arg, str)
        args = [arg.strip() for arg in args]
        return args

    sources = strlist(sources, "sources")
    extra_cflags = strlist(extra_cflags, "extra_cflags")
    extra_cuda_cflags = strlist(extra_cuda_cflags, "extra_cuda_cflags")
    extra_ldflags = strlist(extra_ldflags, "extra_ldflags")
    extra_include_paths = strlist(extra_include_paths, "extra_include_paths")

    with_cuda = any(map(_is_cuda_file, sources)) if with_cuda is None else with_cuda
    with_cudnn = any(["cudnn" in f for f in extra_ldflags])

    if CUDA_ROOT_DIR == None and with_cuda:
        print(
            "No CUDA runtime is found, using {}=/path/to/your/cuda_root_dir".format(
                ev_cuda_root_dir
            )
        )
    if CUDNN_ROOT_DIR == None and with_cudnn:
        print(
            "Cannot find the root directory of cudnn, using {}=/path/to/your/cudnn_root_dir".format(
                ev_cudnn_root_dir
            )
        )

    build_dir = os.path.abspath(
        _get_build_dir(name) if build_dir is None else build_dir
    )
    if not os.path.exists(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    if verbose:
        print("Using {} to build megengine custom op".format(build_dir))

    # phase 2: version check
    version, old_version = version_check(
        name,
        sources,
        [extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths],
        build_dir,
        with_cuda,
        with_cudnn,
        abi_tag,
    )
    if verbose:
        if version != old_version and old_version != None:
            print(
                "Input conditions of custom op {} have changed, bumping to version {}".format(
                    name, version
                )
            )
        print("Building custom op {} with version {}".format(name, version))
    if version == old_version:
        if verbose:
            print(
                "No modifications detected for {}, skipping build step...".format(name)
            )
        return
    name = "{}_v{}".format(name, version)

    # phase 3: compiler and ninja check
    _check_ninja_availability()
    _check_compiler_comatibility()

    # phase 4: setup the compile flags
    user_includes, system_includes = _setup_includes(
        extra_include_paths, with_cuda, with_cudnn
    )
    common_cflags = _setup_common_cflags(user_includes, system_includes)
    cuda_cflags = (
        _setup_cuda_cflags(common_cflags, extra_cuda_cflags) if with_cuda else None
    )
    ldflags = _setup_ldflags(extra_ldflags, with_cuda, with_cudnn)

    if IS_WINDOWS:
        cflags = common_cflags + COMMON_MSVC_FLAGS + extra_cflags
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ["-fPIC", "-std=c++14"] + extra_cflags

    ldflags = _add_shared_flag(ldflags)
    if sys.platform.startswith("darwin"):
        ldflags.append("-undefined dynamic_lookup")
    elif IS_WINDOWS:
        ldflags += ["/link"]
        ldflags = _nt_quote_args(ldflags)

    baton = FileBaton(os.path.join(build_dir, "lock"))
    if baton.try_acquire():
        try:
            # phase 5: generate ninja build file
            objs = [_obj_file_path(src) for src in sources]
            name += ".dll" if IS_WINDOWS else ".so"

            build_file_path = os.path.join(build_dir, "build.ninja")
            if verbose:
                print("Emitting ninja build file {}".format(build_file_path))
            _dump_ninja_file(
                path=build_file_path,
                cflags=cflags,
                post_cflags=None,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=None,
                sources=sources,
                objects=objs,
                ldflags=ldflags,
                library_target=name,
                with_cuda=with_cuda,
            )

            # phase 6: build with ninja
            if verbose:
                print(
                    "Compiling and linking your custom op {}".format(
                        os.path.join(build_dir, name)
                    )
                )
            _build_with_ninja(build_dir, verbose, "compiling error")
        finally:
            baton.release()
    else:
        baton.wait()

    return os.path.join(build_dir, name)


def build_and_load(
    name: str,
    sources: Union[str, List[str]],
    extra_cflags: Union[str, List[str]] = [],
    extra_cuda_cflags: Union[str, List[str]] = [],
    extra_ldflags: Union[str, List[str]] = [],
    extra_include_paths: Union[str, List[str]] = [],
    with_cuda: Optional[bool] = None,
    build_dir: Optional[bool] = None,
    verbose: bool = False,
    abi_tag: Optional[int] = None,
) -> str:
    r"""Build and Load a Custom Op with ninja in the way of just-in-time (JIT).
    Same as the function ``build()`` but load the built dynamic library.

    Args:
        same as ``build()``

    Returns:
        the compiled dynamic library path

    """

    lib_path = build(
        name,
        sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
        with_cuda,
        build_dir,
        verbose,
        abi_tag,
    )
    if verbose:
        print("Load the compiled custom op {}".format(lib_path))
    load(lib_path)
    return lib_path
