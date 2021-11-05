import argparse
import os
import subprocess

def get_git_commit(src_dir):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src_dir).decode('ascii').strip()
    except Exception:
        return 'unknown'

def get_mge_version(version_txt_path):
    v = {}
    with open(version_txt_path) as fp:
        exec(fp.read(), v)
    return v

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate version.py to build path")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--major", type=int, required=True)
    parser.add_argument("--minor", type=int, required=True)
    parser.add_argument("--patch", type=int, required=True)
    parser.add_argument("--rc", type=str, required=False)
    parser.add_argument("--internal", action='store_true')
    args = parser.parse_args()
    python_dir = os.path.dirname(__file__)
    commit_id = get_git_commit(python_dir)
    mge_ver = str(args.major) + "." + str(args.minor) + "." + str(args.patch)
    if args.rc is not None:
        mge_ver += args.rc
    with open(args.output, 'w') as f:
        f.write("__version__ = '{}'\n".format(mge_ver))
        f.write("git_version = {}\n".format(repr(commit_id)))
        if args.internal:
            f.write("__internal__ = True\n")
