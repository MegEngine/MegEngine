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
    args = parser.parse_args()
    python_dir = os.path.dirname(__file__)
    version_txt_path = os.path.join(python_dir, 'version_template.py')
    commit_id = get_git_commit(python_dir)
    mge_ver_map = get_mge_version(version_txt_path)
    mge_ver = mge_ver_map['__version__'] if '__version__' in mge_ver_map else 'unknown'
    mge_intl = mge_ver_map['__internal__'] if '__internal__' in mge_ver_map else False
    with open(args.output, 'w') as f:
        f.write("__version__ = '{}'\n".format(mge_ver))
        f.write("git_version = {}\n".format(repr(commit_id)))
        if mge_intl:
            f.write("__internal__ = True\n")
