#!/bin/bash
set -e

# This script can be used to find WARNINGs and ERRORs hide in Python docstring.
#
# * Usually we use Sphinx (https://www.sphinx-doc.org/) and its' tools
#   to build HTML documentation for Python projects, such as MegEngine. 
# 
# * It simulates the process of automatically extracting docstrings from source code
#   and try generating HTML pages, just like what MegEngine documentation will do.
#
#   Install required Python dependence with pip then you will get following tools:
#
#   * sphinx-apidoc: a tool for automatic generation of Sphinx sources that,  
#     using the autodoc extension,  document a whole package in the style of 
#     other automatic API documentation tools. For more details: 
#     https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
#
#   * sphinx-build: generates documentation frome specific files. For more details:
#     https://www.sphinx-doc.org/en/master/man/sphinx-build.html
#
# [NOTE]: You need build MegEngine first (target: develop) then run this script. 

DOC_PATH=`mktemp -d`
trap 'rm -rf "$DOC_PATH"' EXIT

cd $(dirname $0)/..

python3 -m pip install -r requires.txt
python3 -m pip install -r requires-sphinx.txt
sphinx-apidoc -f -F -e -o $DOC_PATH/source megengine "megengine/core/ops/builtin/*"
PYTHONPATH=. sphinx-build -j auto -c . -W --keep-going \
    $DOC_PATH/source $DOC_PATH/build
