#!/bin/bash -e

FileCheck --enable-var-scope --dump-input=fail "$@"
