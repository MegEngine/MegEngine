#!/bin/bash -e

cd $(dirname $0)

docker build -t env_manylinux2014:latest .
