#!/bin/bash

set -e

function set_tuna_yum_mirror() {
    cp /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak
    local repo=/etc/yum.repos.d/CentOS-Base.repo
    local plugin=/etc/yum/pluginconf.d/fastestmirror.conf
    sed -i "s/mirrorlist=/#mirrorlist=/g" $repo
    sed -i "s/#baseurl/baseurl/g" $repo
    sed -i "s/mirror.centos.org/mirrors.tuna.tsinghua.edu.cn/g" $repo
    sed -i "s/http/https/g" $repo
    sed -i "s/enabled=1/enabled=0/g" $plugin
    yum clean all
    # Build on brainpp unable to pull epel reo metadata so disable this
    # https://unix.stackexchange.com/questions/148144/unable-to-pull-epel-repository-metadata
    yum --disablerepo="epel" update nss
    yum makecache
}

function set_epel() {
    mv /etc/yum.repos.d/epel.repo /etc/yum.repos.d/epel.repo.backup
    mv /etc/yum.repos.d/epel-testing.repo /etc/yum.repos.d/epel-testing.repo.backup
    curl -o /etc/yum.repos.d/epel.repo http://mirrors.aliyun.com/repo/epel-7.repo
}

function set_yum_mirror() {
    mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup
    curl -o /etc/yum.repos.d/CentOS-Base.repo https://mirrors.aliyun.com/repo/Centos-7.repo
    yum makecache 
}

function set_pip_mirror() {
cat > /etc/pip.conf <<EOF
[global]
timeout = 180
index-url = https://mirrors.aliyun.com/pypi/simple
extra-index-url = 
    http://mirrors.i.brainpp.cn/pypi/simple/
    http://pypi.i.brainpp.cn/brain/dev/+simple
trusted-host =
    mirrors.i.brainpp.cn
    pypi.i.brainpp.cn
    mirrors.aliyun.com
EOF
}


function main() {
    local platform=$1
    case $platform in
        brainpp)
            set_epel
            set_yum_mirror
            set_pip_mirror
            ;;
        *)
            echo "No setup required"
            ;;
    esac
}

main "$@"
