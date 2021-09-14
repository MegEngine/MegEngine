# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

ifeq ($(SDKPATH),)
	CURDIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
	SDKPATH = $(CURDIR)/arm64-v8a
endif

TARGET := libmace_loader.so
all: $(TARGET)

# can be find at NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/
CXX = aarch64-linux-android21-clang++

CXXFLAGS = -std=c++14 -I$(SDKPATH)/include \
			-Dmgb_mace_loader_init=mgb_c_opr_init

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CXXFLAGS += -O0 -g
else
    CXXFLAGS += -O2
endif

LDFLAGS = -L$(SDKPATH)/lib -lmace -llog

LDLIBS = -shared -fPIC

$(TARGET): mace_loader.cpp extern_c_opr.h
	$(CXX) $(CXXFLAGS) $(LDLIBS) -o $@ $< $(LDFLAGS)

clean:
	@rm -f $(TARGET)
