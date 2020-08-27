/**
 * \file dnn/test/rocm/megcore/fixture.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <gtest/gtest.h>

class MegcoreROCM : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    int nr_devices() { return nr_devices_; }

private:
    int nr_devices_;
};

// vim: syntax=cpp.doxygen
