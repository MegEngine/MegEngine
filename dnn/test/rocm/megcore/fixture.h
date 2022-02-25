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
