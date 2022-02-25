#include "./fixture.h"
#include "test/cuda/utils.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

void MegcoreCUDA::SetUp() {
    cuda_check(cudaGetDeviceCount(&nr_devices_));
    printf("We have %d GPUs\n", nr_devices_);
}

void MegcoreCUDA::TearDown() {
    cuda_check(cudaDeviceReset());
}

// vim: syntax=cpp.doxygen
