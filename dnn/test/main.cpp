#if defined(ONLY_BUILD_GI_API)
#include <gtest/gtest.h>

int gtest_main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    return ret;
}

int main(int argc, char** argv) {
    return gtest_main(argc, argv);
}
#else
extern "C" int gtest_main(int argc, char** argv);

int main(int argc, char** argv) {
    return gtest_main(argc, argv);
}
#endif
