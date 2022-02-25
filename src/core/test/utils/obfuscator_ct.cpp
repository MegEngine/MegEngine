#include "megbrain/utils/obfuscator_ct.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestObFucatorCT, Normal) {
    auto run = []() {
        std::string obfuscator_str = MGB_OBFUSCATE_STR("mgb0001");
        ASSERT_STREQ(obfuscator_str.c_str(), "mgb0001");
    };

    //! invoke twice
    run();
    run();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
