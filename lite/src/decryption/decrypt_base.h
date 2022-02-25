#pragma once
#include "lite/global.h"
#include "misc.h"

namespace lite {

struct DecryptionStaticData {
    std::unordered_map<
            std::string,
            std::pair<DecryptionFunc, std::shared_ptr<std::vector<uint8_t>>>>
            decryption_methods;
    LITE_MUTEX map_mutex;
};

DecryptionStaticData& decryption_static_data();

template <int count>
struct DecryptionRegister;

}  // namespace lite

#define CONCAT_IMPL(a, b)  a##b
#define MACRO_CONCAT(a, b) CONCAT_IMPL(a, b)

#define REGIST_DECRYPTION_METHOD(name_, func_, key_) \
    REGIST_DECRYPTION_METHOD_WITH_NUM(__COUNTER__, name_, func_, key_)

#define REGIST_DECRYPTION_METHOD_WITH_NUM(number_, name_, func_, key_)            \
    template <>                                                                   \
    struct DecryptionRegister<number_> {                                          \
        DecryptionRegister() { register_decryption_and_key(name_, func_, key_); } \
    };                                                                            \
    namespace {                                                                   \
    DecryptionRegister<number_> MACRO_CONCAT(decryption_, number_);               \
    }

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
