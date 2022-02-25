#pragma once
#include "lite/global.h"
#include "mutex"

namespace lite {

struct ParseInfoStaticData {
    std::unordered_map<std::string, ParseInfoFunc> parse_info_methods;
    LITE_MUTEX map_mutex;
};

ParseInfoStaticData& parse_info_static_data();

template <int count>
struct ParseInfoRegister;
}  // namespace lite

#define REGIST_PARSE_INFO_FUNCTION(name_, func_) \
    REGIST_PARSE_INFO_FUNCTION_WITH_NUM(__COUNTER__, name_, func_)

#define REGIST_PARSE_INFO_FUNCTION_WITH_NUM(number_, name_, func_)      \
    template <>                                                         \
    struct ParseInfoRegister<number_> {                                 \
        ParseInfoRegister() { register_parse_info_func(name_, func_); } \
    };                                                                  \
    namespace {                                                         \
    ParseInfoRegister<number_> parse_info_##number_;                    \
    }

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
