/**
 * \file src/custom/impl/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include <sstream>
#include "megbrain/custom/utils.h"

using namespace mgb;

namespace custom {

void assert_failed_log(
        const char* file, int line, const char* func, const char* expr,
        const char* msg_fmt, ...) {
    std::string msg = ssprintf("`%s' is true at %s:%d: %s", expr, file, line, func);
    if (msg_fmt) {
        msg_fmt = convert_fmt_str(msg_fmt);
        va_list ap;
        va_start(ap, msg_fmt);
        msg.append("\nextra message: ");
        msg.append(svsprintf(msg_fmt, ap));
        va_end(ap);
    }
    printf("%s\n", msg.c_str());
}

UnImpleWarnLog::UnImpleWarnLog(
        const std::string& func, const std::string& attr, const std::string& val) {
    mgb_log_warn(
            "you are using the default custom %s function, the `%s` attribute "
            "of all the outputs tensor will be the same with inputs tensor[0]. "
            "If there is no input tensor, it will be `%s`",
            func.c_str(), attr.c_str(), val.c_str());
}

}  // namespace custom

#endif
