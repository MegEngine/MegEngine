/**
 * \file sdk/load-and-run/main.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "mgblar.h"
#include "megbrain/common.h"

int main(int argc, char **argv) {
    MGB_TRY {
        return mgb_load_and_run_main(argc, argv);
    } MGB_CATCH (std::exception &exc, {
        fprintf(stderr, "caught exception: %s\n", exc.what());
        return -2;
    })
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

