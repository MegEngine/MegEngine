/**
 * \file test/src/include/megbrain/test/megdnn_helper.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megdnn/handle.h"

namespace mgb {
    //! get a naive megdnn handle on CPU, used for checking opr correctness
    megdnn::Handle* megdnn_naive_handle();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
