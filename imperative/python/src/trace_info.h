/**
 * \file imperative/python/src/trace_info.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "inttypes.h"
#include "Python.h"

namespace mgb::imperative::python {

struct TraceInfo {
    int64_t mixin_handle = -1;
    bool recording = false;
    bool copied = false;

    // refer to CompiledTensorProxy in tracing.py, works from second trace step
    PyObject* compiled_info = nullptr;
    // refer to TensorInfo in tracing.py, only works in first trace step
    PyObject* trace_mixin_info = nullptr;

    TraceInfo() = default;

    TraceInfo& operator=(const TraceInfo& that) {
        mixin_handle = that.mixin_handle;
        recording = that.recording;

        trace_mixin_info = that.trace_mixin_info;
        Py_XINCREF(trace_mixin_info);
        compiled_info = that.compiled_info;
        Py_XINCREF(compiled_info);

        copied = true;
        return *this;
    }

    ~TraceInfo() {
        Py_XDECREF(trace_mixin_info);
        Py_XDECREF(compiled_info);
    }

private:
    TraceInfo(const TraceInfo& that) = default;
};

} // namespace mgb::imperative::python
