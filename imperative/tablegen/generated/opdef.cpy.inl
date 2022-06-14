// clang-format off

template<> struct EnumTrait<AdaptivePooling::Mode> {
    static constexpr const char *name = "AdaptivePooling.Mode";
    static constexpr std::underlying_type_t<AdaptivePooling::Mode> max = 3 - 1;
};
template<> PyTypeObject* EnumWrapper<AdaptivePooling::Mode>::type = nullptr;

template<> const char*
EnumWrapper<AdaptivePooling::Mode>::members[] = {"MAX", "AVERAGE", "AVERAGE_COUNT_EXCLUDE_PADDING"};

template<> std::unordered_map<std::string, AdaptivePooling::Mode>
EnumWrapper<AdaptivePooling::Mode>::mem2value = {{normalize_enum("MAX"), AdaptivePooling::Mode::MAX}, {normalize_enum("AVERAGE"), AdaptivePooling::Mode::AVERAGE}, {normalize_enum("AVERAGE_COUNT_EXCLUDE_PADDING"), AdaptivePooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING}};
template<> PyObject* EnumWrapper<AdaptivePooling::Mode>::pyobj_insts[3] = {nullptr};

void _init_py_AdaptivePooling_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<AdaptivePooling::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<AdaptivePooling::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<AdaptivePooling::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<AdaptivePooling::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.AdaptivePooling.Mode",
        // basicsize
        sizeof(EnumWrapper<AdaptivePooling::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("AdaptivePooling.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Mode>*>(inst)->value = AdaptivePooling::Mode::MAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MAX", inst) >= 0);
    EnumWrapper<AdaptivePooling::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Mode>*>(inst)->value = AdaptivePooling::Mode::AVERAGE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "AVERAGE", inst) >= 0);
    EnumWrapper<AdaptivePooling::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Mode>*>(inst)->value = AdaptivePooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "AVERAGE_COUNT_EXCLUDE_PADDING", inst) >= 0);
    EnumWrapper<AdaptivePooling::Mode>::pyobj_insts[2] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<AdaptivePooling::Format> {
    static constexpr const char *name = "AdaptivePooling.Format";
    static constexpr std::underlying_type_t<AdaptivePooling::Format> max = 18 - 1;
};
template<> PyTypeObject* EnumWrapper<AdaptivePooling::Format>::type = nullptr;

template<> const char*
EnumWrapper<AdaptivePooling::Format>::members[] = {"NCHW", "NHWC", "NHWCD4", "NCHW4", "NCHW8", "NCHW32", "NCHW88", "NCHW44", "NCHW44_DOT", "NCHW4_NCHW32", "NCHW32_NCHW4", "NCHW4_NCHW", "NHWC_NCHW", "NHWC_NCHW4_IC_SMALL", "NCHW_NCHW4_IC_SMALL", "CHWN4", "NCHW64", "NCHW4_NHWC"};

template<> std::unordered_map<std::string, AdaptivePooling::Format>
EnumWrapper<AdaptivePooling::Format>::mem2value = {{normalize_enum("NCHW"), AdaptivePooling::Format::NCHW}, {normalize_enum("NHWC"), AdaptivePooling::Format::NHWC}, {normalize_enum("NHWCD4"), AdaptivePooling::Format::NHWCD4}, {normalize_enum("NCHW4"), AdaptivePooling::Format::NCHW4}, {normalize_enum("NCHW8"), AdaptivePooling::Format::NCHW8}, {normalize_enum("NCHW32"), AdaptivePooling::Format::NCHW32}, {normalize_enum("NCHW88"), AdaptivePooling::Format::NCHW88}, {normalize_enum("NCHW44"), AdaptivePooling::Format::NCHW44}, {normalize_enum("NCHW44_DOT"), AdaptivePooling::Format::NCHW44_DOT}, {normalize_enum("NCHW4_NCHW32"), AdaptivePooling::Format::NCHW4_NCHW32}, {normalize_enum("NCHW32_NCHW4"), AdaptivePooling::Format::NCHW32_NCHW4}, {normalize_enum("NCHW4_NCHW"), AdaptivePooling::Format::NCHW4_NCHW}, {normalize_enum("NHWC_NCHW"), AdaptivePooling::Format::NHWC_NCHW}, {normalize_enum("NHWC_NCHW4_IC_SMALL"), AdaptivePooling::Format::NHWC_NCHW4_IC_SMALL}, {normalize_enum("NCHW_NCHW4_IC_SMALL"), AdaptivePooling::Format::NCHW_NCHW4_IC_SMALL}, {normalize_enum("CHWN4"), AdaptivePooling::Format::CHWN4}, {normalize_enum("NCHW64"), AdaptivePooling::Format::NCHW64}, {normalize_enum("NCHW4_NHWC"), AdaptivePooling::Format::NCHW4_NHWC}};
template<> PyObject* EnumWrapper<AdaptivePooling::Format>::pyobj_insts[18] = {nullptr};

void _init_py_AdaptivePooling_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<AdaptivePooling::Format>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<AdaptivePooling::Format>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<AdaptivePooling::Format>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<AdaptivePooling::Format>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.AdaptivePooling.Format",
        // basicsize
        sizeof(EnumWrapper<AdaptivePooling::Format>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Format").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("AdaptivePooling.Format").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NHWC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWC", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NHWCD4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWCD4", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW8", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW32", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW88;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW88", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[6] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW44;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW44", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[7] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW44_DOT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW44_DOT", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[8] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW4_NCHW32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4_NCHW32", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[9] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW32_NCHW4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW32_NCHW4", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[10] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW4_NCHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4_NCHW", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[11] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NHWC_NCHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWC_NCHW", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[12] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NHWC_NCHW4_IC_SMALL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWC_NCHW4_IC_SMALL", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[13] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW_NCHW4_IC_SMALL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW_NCHW4_IC_SMALL", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[14] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::CHWN4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CHWN4", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[15] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW64;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW64", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[16] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<AdaptivePooling::Format>*>(inst)->value = AdaptivePooling::Format::NCHW4_NHWC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4_NHWC", inst) >= 0);
    EnumWrapper<AdaptivePooling::Format>::pyobj_insts[17] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(AdaptivePooling) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(AdaptivePooling)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"shape", serialization<decltype(opdef.shape)>::dump(opdef.shape)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(AdaptivePooling)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("shape");
        if (iter != state.end()) {
            opdef.shape = serialization<decltype(opdef.shape)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(AdaptivePooling)

int PyOp(AdaptivePooling)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "format", "shape", "scope", NULL};
    PyObject *mode = NULL, *format = NULL, *shape = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", const_cast<char**>(kwlist), &mode, &format, &shape, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AdaptivePooling)*>(self)->inst().mode =
                    py::cast<decltype(AdaptivePooling::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AdaptivePooling)*>(self)->inst().format =
                    py::cast<decltype(AdaptivePooling::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (shape) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AdaptivePooling)*>(self)->inst().shape =
                    py::cast<decltype(AdaptivePooling::shape)>(py::handle(shape));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(AdaptivePooling)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(AdaptivePooling, mode), py_set_generic(AdaptivePooling, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("format"), py_get_generic(AdaptivePooling, format), py_set_generic(AdaptivePooling, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("shape"), py_get_generic(AdaptivePooling, shape), py_set_generic(AdaptivePooling, shape), const_cast<char*>("shape"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(AdaptivePooling)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(AdaptivePooling)::getstate, METH_NOARGS, "AdaptivePooling getstate"},
    {const_cast<char*>("__setstate__"), PyOp(AdaptivePooling)::setstate, METH_VARARGS, "AdaptivePooling setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(AdaptivePooling)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(AdaptivePooling)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(AdaptivePooling)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(AdaptivePooling)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., format: Union[str, Format] = ..., shape: list[int] = ...) -> None\n"
};

void _init_py_AdaptivePooling(py::module m) {
    using py_op = PyOp(AdaptivePooling);
    auto& py_type = PyOpType(AdaptivePooling);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.AdaptivePooling";
    py_type.tp_basicsize = sizeof(PyOp(AdaptivePooling));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "AdaptivePooling";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(AdaptivePooling), &PyOp(AdaptivePooling)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_AdaptivePooling_Mode(py_type);
    _init_py_AdaptivePooling_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("AdaptivePooling", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(AdaptivePooling::typeinfo(), &py_type).second);
}

PyOpDefBegin(AddAxis) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(AddAxis)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(AddAxis)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(AddAxis)

int PyOp(AddAxis)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "scope", NULL};
    PyObject *axis = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &axis, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AddAxis)*>(self)->inst().axis =
                    py::cast<decltype(AddAxis::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(AddAxis)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(AddAxis, axis), py_set_generic(AddAxis, axis), const_cast<char*>("axis"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(AddAxis)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(AddAxis)::getstate, METH_NOARGS, "AddAxis getstate"},
    {const_cast<char*>("__setstate__"), PyOp(AddAxis)::setstate, METH_VARARGS, "AddAxis setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(AddAxis)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(AddAxis)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(AddAxis)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(AddAxis)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: list[int] = ...) -> None\n"
};

void _init_py_AddAxis(py::module m) {
    using py_op = PyOp(AddAxis);
    auto& py_type = PyOpType(AddAxis);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.AddAxis";
    py_type.tp_basicsize = sizeof(PyOp(AddAxis));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "AddAxis";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(AddAxis), &PyOp(AddAxis)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("AddAxis", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(AddAxis::typeinfo(), &py_type).second);
}

PyOpDefBegin(Argmax) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Argmax)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Argmax)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Argmax)

int PyOp(Argmax)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "scope", NULL};
    PyObject *axis = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &axis, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Argmax)*>(self)->inst().axis =
                    py::cast<decltype(Argmax::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Argmax)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Argmax, axis), py_set_generic(Argmax, axis), const_cast<char*>("axis"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Argmax)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Argmax)::getstate, METH_NOARGS, "Argmax getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Argmax)::setstate, METH_VARARGS, "Argmax setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Argmax)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Argmax)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Argmax)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Argmax)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ...) -> None\n"
};

void _init_py_Argmax(py::module m) {
    using py_op = PyOp(Argmax);
    auto& py_type = PyOpType(Argmax);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Argmax";
    py_type.tp_basicsize = sizeof(PyOp(Argmax));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Argmax";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Argmax), &PyOp(Argmax)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Argmax", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Argmax::typeinfo(), &py_type).second);
}

PyOpDefBegin(Argmin) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Argmin)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Argmin)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Argmin)

int PyOp(Argmin)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "scope", NULL};
    PyObject *axis = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &axis, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Argmin)*>(self)->inst().axis =
                    py::cast<decltype(Argmin::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Argmin)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Argmin, axis), py_set_generic(Argmin, axis), const_cast<char*>("axis"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Argmin)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Argmin)::getstate, METH_NOARGS, "Argmin getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Argmin)::setstate, METH_VARARGS, "Argmin setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Argmin)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Argmin)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Argmin)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Argmin)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ...) -> None\n"
};

void _init_py_Argmin(py::module m) {
    using py_op = PyOp(Argmin);
    auto& py_type = PyOpType(Argmin);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Argmin";
    py_type.tp_basicsize = sizeof(PyOp(Argmin));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Argmin";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Argmin), &PyOp(Argmin)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Argmin", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Argmin::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Argsort::Order> {
    static constexpr const char *name = "Argsort.Order";
    static constexpr std::underlying_type_t<Argsort::Order> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<Argsort::Order>::type = nullptr;

template<> const char*
EnumWrapper<Argsort::Order>::members[] = {"ASCENDING", "DESCENDING"};

template<> std::unordered_map<std::string, Argsort::Order>
EnumWrapper<Argsort::Order>::mem2value = {{normalize_enum("ASCENDING"), Argsort::Order::ASCENDING}, {normalize_enum("DESCENDING"), Argsort::Order::DESCENDING}};
template<> PyObject* EnumWrapper<Argsort::Order>::pyobj_insts[2] = {nullptr};

void _init_py_Argsort_Order(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Argsort::Order>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Argsort::Order>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Argsort::Order>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Argsort::Order>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Argsort.Order",
        // basicsize
        sizeof(EnumWrapper<Argsort::Order>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Order").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Argsort.Order").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Argsort::Order>*>(inst)->value = Argsort::Order::ASCENDING;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ASCENDING", inst) >= 0);
    EnumWrapper<Argsort::Order>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Argsort::Order>*>(inst)->value = Argsort::Order::DESCENDING;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DESCENDING", inst) >= 0);
    EnumWrapper<Argsort::Order>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Order", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Argsort) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Argsort)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"order", serialization<decltype(opdef.order)>::dump(opdef.order)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Argsort)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("order");
        if (iter != state.end()) {
            opdef.order = serialization<decltype(opdef.order)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Argsort)

int PyOp(Argsort)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"order", "scope", NULL};
    PyObject *order = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &order, &scope))
    return -1;

    if (order) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Argsort)*>(self)->inst().order =
                    py::cast<decltype(Argsort::order)>(py::handle(order));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Argsort)::py_getsetters[] = {
    {const_cast<char*>("order"), py_get_generic(Argsort, order), py_set_generic(Argsort, order), const_cast<char*>("order"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Argsort)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Argsort)::getstate, METH_NOARGS, "Argsort getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Argsort)::setstate, METH_VARARGS, "Argsort setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Argsort)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Argsort)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Argsort)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Argsort)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, order: Union[str, Order] = ...) -> None\n"
};

void _init_py_Argsort(py::module m) {
    using py_op = PyOp(Argsort);
    auto& py_type = PyOpType(Argsort);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Argsort";
    py_type.tp_basicsize = sizeof(PyOp(Argsort));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Argsort";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Argsort), &PyOp(Argsort)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Argsort_Order(py_type);

    PyType_Modified(&py_type);
    m.add_object("Argsort", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Argsort::typeinfo(), &py_type).second);
}

PyOpDefBegin(AssertEqual) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(AssertEqual)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"maxerr", serialization<decltype(opdef.maxerr)>::dump(opdef.maxerr)},
            {"verbose", serialization<decltype(opdef.verbose)>::dump(opdef.verbose)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(AssertEqual)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("maxerr");
        if (iter != state.end()) {
            opdef.maxerr = serialization<decltype(opdef.maxerr)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("verbose");
        if (iter != state.end()) {
            opdef.verbose = serialization<decltype(opdef.verbose)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(AssertEqual)

int PyOp(AssertEqual)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"maxerr", "verbose", "scope", NULL};
    PyObject *maxerr = NULL, *verbose = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &maxerr, &verbose, &scope))
    return -1;

    if (maxerr) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AssertEqual)*>(self)->inst().maxerr =
                    py::cast<decltype(AssertEqual::maxerr)>(py::handle(maxerr));
        } CATCH_ALL(-1)
    }

    if (verbose) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AssertEqual)*>(self)->inst().verbose =
                    py::cast<decltype(AssertEqual::verbose)>(py::handle(verbose));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(AssertEqual)::py_getsetters[] = {
    {const_cast<char*>("maxerr"), py_get_generic(AssertEqual, maxerr), py_set_generic(AssertEqual, maxerr), const_cast<char*>("maxerr"), NULL},
    {const_cast<char*>("verbose"), py_get_generic(AssertEqual, verbose), py_set_generic(AssertEqual, verbose), const_cast<char*>("verbose"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(AssertEqual)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(AssertEqual)::getstate, METH_NOARGS, "AssertEqual getstate"},
    {const_cast<char*>("__setstate__"), PyOp(AssertEqual)::setstate, METH_VARARGS, "AssertEqual setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(AssertEqual)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(AssertEqual)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(AssertEqual)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(AssertEqual)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, maxerr: float = ..., verbose: bool = ...) -> None\n"
};

void _init_py_AssertEqual(py::module m) {
    using py_op = PyOp(AssertEqual);
    auto& py_type = PyOpType(AssertEqual);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.AssertEqual";
    py_type.tp_basicsize = sizeof(PyOp(AssertEqual));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "AssertEqual";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(AssertEqual), &PyOp(AssertEqual)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("AssertEqual", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(AssertEqual::typeinfo(), &py_type).second);
}

PyOpDefBegin(AtlasRuntime) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(AtlasRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"buf", serialization<decltype(opdef.buf)>::dump(opdef.buf)},
            {"buf_size", serialization<decltype(opdef.buf_size)>::dump(opdef.buf_size)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(AtlasRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("buf");
        if (iter != state.end()) {
            opdef.buf = serialization<decltype(opdef.buf)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("buf_size");
        if (iter != state.end()) {
            opdef.buf_size = serialization<decltype(opdef.buf_size)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(AtlasRuntime)

int PyOp(AtlasRuntime)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"buf", "buf_size", "scope", NULL};
    PyObject *buf = NULL, *buf_size = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &buf, &buf_size, &scope))
    return -1;

    if (buf) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AtlasRuntime)*>(self)->inst().buf =
                    py::cast<decltype(AtlasRuntime::buf)>(py::handle(buf));
        } CATCH_ALL(-1)
    }

    if (buf_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(AtlasRuntime)*>(self)->inst().buf_size =
                    py::cast<decltype(AtlasRuntime::buf_size)>(py::handle(buf_size));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(AtlasRuntime)::py_getsetters[] = {
    {const_cast<char*>("buf"), py_get_generic(AtlasRuntime, buf), py_set_generic(AtlasRuntime, buf), const_cast<char*>("buf"), NULL},
    {const_cast<char*>("buf_size"), py_get_generic(AtlasRuntime, buf_size), py_set_generic(AtlasRuntime, buf_size), const_cast<char*>("buf_size"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(AtlasRuntime)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(AtlasRuntime)::getstate, METH_NOARGS, "AtlasRuntime getstate"},
    {const_cast<char*>("__setstate__"), PyOp(AtlasRuntime)::setstate, METH_VARARGS, "AtlasRuntime setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(AtlasRuntime)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(AtlasRuntime)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(AtlasRuntime)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(AtlasRuntime)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, buf: str = ..., buf_size: int = ...) -> None\n"
};

void _init_py_AtlasRuntime(py::module m) {
    using py_op = PyOp(AtlasRuntime);
    auto& py_type = PyOpType(AtlasRuntime);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.AtlasRuntime";
    py_type.tp_basicsize = sizeof(PyOp(AtlasRuntime));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "AtlasRuntime";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(AtlasRuntime), &PyOp(AtlasRuntime)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("AtlasRuntime", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(AtlasRuntime::typeinfo(), &py_type).second);
}

PyOpDefBegin(Barrier) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Barrier)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)},
            {"nr_outputs", serialization<decltype(opdef.nr_outputs)>::dump(opdef.nr_outputs)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Barrier)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("nr_outputs");
        if (iter != state.end()) {
            opdef.nr_outputs = serialization<decltype(opdef.nr_outputs)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Barrier)

int PyOp(Barrier)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"comp_node", "nr_outputs", "scope", NULL};
    PyObject *comp_node = NULL, *nr_outputs = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &comp_node, &nr_outputs, &scope))
    return -1;

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Barrier)*>(self)->inst().comp_node =
                    py::cast<decltype(Barrier::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (nr_outputs) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Barrier)*>(self)->inst().nr_outputs =
                    py::cast<decltype(Barrier::nr_outputs)>(py::handle(nr_outputs));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Barrier)::py_getsetters[] = {
    {const_cast<char*>("comp_node"), py_get_generic(Barrier, comp_node), py_set_generic(Barrier, comp_node), const_cast<char*>("comp_node"), NULL},
    {const_cast<char*>("nr_outputs"), py_get_generic(Barrier, nr_outputs), py_set_generic(Barrier, nr_outputs), const_cast<char*>("nr_outputs"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Barrier)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Barrier)::getstate, METH_NOARGS, "Barrier getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Barrier)::setstate, METH_VARARGS, "Barrier setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Barrier)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Barrier)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Barrier)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Barrier)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, comp_node: str = ..., nr_outputs: int = ...) -> None\n"
};

void _init_py_Barrier(py::module m) {
    using py_op = PyOp(Barrier);
    auto& py_type = PyOpType(Barrier);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Barrier";
    py_type.tp_basicsize = sizeof(PyOp(Barrier));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Barrier";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Barrier), &PyOp(Barrier)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Barrier", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Barrier::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<BatchConvBias::NonlineMode> {
    static constexpr const char *name = "BatchConvBias.NonlineMode";
    static constexpr std::underlying_type_t<BatchConvBias::NonlineMode> max = 4 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchConvBias::NonlineMode>::type = nullptr;

template<> const char*
EnumWrapper<BatchConvBias::NonlineMode>::members[] = {"IDENTITY", "RELU", "SIGMOID", "H_SWISH"};

template<> std::unordered_map<std::string, BatchConvBias::NonlineMode>
EnumWrapper<BatchConvBias::NonlineMode>::mem2value = {{normalize_enum("IDENTITY"), BatchConvBias::NonlineMode::IDENTITY}, {normalize_enum("RELU"), BatchConvBias::NonlineMode::RELU}, {normalize_enum("SIGMOID"), BatchConvBias::NonlineMode::SIGMOID}, {normalize_enum("H_SWISH"), BatchConvBias::NonlineMode::H_SWISH}};
template<> PyObject* EnumWrapper<BatchConvBias::NonlineMode>::pyobj_insts[4] = {nullptr};

void _init_py_BatchConvBias_NonlineMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchConvBias::NonlineMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchConvBias::NonlineMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchConvBias::NonlineMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchConvBias::NonlineMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchConvBias.NonlineMode",
        // basicsize
        sizeof(EnumWrapper<BatchConvBias::NonlineMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("NonlineMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchConvBias.NonlineMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::NonlineMode>*>(inst)->value = BatchConvBias::NonlineMode::IDENTITY;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "IDENTITY", inst) >= 0);
    EnumWrapper<BatchConvBias::NonlineMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::NonlineMode>*>(inst)->value = BatchConvBias::NonlineMode::RELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RELU", inst) >= 0);
    EnumWrapper<BatchConvBias::NonlineMode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::NonlineMode>*>(inst)->value = BatchConvBias::NonlineMode::SIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SIGMOID", inst) >= 0);
    EnumWrapper<BatchConvBias::NonlineMode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::NonlineMode>*>(inst)->value = BatchConvBias::NonlineMode::H_SWISH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "H_SWISH", inst) >= 0);
    EnumWrapper<BatchConvBias::NonlineMode>::pyobj_insts[3] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "NonlineMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<BatchConvBias::Mode> {
    static constexpr const char *name = "BatchConvBias.Mode";
    static constexpr std::underlying_type_t<BatchConvBias::Mode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchConvBias::Mode>::type = nullptr;

template<> const char*
EnumWrapper<BatchConvBias::Mode>::members[] = {"CROSS_CORRELATION", "CONVOLUTION"};

template<> std::unordered_map<std::string, BatchConvBias::Mode>
EnumWrapper<BatchConvBias::Mode>::mem2value = {{normalize_enum("CROSS_CORRELATION"), BatchConvBias::Mode::CROSS_CORRELATION}, {normalize_enum("CONVOLUTION"), BatchConvBias::Mode::CONVOLUTION}};
template<> PyObject* EnumWrapper<BatchConvBias::Mode>::pyobj_insts[2] = {nullptr};

void _init_py_BatchConvBias_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchConvBias::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchConvBias::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchConvBias::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchConvBias::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchConvBias.Mode",
        // basicsize
        sizeof(EnumWrapper<BatchConvBias::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchConvBias.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::Mode>*>(inst)->value = BatchConvBias::Mode::CROSS_CORRELATION;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CROSS_CORRELATION", inst) >= 0);
    EnumWrapper<BatchConvBias::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::Mode>*>(inst)->value = BatchConvBias::Mode::CONVOLUTION;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CONVOLUTION", inst) >= 0);
    EnumWrapper<BatchConvBias::Mode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<BatchConvBias::Sparse> {
    static constexpr const char *name = "BatchConvBias.Sparse";
    static constexpr std::underlying_type_t<BatchConvBias::Sparse> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchConvBias::Sparse>::type = nullptr;

template<> const char*
EnumWrapper<BatchConvBias::Sparse>::members[] = {"DENSE", "GROUP"};

template<> std::unordered_map<std::string, BatchConvBias::Sparse>
EnumWrapper<BatchConvBias::Sparse>::mem2value = {{normalize_enum("DENSE"), BatchConvBias::Sparse::DENSE}, {normalize_enum("GROUP"), BatchConvBias::Sparse::GROUP}};
template<> PyObject* EnumWrapper<BatchConvBias::Sparse>::pyobj_insts[2] = {nullptr};

void _init_py_BatchConvBias_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchConvBias::Sparse>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchConvBias::Sparse>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchConvBias::Sparse>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchConvBias::Sparse>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchConvBias.Sparse",
        // basicsize
        sizeof(EnumWrapper<BatchConvBias::Sparse>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Sparse").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchConvBias.Sparse").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::Sparse>*>(inst)->value = BatchConvBias::Sparse::DENSE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DENSE", inst) >= 0);
    EnumWrapper<BatchConvBias::Sparse>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::Sparse>*>(inst)->value = BatchConvBias::Sparse::GROUP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "GROUP", inst) >= 0);
    EnumWrapper<BatchConvBias::Sparse>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_BatchConvBias_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchConvBias::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<BatchConvBias::ComputeMode> {
    static constexpr const char *name = "BatchConvBias.ComputeMode";
    static constexpr std::underlying_type_t<BatchConvBias::ComputeMode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchConvBias::ComputeMode>::type = nullptr;

template<> const char*
EnumWrapper<BatchConvBias::ComputeMode>::members[] = {"DEFAULT", "FLOAT32"};

template<> std::unordered_map<std::string, BatchConvBias::ComputeMode>
EnumWrapper<BatchConvBias::ComputeMode>::mem2value = {{normalize_enum("DEFAULT"), BatchConvBias::ComputeMode::DEFAULT}, {normalize_enum("FLOAT32"), BatchConvBias::ComputeMode::FLOAT32}};
template<> PyObject* EnumWrapper<BatchConvBias::ComputeMode>::pyobj_insts[2] = {nullptr};

void _init_py_BatchConvBias_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchConvBias::ComputeMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchConvBias::ComputeMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchConvBias::ComputeMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchConvBias::ComputeMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchConvBias.ComputeMode",
        // basicsize
        sizeof(EnumWrapper<BatchConvBias::ComputeMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("ComputeMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchConvBias.ComputeMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::ComputeMode>*>(inst)->value = BatchConvBias::ComputeMode::DEFAULT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DEFAULT", inst) >= 0);
    EnumWrapper<BatchConvBias::ComputeMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchConvBias::ComputeMode>*>(inst)->value = BatchConvBias::ComputeMode::FLOAT32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT32", inst) >= 0);
    EnumWrapper<BatchConvBias::ComputeMode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<BatchConvBias::Strategy> {
    static constexpr const char *name = "BatchConvBias.Strategy";
    static constexpr std::underlying_type_t<BatchConvBias::Strategy> max = (1llu << 4) - 1;
};
template<> PyTypeObject* BitCombinedEnumWrapper<BatchConvBias::Strategy>::type = nullptr;

template<> const char*
BitCombinedEnumWrapper<BatchConvBias::Strategy>::members[] = {"HEURISTIC", "PROFILE", "REPRODUCIBLE", "OPTIMIZED"};

template<> std::unordered_map<std::string, BatchConvBias::Strategy>
BitCombinedEnumWrapper<BatchConvBias::Strategy>::mem2value = {{normalize_enum("HEURISTIC"), BatchConvBias::Strategy::HEURISTIC}, {normalize_enum("PROFILE"), BatchConvBias::Strategy::PROFILE}, {normalize_enum("REPRODUCIBLE"), BatchConvBias::Strategy::REPRODUCIBLE}, {normalize_enum("OPTIMIZED"), BatchConvBias::Strategy::OPTIMIZED}};
template<> PyObject* BitCombinedEnumWrapper<BatchConvBias::Strategy>::pyobj_insts[4] = {nullptr};

void _init_py_BatchConvBias_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<BatchConvBias::Strategy>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)BitCombinedEnumWrapper<BatchConvBias::Strategy>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)BitCombinedEnumWrapper<BatchConvBias::Strategy>::py_repr},
        {Py_tp_richcompare, (void*)BitCombinedEnumWrapper<BatchConvBias::Strategy>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {Py_tp_new, (void*)BitCombinedEnumWrapper<BatchConvBias::Strategy>::py_new_combined_enum},
        {Py_nb_or, (void*)BitCombinedEnumWrapper<BatchConvBias::Strategy>::py_or},
        {Py_nb_and, (void*)BitCombinedEnumWrapper<BatchConvBias::Strategy>::py_and},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchConvBias.Strategy",
        // basicsize
        sizeof(BitCombinedEnumWrapper<BatchConvBias::Strategy>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Strategy").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchConvBias.Strategy").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<BitCombinedEnumWrapper<BatchConvBias::Strategy>*>(inst)->value = BatchConvBias::Strategy::HEURISTIC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "HEURISTIC", inst) >= 0);
    BitCombinedEnumWrapper<BatchConvBias::Strategy>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<BitCombinedEnumWrapper<BatchConvBias::Strategy>*>(inst)->value = BatchConvBias::Strategy::PROFILE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "PROFILE", inst) >= 0);
    BitCombinedEnumWrapper<BatchConvBias::Strategy>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<BitCombinedEnumWrapper<BatchConvBias::Strategy>*>(inst)->value = BatchConvBias::Strategy::REPRODUCIBLE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REPRODUCIBLE", inst) >= 0);
    BitCombinedEnumWrapper<BatchConvBias::Strategy>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<BitCombinedEnumWrapper<BatchConvBias::Strategy>*>(inst)->value = BatchConvBias::Strategy::OPTIMIZED;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "OPTIMIZED", inst) >= 0);
    BitCombinedEnumWrapper<BatchConvBias::Strategy>::pyobj_insts[3] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(BatchConvBias) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"nonlineMode", serialization<decltype(opdef.nonlineMode)>::dump(opdef.nonlineMode)},
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("nonlineMode");
        if (iter != state.end()) {
            opdef.nonlineMode = serialization<decltype(opdef.nonlineMode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchConvBias)

int PyOp(BatchConvBias)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"nonlineMode", "mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "strategy", "workspace_limit", "dtype", "scope", NULL};
    PyObject *nonlineMode = NULL, *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *strategy = NULL, *workspace_limit = NULL, *dtype = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOOOO", const_cast<char**>(kwlist), &nonlineMode, &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &strategy, &workspace_limit, &dtype, &scope))
    return -1;

    if (nonlineMode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().nonlineMode =
                    py::cast<decltype(BatchConvBias::nonlineMode)>(py::handle(nonlineMode));
        } CATCH_ALL(-1)
    }

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().mode =
                    py::cast<decltype(BatchConvBias::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().pad_h =
                    py::cast<decltype(BatchConvBias::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().pad_w =
                    py::cast<decltype(BatchConvBias::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().stride_h =
                    py::cast<decltype(BatchConvBias::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().stride_w =
                    py::cast<decltype(BatchConvBias::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().dilate_h =
                    py::cast<decltype(BatchConvBias::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().dilate_w =
                    py::cast<decltype(BatchConvBias::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().sparse =
                    py::cast<decltype(BatchConvBias::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().format =
                    py::cast<decltype(BatchConvBias::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().compute_mode =
                    py::cast<decltype(BatchConvBias::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().strategy =
                    py::cast<decltype(BatchConvBias::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().workspace_limit =
                    py::cast<decltype(BatchConvBias::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchConvBias)*>(self)->inst().dtype =
                    py::cast<decltype(BatchConvBias::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchConvBias)::py_getsetters[] = {
    {const_cast<char*>("nonlineMode"), py_get_generic(BatchConvBias, nonlineMode), py_set_generic(BatchConvBias, nonlineMode), const_cast<char*>("nonlineMode"), NULL},
    {const_cast<char*>("mode"), py_get_generic(BatchConvBias, mode), py_set_generic(BatchConvBias, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(BatchConvBias, pad_h), py_set_generic(BatchConvBias, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(BatchConvBias, pad_w), py_set_generic(BatchConvBias, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(BatchConvBias, stride_h), py_set_generic(BatchConvBias, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(BatchConvBias, stride_w), py_set_generic(BatchConvBias, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(BatchConvBias, dilate_h), py_set_generic(BatchConvBias, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(BatchConvBias, dilate_w), py_set_generic(BatchConvBias, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(BatchConvBias, sparse), py_set_generic(BatchConvBias, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(BatchConvBias, format), py_set_generic(BatchConvBias, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(BatchConvBias, compute_mode), py_set_generic(BatchConvBias, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(BatchConvBias, strategy), py_set_generic(BatchConvBias, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(BatchConvBias, workspace_limit), py_set_generic(BatchConvBias, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(BatchConvBias, dtype), py_set_generic(BatchConvBias, dtype), const_cast<char*>("dtype"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchConvBias)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchConvBias)::getstate, METH_NOARGS, "BatchConvBias getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchConvBias)::setstate, METH_VARARGS, "BatchConvBias setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchConvBias)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchConvBias)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchConvBias)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchConvBias)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, nonlineMode: Union[str, NonlineMode] = ..., mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ..., strategy: Union[str, Strategy] = ..., dtype: str = ...) -> None\n"
};

void _init_py_BatchConvBias(py::module m) {
    using py_op = PyOp(BatchConvBias);
    auto& py_type = PyOpType(BatchConvBias);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchConvBias";
    py_type.tp_basicsize = sizeof(PyOp(BatchConvBias));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchConvBias";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchConvBias), &PyOp(BatchConvBias)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_BatchConvBias_NonlineMode(py_type);
    _init_py_BatchConvBias_Mode(py_type);
    _init_py_BatchConvBias_Sparse(py_type);
    _init_py_BatchConvBias_Format(py_type);
    _init_py_BatchConvBias_ComputeMode(py_type);
    _init_py_BatchConvBias_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("BatchConvBias", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchConvBias::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<BatchNorm::ParamDim> {
    static constexpr const char *name = "BatchNorm.ParamDim";
    static constexpr std::underlying_type_t<BatchNorm::ParamDim> max = 4 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchNorm::ParamDim>::type = nullptr;

template<> const char*
EnumWrapper<BatchNorm::ParamDim>::members[] = {"DIM_11HW", "DIM_1CHW", "DIM_1C11", "DIM_111C"};

template<> std::unordered_map<std::string, BatchNorm::ParamDim>
EnumWrapper<BatchNorm::ParamDim>::mem2value = {{normalize_enum("DIM_11HW"), BatchNorm::ParamDim::DIM_11HW}, {normalize_enum("DIM_1CHW"), BatchNorm::ParamDim::DIM_1CHW}, {normalize_enum("DIM_1C11"), BatchNorm::ParamDim::DIM_1C11}, {normalize_enum("DIM_111C"), BatchNorm::ParamDim::DIM_111C}};
template<> PyObject* EnumWrapper<BatchNorm::ParamDim>::pyobj_insts[4] = {nullptr};

void _init_py_BatchNorm_ParamDim(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchNorm::ParamDim>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchNorm::ParamDim>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchNorm::ParamDim>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchNorm::ParamDim>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchNorm.ParamDim",
        // basicsize
        sizeof(EnumWrapper<BatchNorm::ParamDim>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("ParamDim").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchNorm.ParamDim").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchNorm::ParamDim>*>(inst)->value = BatchNorm::ParamDim::DIM_11HW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DIM_11HW", inst) >= 0);
    EnumWrapper<BatchNorm::ParamDim>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchNorm::ParamDim>*>(inst)->value = BatchNorm::ParamDim::DIM_1CHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DIM_1CHW", inst) >= 0);
    EnumWrapper<BatchNorm::ParamDim>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchNorm::ParamDim>*>(inst)->value = BatchNorm::ParamDim::DIM_1C11;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DIM_1C11", inst) >= 0);
    EnumWrapper<BatchNorm::ParamDim>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchNorm::ParamDim>*>(inst)->value = BatchNorm::ParamDim::DIM_111C;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DIM_111C", inst) >= 0);
    EnumWrapper<BatchNorm::ParamDim>::pyobj_insts[3] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ParamDim", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<BatchNorm::FwdMode> {
    static constexpr const char *name = "BatchNorm.FwdMode";
    static constexpr std::underlying_type_t<BatchNorm::FwdMode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchNorm::FwdMode>::type = nullptr;

template<> const char*
EnumWrapper<BatchNorm::FwdMode>::members[] = {"TRAINING", "INFERENCE"};

template<> std::unordered_map<std::string, BatchNorm::FwdMode>
EnumWrapper<BatchNorm::FwdMode>::mem2value = {{normalize_enum("TRAINING"), BatchNorm::FwdMode::TRAINING}, {normalize_enum("INFERENCE"), BatchNorm::FwdMode::INFERENCE}};
template<> PyObject* EnumWrapper<BatchNorm::FwdMode>::pyobj_insts[2] = {nullptr};

void _init_py_BatchNorm_FwdMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchNorm::FwdMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchNorm::FwdMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchNorm::FwdMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchNorm::FwdMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchNorm.FwdMode",
        // basicsize
        sizeof(EnumWrapper<BatchNorm::FwdMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("FwdMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchNorm.FwdMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchNorm::FwdMode>*>(inst)->value = BatchNorm::FwdMode::TRAINING;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TRAINING", inst) >= 0);
    EnumWrapper<BatchNorm::FwdMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchNorm::FwdMode>*>(inst)->value = BatchNorm::FwdMode::INFERENCE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "INFERENCE", inst) >= 0);
    EnumWrapper<BatchNorm::FwdMode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "FwdMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(BatchNorm) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchNorm)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"param_dim", serialization<decltype(opdef.param_dim)>::dump(opdef.param_dim)},
            {"fwd_mode", serialization<decltype(opdef.fwd_mode)>::dump(opdef.fwd_mode)},
            {"epsilon", serialization<decltype(opdef.epsilon)>::dump(opdef.epsilon)},
            {"avg_factor", serialization<decltype(opdef.avg_factor)>::dump(opdef.avg_factor)},
            {"scale", serialization<decltype(opdef.scale)>::dump(opdef.scale)},
            {"bias", serialization<decltype(opdef.bias)>::dump(opdef.bias)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchNorm)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("param_dim");
        if (iter != state.end()) {
            opdef.param_dim = serialization<decltype(opdef.param_dim)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("fwd_mode");
        if (iter != state.end()) {
            opdef.fwd_mode = serialization<decltype(opdef.fwd_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("epsilon");
        if (iter != state.end()) {
            opdef.epsilon = serialization<decltype(opdef.epsilon)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("avg_factor");
        if (iter != state.end()) {
            opdef.avg_factor = serialization<decltype(opdef.avg_factor)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("scale");
        if (iter != state.end()) {
            opdef.scale = serialization<decltype(opdef.scale)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bias");
        if (iter != state.end()) {
            opdef.bias = serialization<decltype(opdef.bias)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchNorm)

int PyOp(BatchNorm)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"param_dim", "fwd_mode", "epsilon", "avg_factor", "scale", "bias", "scope", NULL};
    PyObject *param_dim = NULL, *fwd_mode = NULL, *epsilon = NULL, *avg_factor = NULL, *scale = NULL, *bias = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOO", const_cast<char**>(kwlist), &param_dim, &fwd_mode, &epsilon, &avg_factor, &scale, &bias, &scope))
    return -1;

    if (param_dim) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNorm)*>(self)->inst().param_dim =
                    py::cast<decltype(BatchNorm::param_dim)>(py::handle(param_dim));
        } CATCH_ALL(-1)
    }

    if (fwd_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNorm)*>(self)->inst().fwd_mode =
                    py::cast<decltype(BatchNorm::fwd_mode)>(py::handle(fwd_mode));
        } CATCH_ALL(-1)
    }

    if (epsilon) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNorm)*>(self)->inst().epsilon =
                    py::cast<decltype(BatchNorm::epsilon)>(py::handle(epsilon));
        } CATCH_ALL(-1)
    }

    if (avg_factor) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNorm)*>(self)->inst().avg_factor =
                    py::cast<decltype(BatchNorm::avg_factor)>(py::handle(avg_factor));
        } CATCH_ALL(-1)
    }

    if (scale) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNorm)*>(self)->inst().scale =
                    py::cast<decltype(BatchNorm::scale)>(py::handle(scale));
        } CATCH_ALL(-1)
    }

    if (bias) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNorm)*>(self)->inst().bias =
                    py::cast<decltype(BatchNorm::bias)>(py::handle(bias));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchNorm)::py_getsetters[] = {
    {const_cast<char*>("param_dim"), py_get_generic(BatchNorm, param_dim), py_set_generic(BatchNorm, param_dim), const_cast<char*>("param_dim"), NULL},
    {const_cast<char*>("fwd_mode"), py_get_generic(BatchNorm, fwd_mode), py_set_generic(BatchNorm, fwd_mode), const_cast<char*>("fwd_mode"), NULL},
    {const_cast<char*>("epsilon"), py_get_generic(BatchNorm, epsilon), py_set_generic(BatchNorm, epsilon), const_cast<char*>("epsilon"), NULL},
    {const_cast<char*>("avg_factor"), py_get_generic(BatchNorm, avg_factor), py_set_generic(BatchNorm, avg_factor), const_cast<char*>("avg_factor"), NULL},
    {const_cast<char*>("scale"), py_get_generic(BatchNorm, scale), py_set_generic(BatchNorm, scale), const_cast<char*>("scale"), NULL},
    {const_cast<char*>("bias"), py_get_generic(BatchNorm, bias), py_set_generic(BatchNorm, bias), const_cast<char*>("bias"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchNorm)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchNorm)::getstate, METH_NOARGS, "BatchNorm getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchNorm)::setstate, METH_VARARGS, "BatchNorm setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchNorm)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchNorm)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchNorm)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchNorm)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, param_dim: Union[str, ParamDim] = ..., fwd_mode: Union[str, FwdMode] = ..., epsilon: float = ..., avg_factor: float = ..., scale: float = ..., bias: float = ...) -> None\n"
};

void _init_py_BatchNorm(py::module m) {
    using py_op = PyOp(BatchNorm);
    auto& py_type = PyOpType(BatchNorm);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchNorm";
    py_type.tp_basicsize = sizeof(PyOp(BatchNorm));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchNorm";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchNorm), &PyOp(BatchNorm)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_BatchNorm_ParamDim(py_type);
    _init_py_BatchNorm_FwdMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("BatchNorm", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchNorm::typeinfo(), &py_type).second);
}

void _init_py_BatchNormBackward_ParamDim(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchNormBackward::ParamDim>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ParamDim", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_BatchNormBackward_FwdMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchNormBackward::FwdMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "FwdMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(BatchNormBackward) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"param_dim", serialization<decltype(opdef.param_dim)>::dump(opdef.param_dim)},
            {"fwd_mode", serialization<decltype(opdef.fwd_mode)>::dump(opdef.fwd_mode)},
            {"epsilon", serialization<decltype(opdef.epsilon)>::dump(opdef.epsilon)},
            {"avg_factor", serialization<decltype(opdef.avg_factor)>::dump(opdef.avg_factor)},
            {"scale", serialization<decltype(opdef.scale)>::dump(opdef.scale)},
            {"bias", serialization<decltype(opdef.bias)>::dump(opdef.bias)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("param_dim");
        if (iter != state.end()) {
            opdef.param_dim = serialization<decltype(opdef.param_dim)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("fwd_mode");
        if (iter != state.end()) {
            opdef.fwd_mode = serialization<decltype(opdef.fwd_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("epsilon");
        if (iter != state.end()) {
            opdef.epsilon = serialization<decltype(opdef.epsilon)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("avg_factor");
        if (iter != state.end()) {
            opdef.avg_factor = serialization<decltype(opdef.avg_factor)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("scale");
        if (iter != state.end()) {
            opdef.scale = serialization<decltype(opdef.scale)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bias");
        if (iter != state.end()) {
            opdef.bias = serialization<decltype(opdef.bias)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchNormBackward)

int PyOp(BatchNormBackward)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"param_dim", "fwd_mode", "epsilon", "avg_factor", "scale", "bias", "scope", NULL};
    PyObject *param_dim = NULL, *fwd_mode = NULL, *epsilon = NULL, *avg_factor = NULL, *scale = NULL, *bias = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOO", const_cast<char**>(kwlist), &param_dim, &fwd_mode, &epsilon, &avg_factor, &scale, &bias, &scope))
    return -1;

    if (param_dim) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst().param_dim =
                    py::cast<decltype(BatchNormBackward::param_dim)>(py::handle(param_dim));
        } CATCH_ALL(-1)
    }

    if (fwd_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst().fwd_mode =
                    py::cast<decltype(BatchNormBackward::fwd_mode)>(py::handle(fwd_mode));
        } CATCH_ALL(-1)
    }

    if (epsilon) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst().epsilon =
                    py::cast<decltype(BatchNormBackward::epsilon)>(py::handle(epsilon));
        } CATCH_ALL(-1)
    }

    if (avg_factor) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst().avg_factor =
                    py::cast<decltype(BatchNormBackward::avg_factor)>(py::handle(avg_factor));
        } CATCH_ALL(-1)
    }

    if (scale) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst().scale =
                    py::cast<decltype(BatchNormBackward::scale)>(py::handle(scale));
        } CATCH_ALL(-1)
    }

    if (bias) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchNormBackward)*>(self)->inst().bias =
                    py::cast<decltype(BatchNormBackward::bias)>(py::handle(bias));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchNormBackward)::py_getsetters[] = {
    {const_cast<char*>("param_dim"), py_get_generic(BatchNormBackward, param_dim), py_set_generic(BatchNormBackward, param_dim), const_cast<char*>("param_dim"), NULL},
    {const_cast<char*>("fwd_mode"), py_get_generic(BatchNormBackward, fwd_mode), py_set_generic(BatchNormBackward, fwd_mode), const_cast<char*>("fwd_mode"), NULL},
    {const_cast<char*>("epsilon"), py_get_generic(BatchNormBackward, epsilon), py_set_generic(BatchNormBackward, epsilon), const_cast<char*>("epsilon"), NULL},
    {const_cast<char*>("avg_factor"), py_get_generic(BatchNormBackward, avg_factor), py_set_generic(BatchNormBackward, avg_factor), const_cast<char*>("avg_factor"), NULL},
    {const_cast<char*>("scale"), py_get_generic(BatchNormBackward, scale), py_set_generic(BatchNormBackward, scale), const_cast<char*>("scale"), NULL},
    {const_cast<char*>("bias"), py_get_generic(BatchNormBackward, bias), py_set_generic(BatchNormBackward, bias), const_cast<char*>("bias"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchNormBackward)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchNormBackward)::getstate, METH_NOARGS, "BatchNormBackward getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchNormBackward)::setstate, METH_VARARGS, "BatchNormBackward setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchNormBackward)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchNormBackward)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchNormBackward)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchNormBackward)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, param_dim: Union[str, ParamDim] = ..., fwd_mode: Union[str, FwdMode] = ..., epsilon: float = ..., avg_factor: float = ..., scale: float = ..., bias: float = ...) -> None\n"
};

void _init_py_BatchNormBackward(py::module m) {
    using py_op = PyOp(BatchNormBackward);
    auto& py_type = PyOpType(BatchNormBackward);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchNormBackward";
    py_type.tp_basicsize = sizeof(PyOp(BatchNormBackward));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchNormBackward";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchNormBackward), &PyOp(BatchNormBackward)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_BatchNormBackward_ParamDim(py_type);
    _init_py_BatchNormBackward_FwdMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("BatchNormBackward", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchNormBackward::typeinfo(), &py_type).second);
}

PyOpDefBegin(BatchedIncrMeshIndexing) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchedIncrMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchedIncrMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchedIncrMeshIndexing)

int PyOp(BatchedIncrMeshIndexing)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedIncrMeshIndexing)*>(self)->inst().items =
                    py::cast<decltype(BatchedIncrMeshIndexing::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchedIncrMeshIndexing)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(BatchedIncrMeshIndexing, items), py_set_generic(BatchedIncrMeshIndexing, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchedIncrMeshIndexing)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchedIncrMeshIndexing)::getstate, METH_NOARGS, "BatchedIncrMeshIndexing getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchedIncrMeshIndexing)::setstate, METH_VARARGS, "BatchedIncrMeshIndexing setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchedIncrMeshIndexing)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchedIncrMeshIndexing)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchedIncrMeshIndexing)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchedIncrMeshIndexing)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_BatchedIncrMeshIndexing(py::module m) {
    using py_op = PyOp(BatchedIncrMeshIndexing);
    auto& py_type = PyOpType(BatchedIncrMeshIndexing);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchedIncrMeshIndexing";
    py_type.tp_basicsize = sizeof(PyOp(BatchedIncrMeshIndexing));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchedIncrMeshIndexing";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchedIncrMeshIndexing), &PyOp(BatchedIncrMeshIndexing)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("BatchedIncrMeshIndexing", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchedIncrMeshIndexing::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<BatchedMatrixMul::ComputeMode> {
    static constexpr const char *name = "BatchedMatrixMul.ComputeMode";
    static constexpr std::underlying_type_t<BatchedMatrixMul::ComputeMode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchedMatrixMul::ComputeMode>::type = nullptr;

template<> const char*
EnumWrapper<BatchedMatrixMul::ComputeMode>::members[] = {"DEFAULT", "FLOAT32"};

template<> std::unordered_map<std::string, BatchedMatrixMul::ComputeMode>
EnumWrapper<BatchedMatrixMul::ComputeMode>::mem2value = {{normalize_enum("DEFAULT"), BatchedMatrixMul::ComputeMode::DEFAULT}, {normalize_enum("FLOAT32"), BatchedMatrixMul::ComputeMode::FLOAT32}};
template<> PyObject* EnumWrapper<BatchedMatrixMul::ComputeMode>::pyobj_insts[2] = {nullptr};

void _init_py_BatchedMatrixMul_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchedMatrixMul::ComputeMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchedMatrixMul::ComputeMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchedMatrixMul::ComputeMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchedMatrixMul::ComputeMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchedMatrixMul.ComputeMode",
        // basicsize
        sizeof(EnumWrapper<BatchedMatrixMul::ComputeMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("ComputeMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchedMatrixMul.ComputeMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::ComputeMode>*>(inst)->value = BatchedMatrixMul::ComputeMode::DEFAULT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DEFAULT", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::ComputeMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::ComputeMode>*>(inst)->value = BatchedMatrixMul::ComputeMode::FLOAT32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT32", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::ComputeMode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<BatchedMatrixMul::Format> {
    static constexpr const char *name = "BatchedMatrixMul.Format";
    static constexpr std::underlying_type_t<BatchedMatrixMul::Format> max = 5 - 1;
};
template<> PyTypeObject* EnumWrapper<BatchedMatrixMul::Format>::type = nullptr;

template<> const char*
EnumWrapper<BatchedMatrixMul::Format>::members[] = {"DEFAULT", "MK4", "MK8", "MK4_DOT", "N32K4_DOT"};

template<> std::unordered_map<std::string, BatchedMatrixMul::Format>
EnumWrapper<BatchedMatrixMul::Format>::mem2value = {{normalize_enum("DEFAULT"), BatchedMatrixMul::Format::DEFAULT}, {normalize_enum("MK4"), BatchedMatrixMul::Format::MK4}, {normalize_enum("MK8"), BatchedMatrixMul::Format::MK8}, {normalize_enum("MK4_DOT"), BatchedMatrixMul::Format::MK4_DOT}, {normalize_enum("N32K4_DOT"), BatchedMatrixMul::Format::N32K4_DOT}};
template<> PyObject* EnumWrapper<BatchedMatrixMul::Format>::pyobj_insts[5] = {nullptr};

void _init_py_BatchedMatrixMul_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<BatchedMatrixMul::Format>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<BatchedMatrixMul::Format>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<BatchedMatrixMul::Format>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<BatchedMatrixMul::Format>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.BatchedMatrixMul.Format",
        // basicsize
        sizeof(EnumWrapper<BatchedMatrixMul::Format>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Format").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("BatchedMatrixMul.Format").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::Format>*>(inst)->value = BatchedMatrixMul::Format::DEFAULT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DEFAULT", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::Format>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::Format>*>(inst)->value = BatchedMatrixMul::Format::MK4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MK4", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::Format>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::Format>*>(inst)->value = BatchedMatrixMul::Format::MK8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MK8", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::Format>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::Format>*>(inst)->value = BatchedMatrixMul::Format::MK4_DOT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MK4_DOT", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::Format>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<BatchedMatrixMul::Format>*>(inst)->value = BatchedMatrixMul::Format::N32K4_DOT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "N32K4_DOT", inst) >= 0);
    EnumWrapper<BatchedMatrixMul::Format>::pyobj_insts[4] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_BatchedMatrixMul_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<BatchedMatrixMul::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(BatchedMatrixMul) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"transposeA", serialization<decltype(opdef.transposeA)>::dump(opdef.transposeA)},
            {"transposeB", serialization<decltype(opdef.transposeB)>::dump(opdef.transposeB)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)},
            {"dimA", serialization<decltype(opdef.dimA)>::dump(opdef.dimA)},
            {"dimB", serialization<decltype(opdef.dimB)>::dump(opdef.dimB)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("transposeA");
        if (iter != state.end()) {
            opdef.transposeA = serialization<decltype(opdef.transposeA)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("transposeB");
        if (iter != state.end()) {
            opdef.transposeB = serialization<decltype(opdef.transposeB)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dimA");
        if (iter != state.end()) {
            opdef.dimA = serialization<decltype(opdef.dimA)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dimB");
        if (iter != state.end()) {
            opdef.dimB = serialization<decltype(opdef.dimB)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchedMatrixMul)

int PyOp(BatchedMatrixMul)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"transposeA", "transposeB", "compute_mode", "format", "strategy", "workspace_limit", "dimA", "dimB", "scope", NULL};
    PyObject *transposeA = NULL, *transposeB = NULL, *compute_mode = NULL, *format = NULL, *strategy = NULL, *workspace_limit = NULL, *dimA = NULL, *dimB = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", const_cast<char**>(kwlist), &transposeA, &transposeB, &compute_mode, &format, &strategy, &workspace_limit, &dimA, &dimB, &scope))
    return -1;

    if (transposeA) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().transposeA =
                    py::cast<decltype(BatchedMatrixMul::transposeA)>(py::handle(transposeA));
        } CATCH_ALL(-1)
    }

    if (transposeB) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().transposeB =
                    py::cast<decltype(BatchedMatrixMul::transposeB)>(py::handle(transposeB));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().compute_mode =
                    py::cast<decltype(BatchedMatrixMul::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().format =
                    py::cast<decltype(BatchedMatrixMul::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().strategy =
                    py::cast<decltype(BatchedMatrixMul::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().workspace_limit =
                    py::cast<decltype(BatchedMatrixMul::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (dimA) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().dimA =
                    py::cast<decltype(BatchedMatrixMul::dimA)>(py::handle(dimA));
        } CATCH_ALL(-1)
    }

    if (dimB) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMatrixMul)*>(self)->inst().dimB =
                    py::cast<decltype(BatchedMatrixMul::dimB)>(py::handle(dimB));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchedMatrixMul)::py_getsetters[] = {
    {const_cast<char*>("transposeA"), py_get_generic(BatchedMatrixMul, transposeA), py_set_generic(BatchedMatrixMul, transposeA), const_cast<char*>("transposeA"), NULL},
    {const_cast<char*>("transposeB"), py_get_generic(BatchedMatrixMul, transposeB), py_set_generic(BatchedMatrixMul, transposeB), const_cast<char*>("transposeB"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(BatchedMatrixMul, compute_mode), py_set_generic(BatchedMatrixMul, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("format"), py_get_generic(BatchedMatrixMul, format), py_set_generic(BatchedMatrixMul, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(BatchedMatrixMul, strategy), py_set_generic(BatchedMatrixMul, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(BatchedMatrixMul, workspace_limit), py_set_generic(BatchedMatrixMul, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {const_cast<char*>("dimA"), py_get_generic(BatchedMatrixMul, dimA), py_set_generic(BatchedMatrixMul, dimA), const_cast<char*>("dimA"), NULL},
    {const_cast<char*>("dimB"), py_get_generic(BatchedMatrixMul, dimB), py_set_generic(BatchedMatrixMul, dimB), const_cast<char*>("dimB"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchedMatrixMul)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchedMatrixMul)::getstate, METH_NOARGS, "BatchedMatrixMul getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchedMatrixMul)::setstate, METH_VARARGS, "BatchedMatrixMul setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchedMatrixMul)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchedMatrixMul)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchedMatrixMul)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchedMatrixMul)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, transposeA: bool = ..., transposeB: bool = ..., compute_mode: Union[str, ComputeMode] = ..., format: Union[str, Format] = ..., strategy: Union[str, Strategy] = ..., dimA: int = ..., dimB: int = ...) -> None\n"
};

void _init_py_BatchedMatrixMul(py::module m) {
    using py_op = PyOp(BatchedMatrixMul);
    auto& py_type = PyOpType(BatchedMatrixMul);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchedMatrixMul";
    py_type.tp_basicsize = sizeof(PyOp(BatchedMatrixMul));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchedMatrixMul";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchedMatrixMul), &PyOp(BatchedMatrixMul)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_BatchedMatrixMul_ComputeMode(py_type);
    _init_py_BatchedMatrixMul_Format(py_type);
    _init_py_BatchedMatrixMul_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("BatchedMatrixMul", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchedMatrixMul::typeinfo(), &py_type).second);
}

PyOpDefBegin(BatchedMeshIndexing) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchedMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchedMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchedMeshIndexing)

int PyOp(BatchedMeshIndexing)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedMeshIndexing)*>(self)->inst().items =
                    py::cast<decltype(BatchedMeshIndexing::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchedMeshIndexing)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(BatchedMeshIndexing, items), py_set_generic(BatchedMeshIndexing, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchedMeshIndexing)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchedMeshIndexing)::getstate, METH_NOARGS, "BatchedMeshIndexing getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchedMeshIndexing)::setstate, METH_VARARGS, "BatchedMeshIndexing setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchedMeshIndexing)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchedMeshIndexing)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchedMeshIndexing)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchedMeshIndexing)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_BatchedMeshIndexing(py::module m) {
    using py_op = PyOp(BatchedMeshIndexing);
    auto& py_type = PyOpType(BatchedMeshIndexing);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchedMeshIndexing";
    py_type.tp_basicsize = sizeof(PyOp(BatchedMeshIndexing));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchedMeshIndexing";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchedMeshIndexing), &PyOp(BatchedMeshIndexing)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("BatchedMeshIndexing", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchedMeshIndexing::typeinfo(), &py_type).second);
}

PyOpDefBegin(BatchedSetMeshIndexing) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BatchedSetMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BatchedSetMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BatchedSetMeshIndexing)

int PyOp(BatchedSetMeshIndexing)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BatchedSetMeshIndexing)*>(self)->inst().items =
                    py::cast<decltype(BatchedSetMeshIndexing::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BatchedSetMeshIndexing)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(BatchedSetMeshIndexing, items), py_set_generic(BatchedSetMeshIndexing, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BatchedSetMeshIndexing)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BatchedSetMeshIndexing)::getstate, METH_NOARGS, "BatchedSetMeshIndexing getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BatchedSetMeshIndexing)::setstate, METH_VARARGS, "BatchedSetMeshIndexing setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BatchedSetMeshIndexing)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BatchedSetMeshIndexing)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BatchedSetMeshIndexing)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BatchedSetMeshIndexing)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_BatchedSetMeshIndexing(py::module m) {
    using py_op = PyOp(BatchedSetMeshIndexing);
    auto& py_type = PyOpType(BatchedSetMeshIndexing);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BatchedSetMeshIndexing";
    py_type.tp_basicsize = sizeof(PyOp(BatchedSetMeshIndexing));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BatchedSetMeshIndexing";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BatchedSetMeshIndexing), &PyOp(BatchedSetMeshIndexing)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("BatchedSetMeshIndexing", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BatchedSetMeshIndexing::typeinfo(), &py_type).second);
}

PyOpDefBegin(BetaRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(BetaRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(BetaRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(BetaRNG)

int PyOp(BetaRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "handle", "scope", NULL};
    PyObject *seed = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &seed, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BetaRNG)*>(self)->inst().seed =
                    py::cast<decltype(BetaRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(BetaRNG)*>(self)->inst().handle =
                    py::cast<decltype(BetaRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(BetaRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(BetaRNG, seed), py_set_generic(BetaRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("handle"), py_get_generic(BetaRNG, handle), py_set_generic(BetaRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(BetaRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(BetaRNG)::getstate, METH_NOARGS, "BetaRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(BetaRNG)::setstate, METH_VARARGS, "BetaRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(BetaRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(BetaRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(BetaRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(BetaRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., handle: int = ...) -> None\n"
};

void _init_py_BetaRNG(py::module m) {
    using py_op = PyOp(BetaRNG);
    auto& py_type = PyOpType(BetaRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BetaRNG";
    py_type.tp_basicsize = sizeof(PyOp(BetaRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BetaRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(BetaRNG), &PyOp(BetaRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("BetaRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BetaRNG::typeinfo(), &py_type).second);
}

PyOpDefBegin(Borrow) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Borrow)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Borrow)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Borrow)

int PyOp(Borrow)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"comp_node", "scope", NULL};
    PyObject *comp_node = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &comp_node, &scope))
    return -1;

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Borrow)*>(self)->inst().comp_node =
                    py::cast<decltype(Borrow::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Borrow)::py_getsetters[] = {
    {const_cast<char*>("comp_node"), py_get_generic(Borrow, comp_node), py_set_generic(Borrow, comp_node), const_cast<char*>("comp_node"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Borrow)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Borrow)::getstate, METH_NOARGS, "Borrow getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Borrow)::setstate, METH_VARARGS, "Borrow setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Borrow)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Borrow)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Borrow)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Borrow)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, comp_node: str = ...) -> None\n"
};

void _init_py_Borrow(py::module m) {
    using py_op = PyOp(Borrow);
    auto& py_type = PyOpType(Borrow);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Borrow";
    py_type.tp_basicsize = sizeof(PyOp(Borrow));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Borrow";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Borrow), &PyOp(Borrow)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Borrow", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Borrow::typeinfo(), &py_type).second);
}

PyOpDefBegin(Broadcast) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Broadcast)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"shape", serialization<decltype(opdef.shape)>::dump(opdef.shape)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Broadcast)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("shape");
        if (iter != state.end()) {
            opdef.shape = serialization<decltype(opdef.shape)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Broadcast)

int PyOp(Broadcast)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"shape", "scope", NULL};
    PyObject *shape = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &shape, &scope))
    return -1;

    if (shape) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Broadcast)*>(self)->inst().shape =
                    py::cast<decltype(Broadcast::shape)>(py::handle(shape));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Broadcast)::py_getsetters[] = {
    {const_cast<char*>("shape"), py_get_generic(Broadcast, shape), py_set_generic(Broadcast, shape), const_cast<char*>("shape"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Broadcast)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Broadcast)::getstate, METH_NOARGS, "Broadcast getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Broadcast)::setstate, METH_VARARGS, "Broadcast setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Broadcast)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Broadcast)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Broadcast)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Broadcast)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, shape: list[int] = ...) -> None\n"
};

void _init_py_Broadcast(py::module m) {
    using py_op = PyOp(Broadcast);
    auto& py_type = PyOpType(Broadcast);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Broadcast";
    py_type.tp_basicsize = sizeof(PyOp(Broadcast));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Broadcast";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Broadcast), &PyOp(Broadcast)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Broadcast", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Broadcast::typeinfo(), &py_type).second);
}

PyOpDefBegin(CambriconRuntime) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(CambriconRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"buf", serialization<decltype(opdef.buf)>::dump(opdef.buf)},
            {"buf_size", serialization<decltype(opdef.buf_size)>::dump(opdef.buf_size)},
            {"symbol", serialization<decltype(opdef.symbol)>::dump(opdef.symbol)},
            {"tensor_dim_mutable", serialization<decltype(opdef.tensor_dim_mutable)>::dump(opdef.tensor_dim_mutable)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(CambriconRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("buf");
        if (iter != state.end()) {
            opdef.buf = serialization<decltype(opdef.buf)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("buf_size");
        if (iter != state.end()) {
            opdef.buf_size = serialization<decltype(opdef.buf_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("symbol");
        if (iter != state.end()) {
            opdef.symbol = serialization<decltype(opdef.symbol)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("tensor_dim_mutable");
        if (iter != state.end()) {
            opdef.tensor_dim_mutable = serialization<decltype(opdef.tensor_dim_mutable)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(CambriconRuntime)

int PyOp(CambriconRuntime)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"buf", "buf_size", "symbol", "tensor_dim_mutable", "scope", NULL};
    PyObject *buf = NULL, *buf_size = NULL, *symbol = NULL, *tensor_dim_mutable = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &buf, &buf_size, &symbol, &tensor_dim_mutable, &scope))
    return -1;

    if (buf) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CambriconRuntime)*>(self)->inst().buf =
                    py::cast<decltype(CambriconRuntime::buf)>(py::handle(buf));
        } CATCH_ALL(-1)
    }

    if (buf_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CambriconRuntime)*>(self)->inst().buf_size =
                    py::cast<decltype(CambriconRuntime::buf_size)>(py::handle(buf_size));
        } CATCH_ALL(-1)
    }

    if (symbol) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CambriconRuntime)*>(self)->inst().symbol =
                    py::cast<decltype(CambriconRuntime::symbol)>(py::handle(symbol));
        } CATCH_ALL(-1)
    }

    if (tensor_dim_mutable) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CambriconRuntime)*>(self)->inst().tensor_dim_mutable =
                    py::cast<decltype(CambriconRuntime::tensor_dim_mutable)>(py::handle(tensor_dim_mutable));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(CambriconRuntime)::py_getsetters[] = {
    {const_cast<char*>("buf"), py_get_generic(CambriconRuntime, buf), py_set_generic(CambriconRuntime, buf), const_cast<char*>("buf"), NULL},
    {const_cast<char*>("buf_size"), py_get_generic(CambriconRuntime, buf_size), py_set_generic(CambriconRuntime, buf_size), const_cast<char*>("buf_size"), NULL},
    {const_cast<char*>("symbol"), py_get_generic(CambriconRuntime, symbol), py_set_generic(CambriconRuntime, symbol), const_cast<char*>("symbol"), NULL},
    {const_cast<char*>("tensor_dim_mutable"), py_get_generic(CambriconRuntime, tensor_dim_mutable), py_set_generic(CambriconRuntime, tensor_dim_mutable), const_cast<char*>("tensor_dim_mutable"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(CambriconRuntime)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(CambriconRuntime)::getstate, METH_NOARGS, "CambriconRuntime getstate"},
    {const_cast<char*>("__setstate__"), PyOp(CambriconRuntime)::setstate, METH_VARARGS, "CambriconRuntime setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(CambriconRuntime)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(CambriconRuntime)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(CambriconRuntime)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(CambriconRuntime)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, buf: str = ..., buf_size: int = ..., symbol: str = ..., tensor_dim_mutable: bool = ...) -> None\n"
};

void _init_py_CambriconRuntime(py::module m) {
    using py_op = PyOp(CambriconRuntime);
    auto& py_type = PyOpType(CambriconRuntime);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.CambriconRuntime";
    py_type.tp_basicsize = sizeof(PyOp(CambriconRuntime));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "CambriconRuntime";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(CambriconRuntime), &PyOp(CambriconRuntime)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("CambriconRuntime", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(CambriconRuntime::typeinfo(), &py_type).second);
}

PyOpDefBegin(CheckNonFinite) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(CheckNonFinite)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"scale", serialization<decltype(opdef.scale)>::dump(opdef.scale)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(CheckNonFinite)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("scale");
        if (iter != state.end()) {
            opdef.scale = serialization<decltype(opdef.scale)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(CheckNonFinite)

int PyOp(CheckNonFinite)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"scale", "scope", NULL};
    PyObject *scale = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &scale, &scope))
    return -1;

    if (scale) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CheckNonFinite)*>(self)->inst().scale =
                    py::cast<decltype(CheckNonFinite::scale)>(py::handle(scale));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(CheckNonFinite)::py_getsetters[] = {
    {const_cast<char*>("scale"), py_get_generic(CheckNonFinite, scale), py_set_generic(CheckNonFinite, scale), const_cast<char*>("scale"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(CheckNonFinite)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(CheckNonFinite)::getstate, METH_NOARGS, "CheckNonFinite getstate"},
    {const_cast<char*>("__setstate__"), PyOp(CheckNonFinite)::setstate, METH_VARARGS, "CheckNonFinite setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(CheckNonFinite)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(CheckNonFinite)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(CheckNonFinite)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(CheckNonFinite)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, scale: float = ...) -> None\n"
};

void _init_py_CheckNonFinite(py::module m) {
    using py_op = PyOp(CheckNonFinite);
    auto& py_type = PyOpType(CheckNonFinite);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.CheckNonFinite";
    py_type.tp_basicsize = sizeof(PyOp(CheckNonFinite));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "CheckNonFinite";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(CheckNonFinite), &PyOp(CheckNonFinite)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("CheckNonFinite", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(CheckNonFinite::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<CollectiveComm::Mode> {
    static constexpr const char *name = "CollectiveComm.Mode";
    static constexpr std::underlying_type_t<CollectiveComm::Mode> max = 11 - 1;
};
template<> PyTypeObject* EnumWrapper<CollectiveComm::Mode>::type = nullptr;

template<> const char*
EnumWrapper<CollectiveComm::Mode>::members[] = {"REDUCE_SUM", "BROADCAST", "ALL_GATHER", "REDUCE_SCATTER_SUM", "ALL_REDUCE_SUM", "ALL_REDUCE_MAX", "ALL_REDUCE_MIN", "ALL_REDUCE_PROD", "GATHER", "SCATTER", "ALL_TO_ALL"};

template<> std::unordered_map<std::string, CollectiveComm::Mode>
EnumWrapper<CollectiveComm::Mode>::mem2value = {{normalize_enum("REDUCE_SUM"), CollectiveComm::Mode::REDUCE_SUM}, {normalize_enum("BROADCAST"), CollectiveComm::Mode::BROADCAST}, {normalize_enum("ALL_GATHER"), CollectiveComm::Mode::ALL_GATHER}, {normalize_enum("REDUCE_SCATTER_SUM"), CollectiveComm::Mode::REDUCE_SCATTER_SUM}, {normalize_enum("ALL_REDUCE_SUM"), CollectiveComm::Mode::ALL_REDUCE_SUM}, {normalize_enum("ALL_REDUCE_MAX"), CollectiveComm::Mode::ALL_REDUCE_MAX}, {normalize_enum("ALL_REDUCE_MIN"), CollectiveComm::Mode::ALL_REDUCE_MIN}, {normalize_enum("ALL_REDUCE_PROD"), CollectiveComm::Mode::ALL_REDUCE_PROD}, {normalize_enum("GATHER"), CollectiveComm::Mode::GATHER}, {normalize_enum("SCATTER"), CollectiveComm::Mode::SCATTER}, {normalize_enum("ALL_TO_ALL"), CollectiveComm::Mode::ALL_TO_ALL}};
template<> PyObject* EnumWrapper<CollectiveComm::Mode>::pyobj_insts[11] = {nullptr};

void _init_py_CollectiveComm_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<CollectiveComm::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<CollectiveComm::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<CollectiveComm::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<CollectiveComm::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.CollectiveComm.Mode",
        // basicsize
        sizeof(EnumWrapper<CollectiveComm::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("CollectiveComm.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::REDUCE_SUM;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REDUCE_SUM", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::BROADCAST;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BROADCAST", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::ALL_GATHER;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ALL_GATHER", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::REDUCE_SCATTER_SUM;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REDUCE_SCATTER_SUM", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::ALL_REDUCE_SUM;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ALL_REDUCE_SUM", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::ALL_REDUCE_MAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ALL_REDUCE_MAX", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::ALL_REDUCE_MIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ALL_REDUCE_MIN", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[6] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::ALL_REDUCE_PROD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ALL_REDUCE_PROD", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[7] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::GATHER;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "GATHER", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[8] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::SCATTER;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SCATTER", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[9] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CollectiveComm::Mode>*>(inst)->value = CollectiveComm::Mode::ALL_TO_ALL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ALL_TO_ALL", inst) >= 0);
    EnumWrapper<CollectiveComm::Mode>::pyobj_insts[10] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(CollectiveComm) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"key", serialization<decltype(opdef.key)>::dump(opdef.key)},
            {"nr_devices", serialization<decltype(opdef.nr_devices)>::dump(opdef.nr_devices)},
            {"rank", serialization<decltype(opdef.rank)>::dump(opdef.rank)},
            {"is_root", serialization<decltype(opdef.is_root)>::dump(opdef.is_root)},
            {"local_grad", serialization<decltype(opdef.local_grad)>::dump(opdef.local_grad)},
            {"addr", serialization<decltype(opdef.addr)>::dump(opdef.addr)},
            {"port", serialization<decltype(opdef.port)>::dump(opdef.port)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)},
            {"backend", serialization<decltype(opdef.backend)>::dump(opdef.backend)},
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("key");
        if (iter != state.end()) {
            opdef.key = serialization<decltype(opdef.key)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("nr_devices");
        if (iter != state.end()) {
            opdef.nr_devices = serialization<decltype(opdef.nr_devices)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("rank");
        if (iter != state.end()) {
            opdef.rank = serialization<decltype(opdef.rank)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("is_root");
        if (iter != state.end()) {
            opdef.is_root = serialization<decltype(opdef.is_root)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("local_grad");
        if (iter != state.end()) {
            opdef.local_grad = serialization<decltype(opdef.local_grad)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("addr");
        if (iter != state.end()) {
            opdef.addr = serialization<decltype(opdef.addr)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("port");
        if (iter != state.end()) {
            opdef.port = serialization<decltype(opdef.port)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("backend");
        if (iter != state.end()) {
            opdef.backend = serialization<decltype(opdef.backend)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(CollectiveComm)

int PyOp(CollectiveComm)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "key", "nr_devices", "rank", "is_root", "local_grad", "addr", "port", "dtype", "backend", "comp_node", "scope", NULL};
    PyObject *mode = NULL, *key = NULL, *nr_devices = NULL, *rank = NULL, *is_root = NULL, *local_grad = NULL, *addr = NULL, *port = NULL, *dtype = NULL, *backend = NULL, *comp_node = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &key, &nr_devices, &rank, &is_root, &local_grad, &addr, &port, &dtype, &backend, &comp_node, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().mode =
                    py::cast<decltype(CollectiveComm::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (key) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().key =
                    py::cast<decltype(CollectiveComm::key)>(py::handle(key));
        } CATCH_ALL(-1)
    }

    if (nr_devices) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().nr_devices =
                    py::cast<decltype(CollectiveComm::nr_devices)>(py::handle(nr_devices));
        } CATCH_ALL(-1)
    }

    if (rank) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().rank =
                    py::cast<decltype(CollectiveComm::rank)>(py::handle(rank));
        } CATCH_ALL(-1)
    }

    if (is_root) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().is_root =
                    py::cast<decltype(CollectiveComm::is_root)>(py::handle(is_root));
        } CATCH_ALL(-1)
    }

    if (local_grad) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().local_grad =
                    py::cast<decltype(CollectiveComm::local_grad)>(py::handle(local_grad));
        } CATCH_ALL(-1)
    }

    if (addr) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().addr =
                    py::cast<decltype(CollectiveComm::addr)>(py::handle(addr));
        } CATCH_ALL(-1)
    }

    if (port) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().port =
                    py::cast<decltype(CollectiveComm::port)>(py::handle(port));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().dtype =
                    py::cast<decltype(CollectiveComm::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (backend) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().backend =
                    py::cast<decltype(CollectiveComm::backend)>(py::handle(backend));
        } CATCH_ALL(-1)
    }

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CollectiveComm)*>(self)->inst().comp_node =
                    py::cast<decltype(CollectiveComm::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(CollectiveComm)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(CollectiveComm, mode), py_set_generic(CollectiveComm, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("key"), py_get_generic(CollectiveComm, key), py_set_generic(CollectiveComm, key), const_cast<char*>("key"), NULL},
    {const_cast<char*>("nr_devices"), py_get_generic(CollectiveComm, nr_devices), py_set_generic(CollectiveComm, nr_devices), const_cast<char*>("nr_devices"), NULL},
    {const_cast<char*>("rank"), py_get_generic(CollectiveComm, rank), py_set_generic(CollectiveComm, rank), const_cast<char*>("rank"), NULL},
    {const_cast<char*>("is_root"), py_get_generic(CollectiveComm, is_root), py_set_generic(CollectiveComm, is_root), const_cast<char*>("is_root"), NULL},
    {const_cast<char*>("local_grad"), py_get_generic(CollectiveComm, local_grad), py_set_generic(CollectiveComm, local_grad), const_cast<char*>("local_grad"), NULL},
    {const_cast<char*>("addr"), py_get_generic(CollectiveComm, addr), py_set_generic(CollectiveComm, addr), const_cast<char*>("addr"), NULL},
    {const_cast<char*>("port"), py_get_generic(CollectiveComm, port), py_set_generic(CollectiveComm, port), const_cast<char*>("port"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(CollectiveComm, dtype), py_set_generic(CollectiveComm, dtype), const_cast<char*>("dtype"), NULL},
    {const_cast<char*>("backend"), py_get_generic(CollectiveComm, backend), py_set_generic(CollectiveComm, backend), const_cast<char*>("backend"), NULL},
    {const_cast<char*>("comp_node"), py_get_generic(CollectiveComm, comp_node), py_set_generic(CollectiveComm, comp_node), const_cast<char*>("comp_node"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(CollectiveComm)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(CollectiveComm)::getstate, METH_NOARGS, "CollectiveComm getstate"},
    {const_cast<char*>("__setstate__"), PyOp(CollectiveComm)::setstate, METH_VARARGS, "CollectiveComm setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(CollectiveComm)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(CollectiveComm)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(CollectiveComm)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(CollectiveComm)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., key: str = ..., nr_devices: int = ..., rank: int = ..., is_root: bool = ..., local_grad: bool = ..., addr: str = ..., port: int = ..., dtype: str = ..., backend: str = ..., comp_node: str = ...) -> None\n"
};

void _init_py_CollectiveComm(py::module m) {
    using py_op = PyOp(CollectiveComm);
    auto& py_type = PyOpType(CollectiveComm);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.CollectiveComm";
    py_type.tp_basicsize = sizeof(PyOp(CollectiveComm));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "CollectiveComm";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(CollectiveComm), &PyOp(CollectiveComm)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_CollectiveComm_Mode(py_type);

    PyType_Modified(&py_type);
    m.add_object("CollectiveComm", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(CollectiveComm::typeinfo(), &py_type).second);
}

PyOpDefBegin(Concat) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Concat)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Concat)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Concat)

int PyOp(Concat)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "comp_node", "scope", NULL};
    PyObject *axis = NULL, *comp_node = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &axis, &comp_node, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Concat)*>(self)->inst().axis =
                    py::cast<decltype(Concat::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Concat)*>(self)->inst().comp_node =
                    py::cast<decltype(Concat::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Concat)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Concat, axis), py_set_generic(Concat, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("comp_node"), py_get_generic(Concat, comp_node), py_set_generic(Concat, comp_node), const_cast<char*>("comp_node"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Concat)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Concat)::getstate, METH_NOARGS, "Concat getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Concat)::setstate, METH_VARARGS, "Concat setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Concat)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Concat)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Concat)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Concat)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ..., comp_node: str = ...) -> None\n"
};

void _init_py_Concat(py::module m) {
    using py_op = PyOp(Concat);
    auto& py_type = PyOpType(Concat);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Concat";
    py_type.tp_basicsize = sizeof(PyOp(Concat));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Concat";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Concat), &PyOp(Concat)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Concat", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Concat::typeinfo(), &py_type).second);
}

PyOpDefBegin(CondTake) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(CondTake)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(CondTake)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(CondTake)

int PyOp(CondTake)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(CondTake)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(CondTake)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(CondTake)::getstate, METH_NOARGS, "CondTake getstate"},
    {const_cast<char*>("__setstate__"), PyOp(CondTake)::setstate, METH_VARARGS, "CondTake setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(CondTake)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(CondTake)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(CondTake)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(CondTake)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_CondTake(py::module m) {
    using py_op = PyOp(CondTake);
    auto& py_type = PyOpType(CondTake);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.CondTake";
    py_type.tp_basicsize = sizeof(PyOp(CondTake));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "CondTake";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(CondTake), &PyOp(CondTake)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("CondTake", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(CondTake::typeinfo(), &py_type).second);
}

void _init_py_ConvBias_NonlineMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvBias::NonlineMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "NonlineMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvBias_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvBias::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvBias_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvBias::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvBias_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvBias::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvBias_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvBias::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvBias_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<ConvBias::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(ConvBias) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ConvBias)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"nonlineMode", serialization<decltype(opdef.nonlineMode)>::dump(opdef.nonlineMode)},
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ConvBias)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("nonlineMode");
        if (iter != state.end()) {
            opdef.nonlineMode = serialization<decltype(opdef.nonlineMode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ConvBias)

int PyOp(ConvBias)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"nonlineMode", "mode", "sparse", "format", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "compute_mode", "strategy", "workspace_limit", "dtype", "scope", NULL};
    PyObject *nonlineMode = NULL, *mode = NULL, *sparse = NULL, *format = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *compute_mode = NULL, *strategy = NULL, *workspace_limit = NULL, *dtype = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOOOO", const_cast<char**>(kwlist), &nonlineMode, &mode, &sparse, &format, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &compute_mode, &strategy, &workspace_limit, &dtype, &scope))
    return -1;

    if (nonlineMode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().nonlineMode =
                    py::cast<decltype(ConvBias::nonlineMode)>(py::handle(nonlineMode));
        } CATCH_ALL(-1)
    }

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().mode =
                    py::cast<decltype(ConvBias::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().sparse =
                    py::cast<decltype(ConvBias::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().format =
                    py::cast<decltype(ConvBias::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().pad_h =
                    py::cast<decltype(ConvBias::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().pad_w =
                    py::cast<decltype(ConvBias::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().stride_h =
                    py::cast<decltype(ConvBias::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().stride_w =
                    py::cast<decltype(ConvBias::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().dilate_h =
                    py::cast<decltype(ConvBias::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().dilate_w =
                    py::cast<decltype(ConvBias::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().compute_mode =
                    py::cast<decltype(ConvBias::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().strategy =
                    py::cast<decltype(ConvBias::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().workspace_limit =
                    py::cast<decltype(ConvBias::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvBias)*>(self)->inst().dtype =
                    py::cast<decltype(ConvBias::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ConvBias)::py_getsetters[] = {
    {const_cast<char*>("nonlineMode"), py_get_generic(ConvBias, nonlineMode), py_set_generic(ConvBias, nonlineMode), const_cast<char*>("nonlineMode"), NULL},
    {const_cast<char*>("mode"), py_get_generic(ConvBias, mode), py_set_generic(ConvBias, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(ConvBias, sparse), py_set_generic(ConvBias, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(ConvBias, format), py_set_generic(ConvBias, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(ConvBias, pad_h), py_set_generic(ConvBias, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(ConvBias, pad_w), py_set_generic(ConvBias, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(ConvBias, stride_h), py_set_generic(ConvBias, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(ConvBias, stride_w), py_set_generic(ConvBias, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(ConvBias, dilate_h), py_set_generic(ConvBias, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(ConvBias, dilate_w), py_set_generic(ConvBias, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(ConvBias, compute_mode), py_set_generic(ConvBias, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(ConvBias, strategy), py_set_generic(ConvBias, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(ConvBias, workspace_limit), py_set_generic(ConvBias, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(ConvBias, dtype), py_set_generic(ConvBias, dtype), const_cast<char*>("dtype"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ConvBias)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ConvBias)::getstate, METH_NOARGS, "ConvBias getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ConvBias)::setstate, METH_VARARGS, "ConvBias setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ConvBias)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ConvBias)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ConvBias)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ConvBias)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, nonlineMode: Union[str, NonlineMode] = ..., mode: Union[str, Mode] = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., compute_mode: Union[str, ComputeMode] = ..., strategy: Union[str, Strategy] = ..., dtype: str = ...) -> None\n"
};

void _init_py_ConvBias(py::module m) {
    using py_op = PyOp(ConvBias);
    auto& py_type = PyOpType(ConvBias);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ConvBias";
    py_type.tp_basicsize = sizeof(PyOp(ConvBias));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ConvBias";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ConvBias), &PyOp(ConvBias)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_ConvBias_NonlineMode(py_type);
    _init_py_ConvBias_Mode(py_type);
    _init_py_ConvBias_Sparse(py_type);
    _init_py_ConvBias_Format(py_type);
    _init_py_ConvBias_ComputeMode(py_type);
    _init_py_ConvBias_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("ConvBias", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ConvBias::typeinfo(), &py_type).second);
}

void _init_py_Convolution_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<Convolution::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Convolution) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Convolution)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Convolution)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Convolution)

int PyOp(Convolution)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "strategy", "workspace_limit", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *strategy = NULL, *workspace_limit = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &strategy, &workspace_limit, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().mode =
                    py::cast<decltype(Convolution::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().pad_h =
                    py::cast<decltype(Convolution::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().pad_w =
                    py::cast<decltype(Convolution::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().stride_h =
                    py::cast<decltype(Convolution::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().stride_w =
                    py::cast<decltype(Convolution::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().dilate_h =
                    py::cast<decltype(Convolution::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().dilate_w =
                    py::cast<decltype(Convolution::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().sparse =
                    py::cast<decltype(Convolution::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().format =
                    py::cast<decltype(Convolution::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().compute_mode =
                    py::cast<decltype(Convolution::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().strategy =
                    py::cast<decltype(Convolution::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution)*>(self)->inst().workspace_limit =
                    py::cast<decltype(Convolution::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Convolution)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(Convolution, mode), py_set_generic(Convolution, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(Convolution, pad_h), py_set_generic(Convolution, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(Convolution, pad_w), py_set_generic(Convolution, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(Convolution, stride_h), py_set_generic(Convolution, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(Convolution, stride_w), py_set_generic(Convolution, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(Convolution, dilate_h), py_set_generic(Convolution, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(Convolution, dilate_w), py_set_generic(Convolution, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(Convolution, sparse), py_set_generic(Convolution, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(Convolution, format), py_set_generic(Convolution, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(Convolution, compute_mode), py_set_generic(Convolution, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(Convolution, strategy), py_set_generic(Convolution, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(Convolution, workspace_limit), py_set_generic(Convolution, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Convolution)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Convolution)::getstate, METH_NOARGS, "Convolution getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Convolution)::setstate, METH_VARARGS, "Convolution setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Convolution)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Convolution)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Convolution)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Convolution)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ..., strategy: Union[str, Strategy] = ...) -> None\n"
};

void _init_py_Convolution(py::module m) {
    using py_op = PyOp(Convolution);
    auto& py_type = PyOpType(Convolution);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Convolution";
    py_type.tp_basicsize = sizeof(PyOp(Convolution));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Convolution";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Convolution), &PyOp(Convolution)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Convolution_Mode(py_type);
    _init_py_Convolution_Sparse(py_type);
    _init_py_Convolution_Format(py_type);
    _init_py_Convolution_ComputeMode(py_type);
    _init_py_Convolution_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("Convolution", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Convolution::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Convolution3D::Mode> {
    static constexpr const char *name = "Convolution3D.Mode";
    static constexpr std::underlying_type_t<Convolution3D::Mode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<Convolution3D::Mode>::type = nullptr;

template<> const char*
EnumWrapper<Convolution3D::Mode>::members[] = {"CROSS_CORRELATION", "CONVOLUTION"};

template<> std::unordered_map<std::string, Convolution3D::Mode>
EnumWrapper<Convolution3D::Mode>::mem2value = {{normalize_enum("CROSS_CORRELATION"), Convolution3D::Mode::CROSS_CORRELATION}, {normalize_enum("CONVOLUTION"), Convolution3D::Mode::CONVOLUTION}};
template<> PyObject* EnumWrapper<Convolution3D::Mode>::pyobj_insts[2] = {nullptr};

void _init_py_Convolution3D_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3D::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Convolution3D::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Convolution3D::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Convolution3D::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Convolution3D.Mode",
        // basicsize
        sizeof(EnumWrapper<Convolution3D::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Convolution3D.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::Mode>*>(inst)->value = Convolution3D::Mode::CROSS_CORRELATION;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CROSS_CORRELATION", inst) >= 0);
    EnumWrapper<Convolution3D::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::Mode>*>(inst)->value = Convolution3D::Mode::CONVOLUTION;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CONVOLUTION", inst) >= 0);
    EnumWrapper<Convolution3D::Mode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<Convolution3D::Sparse> {
    static constexpr const char *name = "Convolution3D.Sparse";
    static constexpr std::underlying_type_t<Convolution3D::Sparse> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<Convolution3D::Sparse>::type = nullptr;

template<> const char*
EnumWrapper<Convolution3D::Sparse>::members[] = {"DENSE", "GROUP"};

template<> std::unordered_map<std::string, Convolution3D::Sparse>
EnumWrapper<Convolution3D::Sparse>::mem2value = {{normalize_enum("DENSE"), Convolution3D::Sparse::DENSE}, {normalize_enum("GROUP"), Convolution3D::Sparse::GROUP}};
template<> PyObject* EnumWrapper<Convolution3D::Sparse>::pyobj_insts[2] = {nullptr};

void _init_py_Convolution3D_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3D::Sparse>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Convolution3D::Sparse>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Convolution3D::Sparse>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Convolution3D::Sparse>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Convolution3D.Sparse",
        // basicsize
        sizeof(EnumWrapper<Convolution3D::Sparse>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Sparse").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Convolution3D.Sparse").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::Sparse>*>(inst)->value = Convolution3D::Sparse::DENSE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DENSE", inst) >= 0);
    EnumWrapper<Convolution3D::Sparse>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::Sparse>*>(inst)->value = Convolution3D::Sparse::GROUP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "GROUP", inst) >= 0);
    EnumWrapper<Convolution3D::Sparse>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<Convolution3D::DataType> {
    static constexpr const char *name = "Convolution3D.DataType";
    static constexpr std::underlying_type_t<Convolution3D::DataType> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<Convolution3D::DataType>::type = nullptr;

template<> const char*
EnumWrapper<Convolution3D::DataType>::members[] = {"FLOAT", "FLOAT_IO16xC32"};

template<> std::unordered_map<std::string, Convolution3D::DataType>
EnumWrapper<Convolution3D::DataType>::mem2value = {{normalize_enum("FLOAT"), Convolution3D::DataType::FLOAT}, {normalize_enum("FLOAT_IO16xC32"), Convolution3D::DataType::FLOAT_IO16xC32}};
template<> PyObject* EnumWrapper<Convolution3D::DataType>::pyobj_insts[2] = {nullptr};

void _init_py_Convolution3D_DataType(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3D::DataType>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Convolution3D::DataType>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Convolution3D::DataType>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Convolution3D::DataType>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Convolution3D.DataType",
        // basicsize
        sizeof(EnumWrapper<Convolution3D::DataType>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("DataType").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Convolution3D.DataType").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::DataType>*>(inst)->value = Convolution3D::DataType::FLOAT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT", inst) >= 0);
    EnumWrapper<Convolution3D::DataType>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::DataType>*>(inst)->value = Convolution3D::DataType::FLOAT_IO16xC32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT_IO16xC32", inst) >= 0);
    EnumWrapper<Convolution3D::DataType>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "DataType", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<Convolution3D::Format> {
    static constexpr const char *name = "Convolution3D.Format";
    static constexpr std::underlying_type_t<Convolution3D::Format> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<Convolution3D::Format>::type = nullptr;

template<> const char*
EnumWrapper<Convolution3D::Format>::members[] = {"NCDHW", "NDHWC"};

template<> std::unordered_map<std::string, Convolution3D::Format>
EnumWrapper<Convolution3D::Format>::mem2value = {{normalize_enum("NCDHW"), Convolution3D::Format::NCDHW}, {normalize_enum("NDHWC"), Convolution3D::Format::NDHWC}};
template<> PyObject* EnumWrapper<Convolution3D::Format>::pyobj_insts[2] = {nullptr};

void _init_py_Convolution3D_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3D::Format>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Convolution3D::Format>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Convolution3D::Format>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Convolution3D::Format>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Convolution3D.Format",
        // basicsize
        sizeof(EnumWrapper<Convolution3D::Format>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Format").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Convolution3D.Format").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::Format>*>(inst)->value = Convolution3D::Format::NCDHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCDHW", inst) >= 0);
    EnumWrapper<Convolution3D::Format>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Convolution3D::Format>*>(inst)->value = Convolution3D::Format::NDHWC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NDHWC", inst) >= 0);
    EnumWrapper<Convolution3D::Format>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution3D_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<Convolution3D::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Convolution3D) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Convolution3D)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_d", serialization<decltype(opdef.pad_d)>::dump(opdef.pad_d)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_d", serialization<decltype(opdef.stride_d)>::dump(opdef.stride_d)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_d", serialization<decltype(opdef.dilate_d)>::dump(opdef.dilate_d)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"data_type", serialization<decltype(opdef.data_type)>::dump(opdef.data_type)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Convolution3D)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_d");
        if (iter != state.end()) {
            opdef.pad_d = serialization<decltype(opdef.pad_d)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_d");
        if (iter != state.end()) {
            opdef.stride_d = serialization<decltype(opdef.stride_d)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_d");
        if (iter != state.end()) {
            opdef.dilate_d = serialization<decltype(opdef.dilate_d)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("data_type");
        if (iter != state.end()) {
            opdef.data_type = serialization<decltype(opdef.data_type)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Convolution3D)

int PyOp(Convolution3D)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_d", "pad_h", "pad_w", "stride_d", "stride_h", "stride_w", "dilate_d", "dilate_h", "dilate_w", "sparse", "data_type", "format", "strategy", "workspace_limit", "scope", NULL};
    PyObject *mode = NULL, *pad_d = NULL, *pad_h = NULL, *pad_w = NULL, *stride_d = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_d = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *data_type = NULL, *format = NULL, *strategy = NULL, *workspace_limit = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_d, &pad_h, &pad_w, &stride_d, &stride_h, &stride_w, &dilate_d, &dilate_h, &dilate_w, &sparse, &data_type, &format, &strategy, &workspace_limit, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().mode =
                    py::cast<decltype(Convolution3D::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_d) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().pad_d =
                    py::cast<decltype(Convolution3D::pad_d)>(py::handle(pad_d));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().pad_h =
                    py::cast<decltype(Convolution3D::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().pad_w =
                    py::cast<decltype(Convolution3D::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_d) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().stride_d =
                    py::cast<decltype(Convolution3D::stride_d)>(py::handle(stride_d));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().stride_h =
                    py::cast<decltype(Convolution3D::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().stride_w =
                    py::cast<decltype(Convolution3D::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_d) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().dilate_d =
                    py::cast<decltype(Convolution3D::dilate_d)>(py::handle(dilate_d));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().dilate_h =
                    py::cast<decltype(Convolution3D::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().dilate_w =
                    py::cast<decltype(Convolution3D::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().sparse =
                    py::cast<decltype(Convolution3D::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (data_type) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().data_type =
                    py::cast<decltype(Convolution3D::data_type)>(py::handle(data_type));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().format =
                    py::cast<decltype(Convolution3D::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().strategy =
                    py::cast<decltype(Convolution3D::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3D)*>(self)->inst().workspace_limit =
                    py::cast<decltype(Convolution3D::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Convolution3D)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(Convolution3D, mode), py_set_generic(Convolution3D, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_d"), py_get_generic(Convolution3D, pad_d), py_set_generic(Convolution3D, pad_d), const_cast<char*>("pad_d"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(Convolution3D, pad_h), py_set_generic(Convolution3D, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(Convolution3D, pad_w), py_set_generic(Convolution3D, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_d"), py_get_generic(Convolution3D, stride_d), py_set_generic(Convolution3D, stride_d), const_cast<char*>("stride_d"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(Convolution3D, stride_h), py_set_generic(Convolution3D, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(Convolution3D, stride_w), py_set_generic(Convolution3D, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_d"), py_get_generic(Convolution3D, dilate_d), py_set_generic(Convolution3D, dilate_d), const_cast<char*>("dilate_d"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(Convolution3D, dilate_h), py_set_generic(Convolution3D, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(Convolution3D, dilate_w), py_set_generic(Convolution3D, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(Convolution3D, sparse), py_set_generic(Convolution3D, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("data_type"), py_get_generic(Convolution3D, data_type), py_set_generic(Convolution3D, data_type), const_cast<char*>("data_type"), NULL},
    {const_cast<char*>("format"), py_get_generic(Convolution3D, format), py_set_generic(Convolution3D, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(Convolution3D, strategy), py_set_generic(Convolution3D, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(Convolution3D, workspace_limit), py_set_generic(Convolution3D, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Convolution3D)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Convolution3D)::getstate, METH_NOARGS, "Convolution3D getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Convolution3D)::setstate, METH_VARARGS, "Convolution3D setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Convolution3D)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Convolution3D)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Convolution3D)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Convolution3D)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_d: int = ..., pad_h: int = ..., pad_w: int = ..., stride_d: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_d: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., data_type: Union[str, DataType] = ..., format: Union[str, Format] = ..., strategy: Union[str, Strategy] = ...) -> None\n"
};

void _init_py_Convolution3D(py::module m) {
    using py_op = PyOp(Convolution3D);
    auto& py_type = PyOpType(Convolution3D);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Convolution3D";
    py_type.tp_basicsize = sizeof(PyOp(Convolution3D));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Convolution3D";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Convolution3D), &PyOp(Convolution3D)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Convolution3D_Mode(py_type);
    _init_py_Convolution3D_Sparse(py_type);
    _init_py_Convolution3D_DataType(py_type);
    _init_py_Convolution3D_Format(py_type);
    _init_py_Convolution3D_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("Convolution3D", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Convolution3D::typeinfo(), &py_type).second);
}

void _init_py_Convolution3DBackwardData_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3DBackwardData::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution3DBackwardData_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3DBackwardData::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution3DBackwardData_DataType(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3DBackwardData::DataType>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "DataType", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution3DBackwardData_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Convolution3DBackwardData::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Convolution3DBackwardData_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<Convolution3DBackwardData::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Convolution3DBackwardData) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_d", serialization<decltype(opdef.pad_d)>::dump(opdef.pad_d)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_d", serialization<decltype(opdef.stride_d)>::dump(opdef.stride_d)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_d", serialization<decltype(opdef.dilate_d)>::dump(opdef.dilate_d)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"data_type", serialization<decltype(opdef.data_type)>::dump(opdef.data_type)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_d");
        if (iter != state.end()) {
            opdef.pad_d = serialization<decltype(opdef.pad_d)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_d");
        if (iter != state.end()) {
            opdef.stride_d = serialization<decltype(opdef.stride_d)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_d");
        if (iter != state.end()) {
            opdef.dilate_d = serialization<decltype(opdef.dilate_d)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("data_type");
        if (iter != state.end()) {
            opdef.data_type = serialization<decltype(opdef.data_type)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Convolution3DBackwardData)

int PyOp(Convolution3DBackwardData)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_d", "pad_h", "pad_w", "stride_d", "stride_h", "stride_w", "dilate_d", "dilate_h", "dilate_w", "sparse", "data_type", "format", "strategy", "workspace_limit", "scope", NULL};
    PyObject *mode = NULL, *pad_d = NULL, *pad_h = NULL, *pad_w = NULL, *stride_d = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_d = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *data_type = NULL, *format = NULL, *strategy = NULL, *workspace_limit = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_d, &pad_h, &pad_w, &stride_d, &stride_h, &stride_w, &dilate_d, &dilate_h, &dilate_w, &sparse, &data_type, &format, &strategy, &workspace_limit, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().mode =
                    py::cast<decltype(Convolution3DBackwardData::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_d) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().pad_d =
                    py::cast<decltype(Convolution3DBackwardData::pad_d)>(py::handle(pad_d));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().pad_h =
                    py::cast<decltype(Convolution3DBackwardData::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().pad_w =
                    py::cast<decltype(Convolution3DBackwardData::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_d) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().stride_d =
                    py::cast<decltype(Convolution3DBackwardData::stride_d)>(py::handle(stride_d));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().stride_h =
                    py::cast<decltype(Convolution3DBackwardData::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().stride_w =
                    py::cast<decltype(Convolution3DBackwardData::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_d) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().dilate_d =
                    py::cast<decltype(Convolution3DBackwardData::dilate_d)>(py::handle(dilate_d));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().dilate_h =
                    py::cast<decltype(Convolution3DBackwardData::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().dilate_w =
                    py::cast<decltype(Convolution3DBackwardData::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().sparse =
                    py::cast<decltype(Convolution3DBackwardData::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (data_type) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().data_type =
                    py::cast<decltype(Convolution3DBackwardData::data_type)>(py::handle(data_type));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().format =
                    py::cast<decltype(Convolution3DBackwardData::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().strategy =
                    py::cast<decltype(Convolution3DBackwardData::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Convolution3DBackwardData)*>(self)->inst().workspace_limit =
                    py::cast<decltype(Convolution3DBackwardData::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Convolution3DBackwardData)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(Convolution3DBackwardData, mode), py_set_generic(Convolution3DBackwardData, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_d"), py_get_generic(Convolution3DBackwardData, pad_d), py_set_generic(Convolution3DBackwardData, pad_d), const_cast<char*>("pad_d"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(Convolution3DBackwardData, pad_h), py_set_generic(Convolution3DBackwardData, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(Convolution3DBackwardData, pad_w), py_set_generic(Convolution3DBackwardData, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_d"), py_get_generic(Convolution3DBackwardData, stride_d), py_set_generic(Convolution3DBackwardData, stride_d), const_cast<char*>("stride_d"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(Convolution3DBackwardData, stride_h), py_set_generic(Convolution3DBackwardData, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(Convolution3DBackwardData, stride_w), py_set_generic(Convolution3DBackwardData, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_d"), py_get_generic(Convolution3DBackwardData, dilate_d), py_set_generic(Convolution3DBackwardData, dilate_d), const_cast<char*>("dilate_d"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(Convolution3DBackwardData, dilate_h), py_set_generic(Convolution3DBackwardData, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(Convolution3DBackwardData, dilate_w), py_set_generic(Convolution3DBackwardData, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(Convolution3DBackwardData, sparse), py_set_generic(Convolution3DBackwardData, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("data_type"), py_get_generic(Convolution3DBackwardData, data_type), py_set_generic(Convolution3DBackwardData, data_type), const_cast<char*>("data_type"), NULL},
    {const_cast<char*>("format"), py_get_generic(Convolution3DBackwardData, format), py_set_generic(Convolution3DBackwardData, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(Convolution3DBackwardData, strategy), py_set_generic(Convolution3DBackwardData, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(Convolution3DBackwardData, workspace_limit), py_set_generic(Convolution3DBackwardData, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Convolution3DBackwardData)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Convolution3DBackwardData)::getstate, METH_NOARGS, "Convolution3DBackwardData getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Convolution3DBackwardData)::setstate, METH_VARARGS, "Convolution3DBackwardData setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Convolution3DBackwardData)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Convolution3DBackwardData)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Convolution3DBackwardData)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Convolution3DBackwardData)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_d: int = ..., pad_h: int = ..., pad_w: int = ..., stride_d: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_d: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., data_type: Union[str, DataType] = ..., format: Union[str, Format] = ..., strategy: Union[str, Strategy] = ...) -> None\n"
};

void _init_py_Convolution3DBackwardData(py::module m) {
    using py_op = PyOp(Convolution3DBackwardData);
    auto& py_type = PyOpType(Convolution3DBackwardData);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Convolution3DBackwardData";
    py_type.tp_basicsize = sizeof(PyOp(Convolution3DBackwardData));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Convolution3DBackwardData";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Convolution3DBackwardData), &PyOp(Convolution3DBackwardData)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Convolution3DBackwardData_Mode(py_type);
    _init_py_Convolution3DBackwardData_Sparse(py_type);
    _init_py_Convolution3DBackwardData_DataType(py_type);
    _init_py_Convolution3DBackwardData_Format(py_type);
    _init_py_Convolution3DBackwardData_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("Convolution3DBackwardData", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Convolution3DBackwardData::typeinfo(), &py_type).second);
}

void _init_py_ConvolutionBackwardData_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvolutionBackwardData::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvolutionBackwardData_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvolutionBackwardData::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvolutionBackwardData_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvolutionBackwardData::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvolutionBackwardData_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ConvolutionBackwardData::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ConvolutionBackwardData_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<ConvolutionBackwardData::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(ConvolutionBackwardData) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ConvolutionBackwardData)

int PyOp(ConvolutionBackwardData)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "strategy", "workspace_limit", "dtype", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *strategy = NULL, *workspace_limit = NULL, *dtype = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &strategy, &workspace_limit, &dtype, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().mode =
                    py::cast<decltype(ConvolutionBackwardData::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().pad_h =
                    py::cast<decltype(ConvolutionBackwardData::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().pad_w =
                    py::cast<decltype(ConvolutionBackwardData::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().stride_h =
                    py::cast<decltype(ConvolutionBackwardData::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().stride_w =
                    py::cast<decltype(ConvolutionBackwardData::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().dilate_h =
                    py::cast<decltype(ConvolutionBackwardData::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().dilate_w =
                    py::cast<decltype(ConvolutionBackwardData::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().sparse =
                    py::cast<decltype(ConvolutionBackwardData::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().format =
                    py::cast<decltype(ConvolutionBackwardData::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().compute_mode =
                    py::cast<decltype(ConvolutionBackwardData::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().strategy =
                    py::cast<decltype(ConvolutionBackwardData::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().workspace_limit =
                    py::cast<decltype(ConvolutionBackwardData::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ConvolutionBackwardData)*>(self)->inst().dtype =
                    py::cast<decltype(ConvolutionBackwardData::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ConvolutionBackwardData)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(ConvolutionBackwardData, mode), py_set_generic(ConvolutionBackwardData, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(ConvolutionBackwardData, pad_h), py_set_generic(ConvolutionBackwardData, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(ConvolutionBackwardData, pad_w), py_set_generic(ConvolutionBackwardData, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(ConvolutionBackwardData, stride_h), py_set_generic(ConvolutionBackwardData, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(ConvolutionBackwardData, stride_w), py_set_generic(ConvolutionBackwardData, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(ConvolutionBackwardData, dilate_h), py_set_generic(ConvolutionBackwardData, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(ConvolutionBackwardData, dilate_w), py_set_generic(ConvolutionBackwardData, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(ConvolutionBackwardData, sparse), py_set_generic(ConvolutionBackwardData, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(ConvolutionBackwardData, format), py_set_generic(ConvolutionBackwardData, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(ConvolutionBackwardData, compute_mode), py_set_generic(ConvolutionBackwardData, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(ConvolutionBackwardData, strategy), py_set_generic(ConvolutionBackwardData, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(ConvolutionBackwardData, workspace_limit), py_set_generic(ConvolutionBackwardData, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(ConvolutionBackwardData, dtype), py_set_generic(ConvolutionBackwardData, dtype), const_cast<char*>("dtype"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ConvolutionBackwardData)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ConvolutionBackwardData)::getstate, METH_NOARGS, "ConvolutionBackwardData getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ConvolutionBackwardData)::setstate, METH_VARARGS, "ConvolutionBackwardData setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ConvolutionBackwardData)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ConvolutionBackwardData)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ConvolutionBackwardData)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ConvolutionBackwardData)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ..., strategy: Union[str, Strategy] = ..., dtype: str = ...) -> None\n"
};

void _init_py_ConvolutionBackwardData(py::module m) {
    using py_op = PyOp(ConvolutionBackwardData);
    auto& py_type = PyOpType(ConvolutionBackwardData);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ConvolutionBackwardData";
    py_type.tp_basicsize = sizeof(PyOp(ConvolutionBackwardData));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ConvolutionBackwardData";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ConvolutionBackwardData), &PyOp(ConvolutionBackwardData)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_ConvolutionBackwardData_Mode(py_type);
    _init_py_ConvolutionBackwardData_Sparse(py_type);
    _init_py_ConvolutionBackwardData_Format(py_type);
    _init_py_ConvolutionBackwardData_ComputeMode(py_type);
    _init_py_ConvolutionBackwardData_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("ConvolutionBackwardData", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ConvolutionBackwardData::typeinfo(), &py_type).second);
}

PyOpDefBegin(Copy) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Copy)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Copy)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Copy)

int PyOp(Copy)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"comp_node", "scope", NULL};
    PyObject *comp_node = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &comp_node, &scope))
    return -1;

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Copy)*>(self)->inst().comp_node =
                    py::cast<decltype(Copy::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Copy)::py_getsetters[] = {
    {const_cast<char*>("comp_node"), py_get_generic(Copy, comp_node), py_set_generic(Copy, comp_node), const_cast<char*>("comp_node"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Copy)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Copy)::getstate, METH_NOARGS, "Copy getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Copy)::setstate, METH_VARARGS, "Copy setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Copy)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Copy)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Copy)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Copy)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, comp_node: str = ...) -> None\n"
};

void _init_py_Copy(py::module m) {
    using py_op = PyOp(Copy);
    auto& py_type = PyOpType(Copy);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Copy";
    py_type.tp_basicsize = sizeof(PyOp(Copy));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Copy";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Copy), &PyOp(Copy)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Copy", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Copy::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Correlation::Format> {
    static constexpr const char *name = "Correlation.Format";
    static constexpr std::underlying_type_t<Correlation::Format> max = 20 - 1;
};
template<> PyTypeObject* EnumWrapper<Correlation::Format>::type = nullptr;

template<> const char*
EnumWrapper<Correlation::Format>::members[] = {"NCHW", "NHWC", "NHWCD4", "NCHW4", "NCHW8", "NCHW32", "NCHW88", "NCHW44", "NCHW44_DOT", "NCHW_WINOGRAD", "NCHW88_WINOGRAD", "NCHW44_WINOGRAD", "NCHW4_NCHW32", "NCHW32_NCHW4", "NCHW4_NCHW", "NHWC_NCHW", "NHWC_NCHW4_IC_SMALL", "NCHW_NCHW4_IC_SMALL", "CHWN4", "NCHW4_NHWC"};

template<> std::unordered_map<std::string, Correlation::Format>
EnumWrapper<Correlation::Format>::mem2value = {{normalize_enum("NCHW"), Correlation::Format::NCHW}, {normalize_enum("NHWC"), Correlation::Format::NHWC}, {normalize_enum("NHWCD4"), Correlation::Format::NHWCD4}, {normalize_enum("NCHW4"), Correlation::Format::NCHW4}, {normalize_enum("NCHW8"), Correlation::Format::NCHW8}, {normalize_enum("NCHW32"), Correlation::Format::NCHW32}, {normalize_enum("NCHW88"), Correlation::Format::NCHW88}, {normalize_enum("NCHW44"), Correlation::Format::NCHW44}, {normalize_enum("NCHW44_DOT"), Correlation::Format::NCHW44_DOT}, {normalize_enum("NCHW_WINOGRAD"), Correlation::Format::NCHW_WINOGRAD}, {normalize_enum("NCHW88_WINOGRAD"), Correlation::Format::NCHW88_WINOGRAD}, {normalize_enum("NCHW44_WINOGRAD"), Correlation::Format::NCHW44_WINOGRAD}, {normalize_enum("NCHW4_NCHW32"), Correlation::Format::NCHW4_NCHW32}, {normalize_enum("NCHW32_NCHW4"), Correlation::Format::NCHW32_NCHW4}, {normalize_enum("NCHW4_NCHW"), Correlation::Format::NCHW4_NCHW}, {normalize_enum("NHWC_NCHW"), Correlation::Format::NHWC_NCHW}, {normalize_enum("NHWC_NCHW4_IC_SMALL"), Correlation::Format::NHWC_NCHW4_IC_SMALL}, {normalize_enum("NCHW_NCHW4_IC_SMALL"), Correlation::Format::NCHW_NCHW4_IC_SMALL}, {normalize_enum("CHWN4"), Correlation::Format::CHWN4}, {normalize_enum("NCHW4_NHWC"), Correlation::Format::NCHW4_NHWC}};
template<> PyObject* EnumWrapper<Correlation::Format>::pyobj_insts[20] = {nullptr};

void _init_py_Correlation_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Correlation::Format>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Correlation::Format>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Correlation::Format>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Correlation::Format>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Correlation.Format",
        // basicsize
        sizeof(EnumWrapper<Correlation::Format>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Format").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Correlation.Format").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NHWC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWC", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NHWCD4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWCD4", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW8", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW32", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW88;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW88", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[6] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW44;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW44", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[7] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW44_DOT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW44_DOT", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[8] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW_WINOGRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW_WINOGRAD", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[9] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW88_WINOGRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW88_WINOGRAD", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[10] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW44_WINOGRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW44_WINOGRAD", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[11] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW4_NCHW32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4_NCHW32", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[12] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW32_NCHW4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW32_NCHW4", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[13] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW4_NCHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4_NCHW", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[14] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NHWC_NCHW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWC_NCHW", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[15] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NHWC_NCHW4_IC_SMALL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NHWC_NCHW4_IC_SMALL", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[16] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW_NCHW4_IC_SMALL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW_NCHW4_IC_SMALL", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[17] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::CHWN4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CHWN4", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[18] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Correlation::Format>*>(inst)->value = Correlation::Format::NCHW4_NHWC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NCHW4_NHWC", inst) >= 0);
    EnumWrapper<Correlation::Format>::pyobj_insts[19] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Correlation) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Correlation)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"kernel_size", serialization<decltype(opdef.kernel_size)>::dump(opdef.kernel_size)},
            {"max_displacement", serialization<decltype(opdef.max_displacement)>::dump(opdef.max_displacement)},
            {"stride1", serialization<decltype(opdef.stride1)>::dump(opdef.stride1)},
            {"stride2", serialization<decltype(opdef.stride2)>::dump(opdef.stride2)},
            {"pad_size", serialization<decltype(opdef.pad_size)>::dump(opdef.pad_size)},
            {"is_multiply", serialization<decltype(opdef.is_multiply)>::dump(opdef.is_multiply)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Correlation)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("kernel_size");
        if (iter != state.end()) {
            opdef.kernel_size = serialization<decltype(opdef.kernel_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("max_displacement");
        if (iter != state.end()) {
            opdef.max_displacement = serialization<decltype(opdef.max_displacement)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride1");
        if (iter != state.end()) {
            opdef.stride1 = serialization<decltype(opdef.stride1)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride2");
        if (iter != state.end()) {
            opdef.stride2 = serialization<decltype(opdef.stride2)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_size");
        if (iter != state.end()) {
            opdef.pad_size = serialization<decltype(opdef.pad_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("is_multiply");
        if (iter != state.end()) {
            opdef.is_multiply = serialization<decltype(opdef.is_multiply)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Correlation)

int PyOp(Correlation)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"format", "kernel_size", "max_displacement", "stride1", "stride2", "pad_size", "is_multiply", "scope", NULL};
    PyObject *format = NULL, *kernel_size = NULL, *max_displacement = NULL, *stride1 = NULL, *stride2 = NULL, *pad_size = NULL, *is_multiply = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOO", const_cast<char**>(kwlist), &format, &kernel_size, &max_displacement, &stride1, &stride2, &pad_size, &is_multiply, &scope))
    return -1;

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().format =
                    py::cast<decltype(Correlation::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (kernel_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().kernel_size =
                    py::cast<decltype(Correlation::kernel_size)>(py::handle(kernel_size));
        } CATCH_ALL(-1)
    }

    if (max_displacement) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().max_displacement =
                    py::cast<decltype(Correlation::max_displacement)>(py::handle(max_displacement));
        } CATCH_ALL(-1)
    }

    if (stride1) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().stride1 =
                    py::cast<decltype(Correlation::stride1)>(py::handle(stride1));
        } CATCH_ALL(-1)
    }

    if (stride2) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().stride2 =
                    py::cast<decltype(Correlation::stride2)>(py::handle(stride2));
        } CATCH_ALL(-1)
    }

    if (pad_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().pad_size =
                    py::cast<decltype(Correlation::pad_size)>(py::handle(pad_size));
        } CATCH_ALL(-1)
    }

    if (is_multiply) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Correlation)*>(self)->inst().is_multiply =
                    py::cast<decltype(Correlation::is_multiply)>(py::handle(is_multiply));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Correlation)::py_getsetters[] = {
    {const_cast<char*>("format"), py_get_generic(Correlation, format), py_set_generic(Correlation, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("kernel_size"), py_get_generic(Correlation, kernel_size), py_set_generic(Correlation, kernel_size), const_cast<char*>("kernel_size"), NULL},
    {const_cast<char*>("max_displacement"), py_get_generic(Correlation, max_displacement), py_set_generic(Correlation, max_displacement), const_cast<char*>("max_displacement"), NULL},
    {const_cast<char*>("stride1"), py_get_generic(Correlation, stride1), py_set_generic(Correlation, stride1), const_cast<char*>("stride1"), NULL},
    {const_cast<char*>("stride2"), py_get_generic(Correlation, stride2), py_set_generic(Correlation, stride2), const_cast<char*>("stride2"), NULL},
    {const_cast<char*>("pad_size"), py_get_generic(Correlation, pad_size), py_set_generic(Correlation, pad_size), const_cast<char*>("pad_size"), NULL},
    {const_cast<char*>("is_multiply"), py_get_generic(Correlation, is_multiply), py_set_generic(Correlation, is_multiply), const_cast<char*>("is_multiply"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Correlation)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Correlation)::getstate, METH_NOARGS, "Correlation getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Correlation)::setstate, METH_VARARGS, "Correlation setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Correlation)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Correlation)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Correlation)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Correlation)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, format: Union[str, Format] = ..., kernel_size: int = ..., max_displacement: int = ..., stride1: int = ..., stride2: int = ..., pad_size: int = ..., is_multiply: bool = ...) -> None\n"
};

void _init_py_Correlation(py::module m) {
    using py_op = PyOp(Correlation);
    auto& py_type = PyOpType(Correlation);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Correlation";
    py_type.tp_basicsize = sizeof(PyOp(Correlation));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Correlation";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Correlation), &PyOp(Correlation)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Correlation_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("Correlation", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Correlation::typeinfo(), &py_type).second);
}

PyOpDefBegin(Cumsum) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Cumsum)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"exclusive", serialization<decltype(opdef.exclusive)>::dump(opdef.exclusive)},
            {"reverse", serialization<decltype(opdef.reverse)>::dump(opdef.reverse)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Cumsum)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("exclusive");
        if (iter != state.end()) {
            opdef.exclusive = serialization<decltype(opdef.exclusive)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("reverse");
        if (iter != state.end()) {
            opdef.reverse = serialization<decltype(opdef.reverse)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Cumsum)

int PyOp(Cumsum)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "exclusive", "reverse", "scope", NULL};
    PyObject *axis = NULL, *exclusive = NULL, *reverse = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", const_cast<char**>(kwlist), &axis, &exclusive, &reverse, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Cumsum)*>(self)->inst().axis =
                    py::cast<decltype(Cumsum::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (exclusive) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Cumsum)*>(self)->inst().exclusive =
                    py::cast<decltype(Cumsum::exclusive)>(py::handle(exclusive));
        } CATCH_ALL(-1)
    }

    if (reverse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Cumsum)*>(self)->inst().reverse =
                    py::cast<decltype(Cumsum::reverse)>(py::handle(reverse));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Cumsum)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Cumsum, axis), py_set_generic(Cumsum, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("exclusive"), py_get_generic(Cumsum, exclusive), py_set_generic(Cumsum, exclusive), const_cast<char*>("exclusive"), NULL},
    {const_cast<char*>("reverse"), py_get_generic(Cumsum, reverse), py_set_generic(Cumsum, reverse), const_cast<char*>("reverse"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Cumsum)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Cumsum)::getstate, METH_NOARGS, "Cumsum getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Cumsum)::setstate, METH_VARARGS, "Cumsum setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Cumsum)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Cumsum)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Cumsum)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Cumsum)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ..., exclusive: bool = ..., reverse: bool = ...) -> None\n"
};

void _init_py_Cumsum(py::module m) {
    using py_op = PyOp(Cumsum);
    auto& py_type = PyOpType(Cumsum);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Cumsum";
    py_type.tp_basicsize = sizeof(PyOp(Cumsum));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Cumsum";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Cumsum), &PyOp(Cumsum)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Cumsum", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Cumsum::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<CvtColor::Mode> {
    static constexpr const char *name = "CvtColor.Mode";
    static constexpr std::underlying_type_t<CvtColor::Mode> max = 32 - 1;
};
template<> PyTypeObject* EnumWrapper<CvtColor::Mode>::type = nullptr;

template<> const char*
EnumWrapper<CvtColor::Mode>::members[] = {"RGB2GRAY", "RGB2YUV", "YUV2RGB", "GRAY2RGB", "RGBA2RGB", "RGBA2BGR", "RGBA2GRAY", "RGB2BGR", "BGR2GRAY", "BGR2RGB", "YUV2GRAY_NV21", "YUV2RGB_NV21", "YUV2BGR_NV21", "YUV2GRAY_NV12", "YUV2RGB_NV12", "YUV2BGR_NV12", "YUV2GRAY_YV12", "YUV2RGB_YV12", "YUV2BGR_YV12", "YUV2GRAY_YU12", "YUV2RGB_YU12", "YUV2BGR_YU12", "YCrCb2RGB", "YCrCb2BGR", "BT601_YUV2RGB_NV21", "BT601_YUV2BGR_NV21", "BT601_YUV2RGB_NV12", "BT601_YUV2BGR_NV12", "BT601_YUV2RGB_YV12", "BT601_YUV2BGR_YV12", "BT601_YUV2RGB_YU12", "BT601_YUV2BGR_YU12"};

template<> std::unordered_map<std::string, CvtColor::Mode>
EnumWrapper<CvtColor::Mode>::mem2value = {{normalize_enum("RGB2GRAY"), CvtColor::Mode::RGB2GRAY}, {normalize_enum("RGB2YUV"), CvtColor::Mode::RGB2YUV}, {normalize_enum("YUV2RGB"), CvtColor::Mode::YUV2RGB}, {normalize_enum("GRAY2RGB"), CvtColor::Mode::GRAY2RGB}, {normalize_enum("RGBA2RGB"), CvtColor::Mode::RGBA2RGB}, {normalize_enum("RGBA2BGR"), CvtColor::Mode::RGBA2BGR}, {normalize_enum("RGBA2GRAY"), CvtColor::Mode::RGBA2GRAY}, {normalize_enum("RGB2BGR"), CvtColor::Mode::RGB2BGR}, {normalize_enum("BGR2GRAY"), CvtColor::Mode::BGR2GRAY}, {normalize_enum("BGR2RGB"), CvtColor::Mode::BGR2RGB}, {normalize_enum("YUV2GRAY_NV21"), CvtColor::Mode::YUV2GRAY_NV21}, {normalize_enum("YUV2RGB_NV21"), CvtColor::Mode::YUV2RGB_NV21}, {normalize_enum("YUV2BGR_NV21"), CvtColor::Mode::YUV2BGR_NV21}, {normalize_enum("YUV2GRAY_NV12"), CvtColor::Mode::YUV2GRAY_NV12}, {normalize_enum("YUV2RGB_NV12"), CvtColor::Mode::YUV2RGB_NV12}, {normalize_enum("YUV2BGR_NV12"), CvtColor::Mode::YUV2BGR_NV12}, {normalize_enum("YUV2GRAY_YV12"), CvtColor::Mode::YUV2GRAY_YV12}, {normalize_enum("YUV2RGB_YV12"), CvtColor::Mode::YUV2RGB_YV12}, {normalize_enum("YUV2BGR_YV12"), CvtColor::Mode::YUV2BGR_YV12}, {normalize_enum("YUV2GRAY_YU12"), CvtColor::Mode::YUV2GRAY_YU12}, {normalize_enum("YUV2RGB_YU12"), CvtColor::Mode::YUV2RGB_YU12}, {normalize_enum("YUV2BGR_YU12"), CvtColor::Mode::YUV2BGR_YU12}, {normalize_enum("YCrCb2RGB"), CvtColor::Mode::YCrCb2RGB}, {normalize_enum("YCrCb2BGR"), CvtColor::Mode::YCrCb2BGR}, {normalize_enum("BT601_YUV2RGB_NV21"), CvtColor::Mode::BT601_YUV2RGB_NV21}, {normalize_enum("BT601_YUV2BGR_NV21"), CvtColor::Mode::BT601_YUV2BGR_NV21}, {normalize_enum("BT601_YUV2RGB_NV12"), CvtColor::Mode::BT601_YUV2RGB_NV12}, {normalize_enum("BT601_YUV2BGR_NV12"), CvtColor::Mode::BT601_YUV2BGR_NV12}, {normalize_enum("BT601_YUV2RGB_YV12"), CvtColor::Mode::BT601_YUV2RGB_YV12}, {normalize_enum("BT601_YUV2BGR_YV12"), CvtColor::Mode::BT601_YUV2BGR_YV12}, {normalize_enum("BT601_YUV2RGB_YU12"), CvtColor::Mode::BT601_YUV2RGB_YU12}, {normalize_enum("BT601_YUV2BGR_YU12"), CvtColor::Mode::BT601_YUV2BGR_YU12}};
template<> PyObject* EnumWrapper<CvtColor::Mode>::pyobj_insts[32] = {nullptr};

void _init_py_CvtColor_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<CvtColor::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<CvtColor::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<CvtColor::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<CvtColor::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.CvtColor.Mode",
        // basicsize
        sizeof(EnumWrapper<CvtColor::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("CvtColor.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::RGB2GRAY;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RGB2GRAY", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::RGB2YUV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RGB2YUV", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2RGB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2RGB", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::GRAY2RGB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "GRAY2RGB", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::RGBA2RGB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RGBA2RGB", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::RGBA2BGR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RGBA2BGR", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::RGBA2GRAY;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RGBA2GRAY", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[6] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::RGB2BGR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RGB2BGR", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[7] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BGR2GRAY;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BGR2GRAY", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[8] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BGR2RGB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BGR2RGB", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[9] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2GRAY_NV21;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2GRAY_NV21", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[10] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2RGB_NV21;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2RGB_NV21", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[11] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2BGR_NV21;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2BGR_NV21", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[12] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2GRAY_NV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2GRAY_NV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[13] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2RGB_NV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2RGB_NV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[14] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2BGR_NV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2BGR_NV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[15] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2GRAY_YV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2GRAY_YV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[16] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2RGB_YV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2RGB_YV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[17] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2BGR_YV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2BGR_YV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[18] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2GRAY_YU12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2GRAY_YU12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[19] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2RGB_YU12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2RGB_YU12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[20] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YUV2BGR_YU12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YUV2BGR_YU12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[21] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YCrCb2RGB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YCrCb2RGB", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[22] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::YCrCb2BGR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "YCrCb2BGR", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[23] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2RGB_NV21;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2RGB_NV21", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[24] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2BGR_NV21;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2BGR_NV21", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[25] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2RGB_NV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2RGB_NV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[26] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2BGR_NV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2BGR_NV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[27] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2RGB_YV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2RGB_YV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[28] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2BGR_YV12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2BGR_YV12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[29] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2RGB_YU12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2RGB_YU12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[30] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<CvtColor::Mode>*>(inst)->value = CvtColor::Mode::BT601_YUV2BGR_YU12;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "BT601_YUV2BGR_YU12", inst) >= 0);
    EnumWrapper<CvtColor::Mode>::pyobj_insts[31] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(CvtColor) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(CvtColor)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(CvtColor)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(CvtColor)

int PyOp(CvtColor)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "scope", NULL};
    PyObject *mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &mode, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(CvtColor)*>(self)->inst().mode =
                    py::cast<decltype(CvtColor::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(CvtColor)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(CvtColor, mode), py_set_generic(CvtColor, mode), const_cast<char*>("mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(CvtColor)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(CvtColor)::getstate, METH_NOARGS, "CvtColor getstate"},
    {const_cast<char*>("__setstate__"), PyOp(CvtColor)::setstate, METH_VARARGS, "CvtColor setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(CvtColor)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(CvtColor)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(CvtColor)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(CvtColor)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ...) -> None\n"
};

void _init_py_CvtColor(py::module m) {
    using py_op = PyOp(CvtColor);
    auto& py_type = PyOpType(CvtColor);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.CvtColor";
    py_type.tp_basicsize = sizeof(PyOp(CvtColor));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "CvtColor";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(CvtColor), &PyOp(CvtColor)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_CvtColor_Mode(py_type);

    PyType_Modified(&py_type);
    m.add_object("CvtColor", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(CvtColor::typeinfo(), &py_type).second);
}

void _init_py_DeformableConv_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<DeformableConv::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_DeformableConv_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<DeformableConv::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_DeformableConv_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<DeformableConv::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_DeformableConv_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<DeformableConv::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_DeformableConv_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<DeformableConv::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(DeformableConv) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(DeformableConv)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(DeformableConv)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(DeformableConv)

int PyOp(DeformableConv)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "strategy", "workspace_limit", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *strategy = NULL, *workspace_limit = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &strategy, &workspace_limit, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().mode =
                    py::cast<decltype(DeformableConv::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().pad_h =
                    py::cast<decltype(DeformableConv::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().pad_w =
                    py::cast<decltype(DeformableConv::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().stride_h =
                    py::cast<decltype(DeformableConv::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().stride_w =
                    py::cast<decltype(DeformableConv::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().dilate_h =
                    py::cast<decltype(DeformableConv::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().dilate_w =
                    py::cast<decltype(DeformableConv::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().sparse =
                    py::cast<decltype(DeformableConv::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().format =
                    py::cast<decltype(DeformableConv::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().compute_mode =
                    py::cast<decltype(DeformableConv::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().strategy =
                    py::cast<decltype(DeformableConv::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformableConv)*>(self)->inst().workspace_limit =
                    py::cast<decltype(DeformableConv::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(DeformableConv)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(DeformableConv, mode), py_set_generic(DeformableConv, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(DeformableConv, pad_h), py_set_generic(DeformableConv, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(DeformableConv, pad_w), py_set_generic(DeformableConv, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(DeformableConv, stride_h), py_set_generic(DeformableConv, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(DeformableConv, stride_w), py_set_generic(DeformableConv, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(DeformableConv, dilate_h), py_set_generic(DeformableConv, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(DeformableConv, dilate_w), py_set_generic(DeformableConv, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(DeformableConv, sparse), py_set_generic(DeformableConv, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(DeformableConv, format), py_set_generic(DeformableConv, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(DeformableConv, compute_mode), py_set_generic(DeformableConv, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(DeformableConv, strategy), py_set_generic(DeformableConv, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(DeformableConv, workspace_limit), py_set_generic(DeformableConv, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(DeformableConv)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(DeformableConv)::getstate, METH_NOARGS, "DeformableConv getstate"},
    {const_cast<char*>("__setstate__"), PyOp(DeformableConv)::setstate, METH_VARARGS, "DeformableConv setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(DeformableConv)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(DeformableConv)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(DeformableConv)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(DeformableConv)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ..., strategy: Union[str, Strategy] = ...) -> None\n"
};

void _init_py_DeformableConv(py::module m) {
    using py_op = PyOp(DeformableConv);
    auto& py_type = PyOpType(DeformableConv);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.DeformableConv";
    py_type.tp_basicsize = sizeof(PyOp(DeformableConv));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "DeformableConv";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(DeformableConv), &PyOp(DeformableConv)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_DeformableConv_Mode(py_type);
    _init_py_DeformableConv_Sparse(py_type);
    _init_py_DeformableConv_Format(py_type);
    _init_py_DeformableConv_ComputeMode(py_type);
    _init_py_DeformableConv_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("DeformableConv", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(DeformableConv::typeinfo(), &py_type).second);
}

PyOpDefBegin(DeformablePSROIPooling) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"no_trans", serialization<decltype(opdef.no_trans)>::dump(opdef.no_trans)},
            {"spatial_scale", serialization<decltype(opdef.spatial_scale)>::dump(opdef.spatial_scale)},
            {"trans_std", serialization<decltype(opdef.trans_std)>::dump(opdef.trans_std)},
            {"pooled_h", serialization<decltype(opdef.pooled_h)>::dump(opdef.pooled_h)},
            {"pooled_w", serialization<decltype(opdef.pooled_w)>::dump(opdef.pooled_w)},
            {"part_size", serialization<decltype(opdef.part_size)>::dump(opdef.part_size)},
            {"sample_per_part", serialization<decltype(opdef.sample_per_part)>::dump(opdef.sample_per_part)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("no_trans");
        if (iter != state.end()) {
            opdef.no_trans = serialization<decltype(opdef.no_trans)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("spatial_scale");
        if (iter != state.end()) {
            opdef.spatial_scale = serialization<decltype(opdef.spatial_scale)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("trans_std");
        if (iter != state.end()) {
            opdef.trans_std = serialization<decltype(opdef.trans_std)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pooled_h");
        if (iter != state.end()) {
            opdef.pooled_h = serialization<decltype(opdef.pooled_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pooled_w");
        if (iter != state.end()) {
            opdef.pooled_w = serialization<decltype(opdef.pooled_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("part_size");
        if (iter != state.end()) {
            opdef.part_size = serialization<decltype(opdef.part_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sample_per_part");
        if (iter != state.end()) {
            opdef.sample_per_part = serialization<decltype(opdef.sample_per_part)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(DeformablePSROIPooling)

int PyOp(DeformablePSROIPooling)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"no_trans", "spatial_scale", "trans_std", "pooled_h", "pooled_w", "part_size", "sample_per_part", "scope", NULL};
    PyObject *no_trans = NULL, *spatial_scale = NULL, *trans_std = NULL, *pooled_h = NULL, *pooled_w = NULL, *part_size = NULL, *sample_per_part = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOO", const_cast<char**>(kwlist), &no_trans, &spatial_scale, &trans_std, &pooled_h, &pooled_w, &part_size, &sample_per_part, &scope))
    return -1;

    if (no_trans) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().no_trans =
                    py::cast<decltype(DeformablePSROIPooling::no_trans)>(py::handle(no_trans));
        } CATCH_ALL(-1)
    }

    if (spatial_scale) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().spatial_scale =
                    py::cast<decltype(DeformablePSROIPooling::spatial_scale)>(py::handle(spatial_scale));
        } CATCH_ALL(-1)
    }

    if (trans_std) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().trans_std =
                    py::cast<decltype(DeformablePSROIPooling::trans_std)>(py::handle(trans_std));
        } CATCH_ALL(-1)
    }

    if (pooled_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().pooled_h =
                    py::cast<decltype(DeformablePSROIPooling::pooled_h)>(py::handle(pooled_h));
        } CATCH_ALL(-1)
    }

    if (pooled_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().pooled_w =
                    py::cast<decltype(DeformablePSROIPooling::pooled_w)>(py::handle(pooled_w));
        } CATCH_ALL(-1)
    }

    if (part_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().part_size =
                    py::cast<decltype(DeformablePSROIPooling::part_size)>(py::handle(part_size));
        } CATCH_ALL(-1)
    }

    if (sample_per_part) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(DeformablePSROIPooling)*>(self)->inst().sample_per_part =
                    py::cast<decltype(DeformablePSROIPooling::sample_per_part)>(py::handle(sample_per_part));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(DeformablePSROIPooling)::py_getsetters[] = {
    {const_cast<char*>("no_trans"), py_get_generic(DeformablePSROIPooling, no_trans), py_set_generic(DeformablePSROIPooling, no_trans), const_cast<char*>("no_trans"), NULL},
    {const_cast<char*>("spatial_scale"), py_get_generic(DeformablePSROIPooling, spatial_scale), py_set_generic(DeformablePSROIPooling, spatial_scale), const_cast<char*>("spatial_scale"), NULL},
    {const_cast<char*>("trans_std"), py_get_generic(DeformablePSROIPooling, trans_std), py_set_generic(DeformablePSROIPooling, trans_std), const_cast<char*>("trans_std"), NULL},
    {const_cast<char*>("pooled_h"), py_get_generic(DeformablePSROIPooling, pooled_h), py_set_generic(DeformablePSROIPooling, pooled_h), const_cast<char*>("pooled_h"), NULL},
    {const_cast<char*>("pooled_w"), py_get_generic(DeformablePSROIPooling, pooled_w), py_set_generic(DeformablePSROIPooling, pooled_w), const_cast<char*>("pooled_w"), NULL},
    {const_cast<char*>("part_size"), py_get_generic(DeformablePSROIPooling, part_size), py_set_generic(DeformablePSROIPooling, part_size), const_cast<char*>("part_size"), NULL},
    {const_cast<char*>("sample_per_part"), py_get_generic(DeformablePSROIPooling, sample_per_part), py_set_generic(DeformablePSROIPooling, sample_per_part), const_cast<char*>("sample_per_part"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(DeformablePSROIPooling)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(DeformablePSROIPooling)::getstate, METH_NOARGS, "DeformablePSROIPooling getstate"},
    {const_cast<char*>("__setstate__"), PyOp(DeformablePSROIPooling)::setstate, METH_VARARGS, "DeformablePSROIPooling setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(DeformablePSROIPooling)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(DeformablePSROIPooling)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(DeformablePSROIPooling)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(DeformablePSROIPooling)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, no_trans: bool = ..., spatial_scale: float = ..., trans_std: float = ..., pooled_h: int = ..., pooled_w: int = ..., part_size: int = ..., sample_per_part: int = ...) -> None\n"
};

void _init_py_DeformablePSROIPooling(py::module m) {
    using py_op = PyOp(DeformablePSROIPooling);
    auto& py_type = PyOpType(DeformablePSROIPooling);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.DeformablePSROIPooling";
    py_type.tp_basicsize = sizeof(PyOp(DeformablePSROIPooling));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "DeformablePSROIPooling";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(DeformablePSROIPooling), &PyOp(DeformablePSROIPooling)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("DeformablePSROIPooling", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(DeformablePSROIPooling::typeinfo(), &py_type).second);
}

PyOpDefBegin(Diag) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Diag)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"k", serialization<decltype(opdef.k)>::dump(opdef.k)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Diag)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("k");
        if (iter != state.end()) {
            opdef.k = serialization<decltype(opdef.k)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Diag)

int PyOp(Diag)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"k", "scope", NULL};
    PyObject *k = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &k, &scope))
    return -1;

    if (k) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Diag)*>(self)->inst().k =
                    py::cast<decltype(Diag::k)>(py::handle(k));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Diag)::py_getsetters[] = {
    {const_cast<char*>("k"), py_get_generic(Diag, k), py_set_generic(Diag, k), const_cast<char*>("k"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Diag)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Diag)::getstate, METH_NOARGS, "Diag getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Diag)::setstate, METH_VARARGS, "Diag setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Diag)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Diag)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Diag)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Diag)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, k: int = ...) -> None\n"
};

void _init_py_Diag(py::module m) {
    using py_op = PyOp(Diag);
    auto& py_type = PyOpType(Diag);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Diag";
    py_type.tp_basicsize = sizeof(PyOp(Diag));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Diag";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Diag), &PyOp(Diag)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Diag", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Diag::typeinfo(), &py_type).second);
}

PyOpDefBegin(Dimshuffle) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Dimshuffle)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"pattern", serialization<decltype(opdef.pattern)>::dump(opdef.pattern)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Dimshuffle)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("pattern");
        if (iter != state.end()) {
            opdef.pattern = serialization<decltype(opdef.pattern)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Dimshuffle)

int PyOp(Dimshuffle)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"pattern", "scope", NULL};
    PyObject *pattern = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &pattern, &scope))
    return -1;

    if (pattern) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Dimshuffle)*>(self)->inst().pattern =
                    py::cast<decltype(Dimshuffle::pattern)>(py::handle(pattern));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Dimshuffle)::py_getsetters[] = {
    {const_cast<char*>("pattern"), py_get_generic(Dimshuffle, pattern), py_set_generic(Dimshuffle, pattern), const_cast<char*>("pattern"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Dimshuffle)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Dimshuffle)::getstate, METH_NOARGS, "Dimshuffle getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Dimshuffle)::setstate, METH_VARARGS, "Dimshuffle setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Dimshuffle)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Dimshuffle)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Dimshuffle)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Dimshuffle)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, pattern: list[int] = ...) -> None\n"
};

void _init_py_Dimshuffle(py::module m) {
    using py_op = PyOp(Dimshuffle);
    auto& py_type = PyOpType(Dimshuffle);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Dimshuffle";
    py_type.tp_basicsize = sizeof(PyOp(Dimshuffle));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Dimshuffle";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Dimshuffle), &PyOp(Dimshuffle)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Dimshuffle", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Dimshuffle::typeinfo(), &py_type).second);
}

PyOpDefBegin(Dot) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Dot)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Dot)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Dot)

int PyOp(Dot)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(Dot)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Dot)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Dot)::getstate, METH_NOARGS, "Dot getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Dot)::setstate, METH_VARARGS, "Dot setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Dot)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Dot)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Dot)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Dot)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_Dot(py::module m) {
    using py_op = PyOp(Dot);
    auto& py_type = PyOpType(Dot);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Dot";
    py_type.tp_basicsize = sizeof(PyOp(Dot));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Dot";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Dot), &PyOp(Dot)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Dot", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Dot::typeinfo(), &py_type).second);
}

PyOpDefBegin(Dropout) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Dropout)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"drop_prob", serialization<decltype(opdef.drop_prob)>::dump(opdef.drop_prob)},
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Dropout)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("drop_prob");
        if (iter != state.end()) {
            opdef.drop_prob = serialization<decltype(opdef.drop_prob)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Dropout)

int PyOp(Dropout)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"drop_prob", "seed", "handle", "scope", NULL};
    PyObject *drop_prob = NULL, *seed = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", const_cast<char**>(kwlist), &drop_prob, &seed, &handle, &scope))
    return -1;

    if (drop_prob) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Dropout)*>(self)->inst().drop_prob =
                    py::cast<decltype(Dropout::drop_prob)>(py::handle(drop_prob));
        } CATCH_ALL(-1)
    }

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Dropout)*>(self)->inst().seed =
                    py::cast<decltype(Dropout::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Dropout)*>(self)->inst().handle =
                    py::cast<decltype(Dropout::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Dropout)::py_getsetters[] = {
    {const_cast<char*>("drop_prob"), py_get_generic(Dropout, drop_prob), py_set_generic(Dropout, drop_prob), const_cast<char*>("drop_prob"), NULL},
    {const_cast<char*>("seed"), py_get_generic(Dropout, seed), py_set_generic(Dropout, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("handle"), py_get_generic(Dropout, handle), py_set_generic(Dropout, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Dropout)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Dropout)::getstate, METH_NOARGS, "Dropout getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Dropout)::setstate, METH_VARARGS, "Dropout setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Dropout)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Dropout)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Dropout)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Dropout)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, drop_prob: float = ..., seed: int = ..., handle: int = ...) -> None\n"
};

void _init_py_Dropout(py::module m) {
    using py_op = PyOp(Dropout);
    auto& py_type = PyOpType(Dropout);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Dropout";
    py_type.tp_basicsize = sizeof(PyOp(Dropout));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Dropout";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Dropout), &PyOp(Dropout)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Dropout", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Dropout::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Elemwise::Mode> {
    static constexpr const char *name = "Elemwise.Mode";
    static constexpr std::underlying_type_t<Elemwise::Mode> max = 86 - 1;
};
template<> PyTypeObject* EnumWrapper<Elemwise::Mode>::type = nullptr;

template<> const char*
EnumWrapper<Elemwise::Mode>::members[] = {"RELU", "ABS", "ACOS", "ASIN", "CEIL", "COS", "EXP", "EXPM1", "FLOOR", "LOG", "LOG1P", "NEGATE", "SIGMOID", "SIN", "TANH", "ABS_GRAD", "ADD", "FLOOR_DIV", "MAX", "MIN", "MOD", "MUL", "POW", "SIGMOID_GRAD", "SUB", "SWITCH_GT0", "TANH_GRAD", "TRUE_DIV", "LOG_SUM_EXP", "LT", "LEQ", "EQ", "SHL", "SHR", "COND_LEQ_MOV", "FUSE_MUL_ADD3", "FUSE_MUL_ADD4", "FUSE_ADD_RELU", "FUSE_ADD_SIGMOID", "FUSE_ADD_TANH", "FAST_TANH", "FAST_TANH_GRAD", "ROUND", "RMULH", "ATAN2", "ERF", "ERFINV", "ERFC", "ERFCINV", "H_SWISH", "H_SWISH_GRAD", "FUSE_ADD_H_SWISH", "NOT", "AND", "OR", "XOR", "SILU", "SILU_GRAD", "GELU", "GELU_GRAD", "COND_LT_MOV", "NEQ", "ISNAN", "ISINF", "SINH", "COSH", "ASINH", "ACOSH", "ATANH", "TAN", "ASINH_GRAD", "ACOSH_GRAD", "ATANH_GRAD", "PRELU", "CLIP", "PRELU_GRAD", "SOFTPLUS", "SOFTPLUS_GRAD", "RELU6", "RELU6_GRAD", "HSIGMOID", "HSIGMOID_GRAD", "LOGSIGMOID", "SQRT", "SQUARE", "SIGN"};

template<> std::unordered_map<std::string, Elemwise::Mode>
EnumWrapper<Elemwise::Mode>::mem2value = {{normalize_enum("RELU"), Elemwise::Mode::RELU}, {normalize_enum("ABS"), Elemwise::Mode::ABS}, {normalize_enum("ACOS"), Elemwise::Mode::ACOS}, {normalize_enum("ASIN"), Elemwise::Mode::ASIN}, {normalize_enum("CEIL"), Elemwise::Mode::CEIL}, {normalize_enum("COS"), Elemwise::Mode::COS}, {normalize_enum("EXP"), Elemwise::Mode::EXP}, {normalize_enum("EXPM1"), Elemwise::Mode::EXPM1}, {normalize_enum("FLOOR"), Elemwise::Mode::FLOOR}, {normalize_enum("LOG"), Elemwise::Mode::LOG}, {normalize_enum("LOG1P"), Elemwise::Mode::LOG1P}, {normalize_enum("NEGATE"), Elemwise::Mode::NEGATE}, {normalize_enum("SIGMOID"), Elemwise::Mode::SIGMOID}, {normalize_enum("SIN"), Elemwise::Mode::SIN}, {normalize_enum("TANH"), Elemwise::Mode::TANH}, {normalize_enum("ABS_GRAD"), Elemwise::Mode::ABS_GRAD}, {normalize_enum("ADD"), Elemwise::Mode::ADD}, {normalize_enum("FLOOR_DIV"), Elemwise::Mode::FLOOR_DIV}, {normalize_enum("MAX"), Elemwise::Mode::MAX}, {normalize_enum("MIN"), Elemwise::Mode::MIN}, {normalize_enum("MOD"), Elemwise::Mode::MOD}, {normalize_enum("MUL"), Elemwise::Mode::MUL}, {normalize_enum("POW"), Elemwise::Mode::POW}, {normalize_enum("SIGMOID_GRAD"), Elemwise::Mode::SIGMOID_GRAD}, {normalize_enum("SUB"), Elemwise::Mode::SUB}, {normalize_enum("SWITCH_GT0"), Elemwise::Mode::SWITCH_GT0}, {normalize_enum("TANH_GRAD"), Elemwise::Mode::TANH_GRAD}, {normalize_enum("TRUE_DIV"), Elemwise::Mode::TRUE_DIV}, {normalize_enum("LOG_SUM_EXP"), Elemwise::Mode::LOG_SUM_EXP}, {normalize_enum("LT"), Elemwise::Mode::LT}, {normalize_enum("LEQ"), Elemwise::Mode::LEQ}, {normalize_enum("EQ"), Elemwise::Mode::EQ}, {normalize_enum("SHL"), Elemwise::Mode::SHL}, {normalize_enum("SHR"), Elemwise::Mode::SHR}, {normalize_enum("COND_LEQ_MOV"), Elemwise::Mode::COND_LEQ_MOV}, {normalize_enum("FUSE_MUL_ADD3"), Elemwise::Mode::FUSE_MUL_ADD3}, {normalize_enum("FUSE_MUL_ADD4"), Elemwise::Mode::FUSE_MUL_ADD4}, {normalize_enum("FUSE_ADD_RELU"), Elemwise::Mode::FUSE_ADD_RELU}, {normalize_enum("FUSE_ADD_SIGMOID"), Elemwise::Mode::FUSE_ADD_SIGMOID}, {normalize_enum("FUSE_ADD_TANH"), Elemwise::Mode::FUSE_ADD_TANH}, {normalize_enum("FAST_TANH"), Elemwise::Mode::FAST_TANH}, {normalize_enum("FAST_TANH_GRAD"), Elemwise::Mode::FAST_TANH_GRAD}, {normalize_enum("ROUND"), Elemwise::Mode::ROUND}, {normalize_enum("RMULH"), Elemwise::Mode::RMULH}, {normalize_enum("ATAN2"), Elemwise::Mode::ATAN2}, {normalize_enum("ERF"), Elemwise::Mode::ERF}, {normalize_enum("ERFINV"), Elemwise::Mode::ERFINV}, {normalize_enum("ERFC"), Elemwise::Mode::ERFC}, {normalize_enum("ERFCINV"), Elemwise::Mode::ERFCINV}, {normalize_enum("H_SWISH"), Elemwise::Mode::H_SWISH}, {normalize_enum("H_SWISH_GRAD"), Elemwise::Mode::H_SWISH_GRAD}, {normalize_enum("FUSE_ADD_H_SWISH"), Elemwise::Mode::FUSE_ADD_H_SWISH}, {normalize_enum("NOT"), Elemwise::Mode::NOT}, {normalize_enum("AND"), Elemwise::Mode::AND}, {normalize_enum("OR"), Elemwise::Mode::OR}, {normalize_enum("XOR"), Elemwise::Mode::XOR}, {normalize_enum("SILU"), Elemwise::Mode::SILU}, {normalize_enum("SILU_GRAD"), Elemwise::Mode::SILU_GRAD}, {normalize_enum("GELU"), Elemwise::Mode::GELU}, {normalize_enum("GELU_GRAD"), Elemwise::Mode::GELU_GRAD}, {normalize_enum("COND_LT_MOV"), Elemwise::Mode::COND_LT_MOV}, {normalize_enum("NEQ"), Elemwise::Mode::NEQ}, {normalize_enum("ISNAN"), Elemwise::Mode::ISNAN}, {normalize_enum("ISINF"), Elemwise::Mode::ISINF}, {normalize_enum("SINH"), Elemwise::Mode::SINH}, {normalize_enum("COSH"), Elemwise::Mode::COSH}, {normalize_enum("ASINH"), Elemwise::Mode::ASINH}, {normalize_enum("ACOSH"), Elemwise::Mode::ACOSH}, {normalize_enum("ATANH"), Elemwise::Mode::ATANH}, {normalize_enum("TAN"), Elemwise::Mode::TAN}, {normalize_enum("ASINH_GRAD"), Elemwise::Mode::ASINH_GRAD}, {normalize_enum("ACOSH_GRAD"), Elemwise::Mode::ACOSH_GRAD}, {normalize_enum("ATANH_GRAD"), Elemwise::Mode::ATANH_GRAD}, {normalize_enum("PRELU"), Elemwise::Mode::PRELU}, {normalize_enum("CLIP"), Elemwise::Mode::CLIP}, {normalize_enum("PRELU_GRAD"), Elemwise::Mode::PRELU_GRAD}, {normalize_enum("SOFTPLUS"), Elemwise::Mode::SOFTPLUS}, {normalize_enum("SOFTPLUS_GRAD"), Elemwise::Mode::SOFTPLUS_GRAD}, {normalize_enum("RELU6"), Elemwise::Mode::RELU6}, {normalize_enum("RELU6_GRAD"), Elemwise::Mode::RELU6_GRAD}, {normalize_enum("HSIGMOID"), Elemwise::Mode::HSIGMOID}, {normalize_enum("HSIGMOID_GRAD"), Elemwise::Mode::HSIGMOID_GRAD}, {normalize_enum("LOGSIGMOID"), Elemwise::Mode::LOGSIGMOID}, {normalize_enum("SQRT"), Elemwise::Mode::SQRT}, {normalize_enum("SQUARE"), Elemwise::Mode::SQUARE}, {normalize_enum("SIGN"), Elemwise::Mode::SIGN}};
template<> PyObject* EnumWrapper<Elemwise::Mode>::pyobj_insts[86] = {nullptr};

void _init_py_Elemwise_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Elemwise::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Elemwise::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Elemwise::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Elemwise::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Elemwise.Mode",
        // basicsize
        sizeof(EnumWrapper<Elemwise::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Elemwise.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::RELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RELU", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ABS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ABS", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ACOS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ACOS", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ASIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ASIN", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::CEIL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CEIL", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::COS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "COS", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::EXP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "EXP", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[6] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::EXPM1;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "EXPM1", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[7] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FLOOR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOOR", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[8] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::LOG;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LOG", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[9] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::LOG1P;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LOG1P", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[10] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::NEGATE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NEGATE", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[11] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SIGMOID", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[12] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SIN", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[13] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::TANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TANH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[14] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ABS_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ABS_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[15] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ADD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ADD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[16] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FLOOR_DIV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOOR_DIV", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[17] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::MAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MAX", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[18] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::MIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MIN", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[19] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::MOD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MOD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[20] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::MUL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MUL", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[21] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::POW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "POW", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[22] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SIGMOID_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SIGMOID_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[23] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SUB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SUB", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[24] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SWITCH_GT0;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SWITCH_GT0", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[25] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::TANH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TANH_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[26] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::TRUE_DIV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TRUE_DIV", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[27] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::LOG_SUM_EXP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LOG_SUM_EXP", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[28] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::LT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LT", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[29] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::LEQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LEQ", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[30] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::EQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "EQ", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[31] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SHL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SHL", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[32] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SHR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SHR", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[33] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::COND_LEQ_MOV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "COND_LEQ_MOV", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[34] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FUSE_MUL_ADD3;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_MUL_ADD3", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[35] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FUSE_MUL_ADD4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_MUL_ADD4", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[36] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FUSE_ADD_RELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_ADD_RELU", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[37] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FUSE_ADD_SIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_ADD_SIGMOID", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[38] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FUSE_ADD_TANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_ADD_TANH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[39] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FAST_TANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FAST_TANH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[40] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FAST_TANH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FAST_TANH_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[41] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ROUND;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ROUND", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[42] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::RMULH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RMULH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[43] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ATAN2;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ATAN2", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[44] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ERF;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ERF", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[45] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ERFINV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ERFINV", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[46] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ERFC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ERFC", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[47] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ERFCINV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ERFCINV", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[48] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::H_SWISH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "H_SWISH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[49] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::H_SWISH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "H_SWISH_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[50] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::FUSE_ADD_H_SWISH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_ADD_H_SWISH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[51] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::NOT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NOT", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[52] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::AND;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "AND", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[53] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::OR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "OR", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[54] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::XOR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "XOR", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[55] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SILU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SILU", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[56] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SILU_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SILU_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[57] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::GELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "GELU", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[58] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::GELU_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "GELU_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[59] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::COND_LT_MOV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "COND_LT_MOV", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[60] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::NEQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NEQ", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[61] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ISNAN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ISNAN", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[62] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ISINF;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ISINF", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[63] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SINH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SINH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[64] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::COSH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "COSH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[65] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ASINH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ASINH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[66] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ACOSH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ACOSH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[67] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ATANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ATANH", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[68] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::TAN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TAN", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[69] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ASINH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ASINH_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[70] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ACOSH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ACOSH_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[71] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::ATANH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ATANH_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[72] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::PRELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "PRELU", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[73] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::CLIP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CLIP", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[74] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::PRELU_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "PRELU_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[75] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SOFTPLUS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SOFTPLUS", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[76] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SOFTPLUS_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SOFTPLUS_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[77] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::RELU6;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RELU6", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[78] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::RELU6_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RELU6_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[79] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::HSIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "HSIGMOID", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[80] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::HSIGMOID_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "HSIGMOID_GRAD", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[81] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::LOGSIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LOGSIGMOID", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[82] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SQRT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SQRT", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[83] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SQUARE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SQUARE", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[84] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Elemwise::Mode>*>(inst)->value = Elemwise::Mode::SIGN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SIGN", inst) >= 0);
    EnumWrapper<Elemwise::Mode>::pyobj_insts[85] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Elemwise) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Elemwise)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Elemwise)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Elemwise)

int PyOp(Elemwise)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "scope", NULL};
    PyObject *mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &mode, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Elemwise)*>(self)->inst().mode =
                    py::cast<decltype(Elemwise::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Elemwise)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(Elemwise, mode), py_set_generic(Elemwise, mode), const_cast<char*>("mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Elemwise)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Elemwise)::getstate, METH_NOARGS, "Elemwise getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Elemwise)::setstate, METH_VARARGS, "Elemwise setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Elemwise)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Elemwise)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Elemwise)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Elemwise)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ...) -> None\n"
};

void _init_py_Elemwise(py::module m) {
    using py_op = PyOp(Elemwise);
    auto& py_type = PyOpType(Elemwise);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Elemwise";
    py_type.tp_basicsize = sizeof(PyOp(Elemwise));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Elemwise";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Elemwise), &PyOp(Elemwise)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Elemwise_Mode(py_type);

    PyType_Modified(&py_type);
    m.add_object("Elemwise", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Elemwise::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<ElemwiseMultiType::Mode> {
    static constexpr const char *name = "ElemwiseMultiType.Mode";
    static constexpr std::underlying_type_t<ElemwiseMultiType::Mode> max = 64 - 1;
};
template<> PyTypeObject* EnumWrapper<ElemwiseMultiType::Mode>::type = nullptr;

template<> const char*
EnumWrapper<ElemwiseMultiType::Mode>::members[] = {"FUSE_MUL_ADD3_INT16x32x32x32", "FUSE_MUL_ADD3_IXxF32xF32xI8", "ROUND_SHR_SATURATE_IXxI8xI8", "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8", "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8", "ROUND_SHR_SATURATE_IXxI8xI16", "QADD", "QFUSE_ADD_RELU", "QMUL", "QMIN", "QMAX", "QSUB", "QTRUE_DIV", "QFUSE_ADD_SIGMOID", "QFUSE_ADD_TANH", "QRELU", "QABS", "QSIGMOID", "QEXP", "QTANH", "QFUSE_MUL_ADD3", "QFAST_TANH", "QNEGATE", "QACOS", "QASIN", "QCEIL", "QCOS", "QEXPM1", "QFLOOR", "QLOG", "QLOG1P", "QSIN", "QROUND", "QERF", "QERFINV", "QERFC", "QERFCINV", "QABS_GRAD", "QFLOOR_DIV", "QMOD", "QSIGMOID_GRAD", "QSWITCH_GT0", "QTANH_GRAD", "QLT", "QLEQ", "QEQ", "QPOW", "QLOG_SUM_EXP", "QFAST_TANH_GRAD", "QATAN2", "QCOND_LEQ_MOV", "QH_SWISH", "QFUSE_ADD_H_SWISH", "QH_SWISH_GRAD", "FUSE_MUL_ADD3_INT16xF32xF32xF32", "MUL_INT16xF32xF32", "FUSE_MUL_ADD3_UINT8xF32xF32xF32", "QCOND_LT_MOV", "EQ", "NEQ", "LT", "LEQ", "ISNAN", "ISINF"};

template<> std::unordered_map<std::string, ElemwiseMultiType::Mode>
EnumWrapper<ElemwiseMultiType::Mode>::mem2value = {{normalize_enum("FUSE_MUL_ADD3_INT16x32x32x32"), ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32}, {normalize_enum("FUSE_MUL_ADD3_IXxF32xF32xI8"), ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8}, {normalize_enum("ROUND_SHR_SATURATE_IXxI8xI8"), ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8}, {normalize_enum("FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8"), ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8}, {normalize_enum("FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8"), ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8}, {normalize_enum("ROUND_SHR_SATURATE_IXxI8xI16"), ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16}, {normalize_enum("QADD"), ElemwiseMultiType::Mode::QADD}, {normalize_enum("QFUSE_ADD_RELU"), ElemwiseMultiType::Mode::QFUSE_ADD_RELU}, {normalize_enum("QMUL"), ElemwiseMultiType::Mode::QMUL}, {normalize_enum("QMIN"), ElemwiseMultiType::Mode::QMIN}, {normalize_enum("QMAX"), ElemwiseMultiType::Mode::QMAX}, {normalize_enum("QSUB"), ElemwiseMultiType::Mode::QSUB}, {normalize_enum("QTRUE_DIV"), ElemwiseMultiType::Mode::QTRUE_DIV}, {normalize_enum("QFUSE_ADD_SIGMOID"), ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID}, {normalize_enum("QFUSE_ADD_TANH"), ElemwiseMultiType::Mode::QFUSE_ADD_TANH}, {normalize_enum("QRELU"), ElemwiseMultiType::Mode::QRELU}, {normalize_enum("QABS"), ElemwiseMultiType::Mode::QABS}, {normalize_enum("QSIGMOID"), ElemwiseMultiType::Mode::QSIGMOID}, {normalize_enum("QEXP"), ElemwiseMultiType::Mode::QEXP}, {normalize_enum("QTANH"), ElemwiseMultiType::Mode::QTANH}, {normalize_enum("QFUSE_MUL_ADD3"), ElemwiseMultiType::Mode::QFUSE_MUL_ADD3}, {normalize_enum("QFAST_TANH"), ElemwiseMultiType::Mode::QFAST_TANH}, {normalize_enum("QNEGATE"), ElemwiseMultiType::Mode::QNEGATE}, {normalize_enum("QACOS"), ElemwiseMultiType::Mode::QACOS}, {normalize_enum("QASIN"), ElemwiseMultiType::Mode::QASIN}, {normalize_enum("QCEIL"), ElemwiseMultiType::Mode::QCEIL}, {normalize_enum("QCOS"), ElemwiseMultiType::Mode::QCOS}, {normalize_enum("QEXPM1"), ElemwiseMultiType::Mode::QEXPM1}, {normalize_enum("QFLOOR"), ElemwiseMultiType::Mode::QFLOOR}, {normalize_enum("QLOG"), ElemwiseMultiType::Mode::QLOG}, {normalize_enum("QLOG1P"), ElemwiseMultiType::Mode::QLOG1P}, {normalize_enum("QSIN"), ElemwiseMultiType::Mode::QSIN}, {normalize_enum("QROUND"), ElemwiseMultiType::Mode::QROUND}, {normalize_enum("QERF"), ElemwiseMultiType::Mode::QERF}, {normalize_enum("QERFINV"), ElemwiseMultiType::Mode::QERFINV}, {normalize_enum("QERFC"), ElemwiseMultiType::Mode::QERFC}, {normalize_enum("QERFCINV"), ElemwiseMultiType::Mode::QERFCINV}, {normalize_enum("QABS_GRAD"), ElemwiseMultiType::Mode::QABS_GRAD}, {normalize_enum("QFLOOR_DIV"), ElemwiseMultiType::Mode::QFLOOR_DIV}, {normalize_enum("QMOD"), ElemwiseMultiType::Mode::QMOD}, {normalize_enum("QSIGMOID_GRAD"), ElemwiseMultiType::Mode::QSIGMOID_GRAD}, {normalize_enum("QSWITCH_GT0"), ElemwiseMultiType::Mode::QSWITCH_GT0}, {normalize_enum("QTANH_GRAD"), ElemwiseMultiType::Mode::QTANH_GRAD}, {normalize_enum("QLT"), ElemwiseMultiType::Mode::QLT}, {normalize_enum("QLEQ"), ElemwiseMultiType::Mode::QLEQ}, {normalize_enum("QEQ"), ElemwiseMultiType::Mode::QEQ}, {normalize_enum("QPOW"), ElemwiseMultiType::Mode::QPOW}, {normalize_enum("QLOG_SUM_EXP"), ElemwiseMultiType::Mode::QLOG_SUM_EXP}, {normalize_enum("QFAST_TANH_GRAD"), ElemwiseMultiType::Mode::QFAST_TANH_GRAD}, {normalize_enum("QATAN2"), ElemwiseMultiType::Mode::QATAN2}, {normalize_enum("QCOND_LEQ_MOV"), ElemwiseMultiType::Mode::QCOND_LEQ_MOV}, {normalize_enum("QH_SWISH"), ElemwiseMultiType::Mode::QH_SWISH}, {normalize_enum("QFUSE_ADD_H_SWISH"), ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH}, {normalize_enum("QH_SWISH_GRAD"), ElemwiseMultiType::Mode::QH_SWISH_GRAD}, {normalize_enum("FUSE_MUL_ADD3_INT16xF32xF32xF32"), ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32}, {normalize_enum("MUL_INT16xF32xF32"), ElemwiseMultiType::Mode::MUL_INT16xF32xF32}, {normalize_enum("FUSE_MUL_ADD3_UINT8xF32xF32xF32"), ElemwiseMultiType::Mode::FUSE_MUL_ADD3_UINT8xF32xF32xF32}, {normalize_enum("QCOND_LT_MOV"), ElemwiseMultiType::Mode::QCOND_LT_MOV}, {normalize_enum("EQ"), ElemwiseMultiType::Mode::EQ}, {normalize_enum("NEQ"), ElemwiseMultiType::Mode::NEQ}, {normalize_enum("LT"), ElemwiseMultiType::Mode::LT}, {normalize_enum("LEQ"), ElemwiseMultiType::Mode::LEQ}, {normalize_enum("ISNAN"), ElemwiseMultiType::Mode::ISNAN}, {normalize_enum("ISINF"), ElemwiseMultiType::Mode::ISINF}};
template<> PyObject* EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[64] = {nullptr};

void _init_py_ElemwiseMultiType_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ElemwiseMultiType::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<ElemwiseMultiType::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<ElemwiseMultiType::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<ElemwiseMultiType::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.ElemwiseMultiType.Mode",
        // basicsize
        sizeof(EnumWrapper<ElemwiseMultiType::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("ElemwiseMultiType.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_MUL_ADD3_INT16x32x32x32", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_MUL_ADD3_IXxF32xF32xI8", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ROUND_SHR_SATURATE_IXxI8xI8", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ROUND_SHR_SATURATE_IXxI8xI16", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QADD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QADD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[6] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFUSE_ADD_RELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFUSE_ADD_RELU", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[7] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QMUL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QMUL", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[8] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QMIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QMIN", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[9] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QMAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QMAX", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[10] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QSUB;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QSUB", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[11] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QTRUE_DIV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QTRUE_DIV", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[12] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFUSE_ADD_SIGMOID", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[13] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFUSE_ADD_TANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFUSE_ADD_TANH", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[14] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QRELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QRELU", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[15] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QABS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QABS", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[16] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QSIGMOID;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QSIGMOID", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[17] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QEXP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QEXP", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[18] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QTANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QTANH", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[19] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFUSE_MUL_ADD3;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFUSE_MUL_ADD3", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[20] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFAST_TANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFAST_TANH", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[21] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QNEGATE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QNEGATE", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[22] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QACOS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QACOS", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[23] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QASIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QASIN", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[24] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QCEIL;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QCEIL", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[25] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QCOS;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QCOS", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[26] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QEXPM1;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QEXPM1", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[27] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFLOOR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFLOOR", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[28] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QLOG;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QLOG", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[29] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QLOG1P;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QLOG1P", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[30] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QSIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QSIN", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[31] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QROUND;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QROUND", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[32] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QERF;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QERF", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[33] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QERFINV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QERFINV", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[34] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QERFC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QERFC", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[35] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QERFCINV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QERFCINV", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[36] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QABS_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QABS_GRAD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[37] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFLOOR_DIV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFLOOR_DIV", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[38] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QMOD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QMOD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[39] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QSIGMOID_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QSIGMOID_GRAD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[40] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QSWITCH_GT0;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QSWITCH_GT0", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[41] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QTANH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QTANH_GRAD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[42] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QLT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QLT", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[43] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QLEQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QLEQ", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[44] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QEQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QEQ", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[45] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QPOW;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QPOW", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[46] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QLOG_SUM_EXP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QLOG_SUM_EXP", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[47] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFAST_TANH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFAST_TANH_GRAD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[48] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QATAN2;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QATAN2", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[49] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QCOND_LEQ_MOV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QCOND_LEQ_MOV", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[50] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QH_SWISH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QH_SWISH", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[51] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QFUSE_ADD_H_SWISH", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[52] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QH_SWISH_GRAD;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QH_SWISH_GRAD", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[53] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_MUL_ADD3_INT16xF32xF32xF32", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[54] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::MUL_INT16xF32xF32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MUL_INT16xF32xF32", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[55] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::FUSE_MUL_ADD3_UINT8xF32xF32xF32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FUSE_MUL_ADD3_UINT8xF32xF32xF32", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[56] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::QCOND_LT_MOV;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QCOND_LT_MOV", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[57] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::EQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "EQ", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[58] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::NEQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NEQ", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[59] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::LT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LT", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[60] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::LEQ;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LEQ", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[61] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::ISNAN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ISNAN", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[62] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ElemwiseMultiType::Mode>*>(inst)->value = ElemwiseMultiType::Mode::ISINF;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ISINF", inst) >= 0);
    EnumWrapper<ElemwiseMultiType::Mode>::pyobj_insts[63] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(ElemwiseMultiType) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ElemwiseMultiType)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ElemwiseMultiType)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ElemwiseMultiType)

int PyOp(ElemwiseMultiType)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "dtype", "scope", NULL};
    PyObject *mode = NULL, *dtype = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &mode, &dtype, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ElemwiseMultiType)*>(self)->inst().mode =
                    py::cast<decltype(ElemwiseMultiType::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ElemwiseMultiType)*>(self)->inst().dtype =
                    py::cast<decltype(ElemwiseMultiType::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ElemwiseMultiType)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(ElemwiseMultiType, mode), py_set_generic(ElemwiseMultiType, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(ElemwiseMultiType, dtype), py_set_generic(ElemwiseMultiType, dtype), const_cast<char*>("dtype"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ElemwiseMultiType)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ElemwiseMultiType)::getstate, METH_NOARGS, "ElemwiseMultiType getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ElemwiseMultiType)::setstate, METH_VARARGS, "ElemwiseMultiType setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ElemwiseMultiType)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ElemwiseMultiType)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ElemwiseMultiType)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ElemwiseMultiType)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., dtype: str = ...) -> None\n"
};

void _init_py_ElemwiseMultiType(py::module m) {
    using py_op = PyOp(ElemwiseMultiType);
    auto& py_type = PyOpType(ElemwiseMultiType);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ElemwiseMultiType";
    py_type.tp_basicsize = sizeof(PyOp(ElemwiseMultiType));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ElemwiseMultiType";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ElemwiseMultiType), &PyOp(ElemwiseMultiType)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_ElemwiseMultiType_Mode(py_type);

    PyType_Modified(&py_type);
    m.add_object("ElemwiseMultiType", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ElemwiseMultiType::typeinfo(), &py_type).second);
}

PyOpDefBegin(ExternOpr) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ExternOpr)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"output_shapes", serialization<decltype(opdef.output_shapes)>::dump(opdef.output_shapes)},
            {"name", serialization<decltype(opdef.name)>::dump(opdef.name)},
            {"data", serialization<decltype(opdef.data)>::dump(opdef.data)},
            {"data_len", serialization<decltype(opdef.data_len)>::dump(opdef.data_len)},
            {"output_dtypes", serialization<decltype(opdef.output_dtypes)>::dump(opdef.output_dtypes)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ExternOpr)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("output_shapes");
        if (iter != state.end()) {
            opdef.output_shapes = serialization<decltype(opdef.output_shapes)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("name");
        if (iter != state.end()) {
            opdef.name = serialization<decltype(opdef.name)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("data");
        if (iter != state.end()) {
            opdef.data = serialization<decltype(opdef.data)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("data_len");
        if (iter != state.end()) {
            opdef.data_len = serialization<decltype(opdef.data_len)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("output_dtypes");
        if (iter != state.end()) {
            opdef.output_dtypes = serialization<decltype(opdef.output_dtypes)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ExternOpr)

int PyOp(ExternOpr)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"output_shapes", "name", "data", "data_len", "output_dtypes", "scope", NULL};
    PyObject *output_shapes = NULL, *name = NULL, *data = NULL, *data_len = NULL, *output_dtypes = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOO", const_cast<char**>(kwlist), &output_shapes, &name, &data, &data_len, &output_dtypes, &scope))
    return -1;

    if (output_shapes) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ExternOpr)*>(self)->inst().output_shapes =
                    py::cast<decltype(ExternOpr::output_shapes)>(py::handle(output_shapes));
        } CATCH_ALL(-1)
    }

    if (name) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ExternOpr)*>(self)->inst().name =
                    py::cast<decltype(ExternOpr::name)>(py::handle(name));
        } CATCH_ALL(-1)
    }

    if (data) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ExternOpr)*>(self)->inst().data =
                    py::cast<decltype(ExternOpr::data)>(py::handle(data));
        } CATCH_ALL(-1)
    }

    if (data_len) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ExternOpr)*>(self)->inst().data_len =
                    py::cast<decltype(ExternOpr::data_len)>(py::handle(data_len));
        } CATCH_ALL(-1)
    }

    if (output_dtypes) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ExternOpr)*>(self)->inst().output_dtypes =
                    py::cast<decltype(ExternOpr::output_dtypes)>(py::handle(output_dtypes));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ExternOpr)::py_getsetters[] = {
    {const_cast<char*>("output_shapes"), py_get_generic(ExternOpr, output_shapes), py_set_generic(ExternOpr, output_shapes), const_cast<char*>("output_shapes"), NULL},
    {const_cast<char*>("name"), py_get_generic(ExternOpr, name), py_set_generic(ExternOpr, name), const_cast<char*>("name"), NULL},
    {const_cast<char*>("data"), py_get_generic(ExternOpr, data), py_set_generic(ExternOpr, data), const_cast<char*>("data"), NULL},
    {const_cast<char*>("data_len"), py_get_generic(ExternOpr, data_len), py_set_generic(ExternOpr, data_len), const_cast<char*>("data_len"), NULL},
    {const_cast<char*>("output_dtypes"), py_get_generic(ExternOpr, output_dtypes), py_set_generic(ExternOpr, output_dtypes), const_cast<char*>("output_dtypes"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ExternOpr)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ExternOpr)::getstate, METH_NOARGS, "ExternOpr getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ExternOpr)::setstate, METH_VARARGS, "ExternOpr setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ExternOpr)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ExternOpr)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ExternOpr)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ExternOpr)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, output_shapes: list[list[int]] = ..., name: str = ..., data: str = ..., data_len: int = ..., output_dtypes: list[str] = ...) -> None\n"
};

void _init_py_ExternOpr(py::module m) {
    using py_op = PyOp(ExternOpr);
    auto& py_type = PyOpType(ExternOpr);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ExternOpr";
    py_type.tp_basicsize = sizeof(PyOp(ExternOpr));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ExternOpr";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ExternOpr), &PyOp(ExternOpr)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("ExternOpr", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ExternOpr::typeinfo(), &py_type).second);
}

PyOpDefBegin(Eye) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Eye)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"k", serialization<decltype(opdef.k)>::dump(opdef.k)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)},
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Eye)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("k");
        if (iter != state.end()) {
            opdef.k = serialization<decltype(opdef.k)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Eye)

int PyOp(Eye)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"k", "dtype", "comp_node", "scope", NULL};
    PyObject *k = NULL, *dtype = NULL, *comp_node = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", const_cast<char**>(kwlist), &k, &dtype, &comp_node, &scope))
    return -1;

    if (k) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Eye)*>(self)->inst().k =
                    py::cast<decltype(Eye::k)>(py::handle(k));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Eye)*>(self)->inst().dtype =
                    py::cast<decltype(Eye::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Eye)*>(self)->inst().comp_node =
                    py::cast<decltype(Eye::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Eye)::py_getsetters[] = {
    {const_cast<char*>("k"), py_get_generic(Eye, k), py_set_generic(Eye, k), const_cast<char*>("k"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(Eye, dtype), py_set_generic(Eye, dtype), const_cast<char*>("dtype"), NULL},
    {const_cast<char*>("comp_node"), py_get_generic(Eye, comp_node), py_set_generic(Eye, comp_node), const_cast<char*>("comp_node"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Eye)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Eye)::getstate, METH_NOARGS, "Eye getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Eye)::setstate, METH_VARARGS, "Eye setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Eye)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Eye)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Eye)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Eye)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, k: int = ..., dtype: str = ..., comp_node: str = ...) -> None\n"
};

void _init_py_Eye(py::module m) {
    using py_op = PyOp(Eye);
    auto& py_type = PyOpType(Eye);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Eye";
    py_type.tp_basicsize = sizeof(PyOp(Eye));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Eye";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Eye), &PyOp(Eye)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Eye", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Eye::typeinfo(), &py_type).second);
}

PyOpDefBegin(FakeQuant) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(FakeQuant)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"qmin", serialization<decltype(opdef.qmin)>::dump(opdef.qmin)},
            {"qmax", serialization<decltype(opdef.qmax)>::dump(opdef.qmax)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(FakeQuant)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("qmin");
        if (iter != state.end()) {
            opdef.qmin = serialization<decltype(opdef.qmin)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("qmax");
        if (iter != state.end()) {
            opdef.qmax = serialization<decltype(opdef.qmax)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(FakeQuant)

int PyOp(FakeQuant)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"qmin", "qmax", "scope", NULL};
    PyObject *qmin = NULL, *qmax = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &qmin, &qmax, &scope))
    return -1;

    if (qmin) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(FakeQuant)*>(self)->inst().qmin =
                    py::cast<decltype(FakeQuant::qmin)>(py::handle(qmin));
        } CATCH_ALL(-1)
    }

    if (qmax) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(FakeQuant)*>(self)->inst().qmax =
                    py::cast<decltype(FakeQuant::qmax)>(py::handle(qmax));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(FakeQuant)::py_getsetters[] = {
    {const_cast<char*>("qmin"), py_get_generic(FakeQuant, qmin), py_set_generic(FakeQuant, qmin), const_cast<char*>("qmin"), NULL},
    {const_cast<char*>("qmax"), py_get_generic(FakeQuant, qmax), py_set_generic(FakeQuant, qmax), const_cast<char*>("qmax"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(FakeQuant)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(FakeQuant)::getstate, METH_NOARGS, "FakeQuant getstate"},
    {const_cast<char*>("__setstate__"), PyOp(FakeQuant)::setstate, METH_VARARGS, "FakeQuant setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(FakeQuant)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(FakeQuant)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(FakeQuant)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(FakeQuant)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, qmin: int = ..., qmax: int = ...) -> None\n"
};

void _init_py_FakeQuant(py::module m) {
    using py_op = PyOp(FakeQuant);
    auto& py_type = PyOpType(FakeQuant);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.FakeQuant";
    py_type.tp_basicsize = sizeof(PyOp(FakeQuant));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "FakeQuant";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(FakeQuant), &PyOp(FakeQuant)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("FakeQuant", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(FakeQuant::typeinfo(), &py_type).second);
}

PyOpDefBegin(FastpathCopy) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(FastpathCopy)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(FastpathCopy)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(FastpathCopy)

int PyOp(FastpathCopy)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(FastpathCopy)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(FastpathCopy)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(FastpathCopy)::getstate, METH_NOARGS, "FastpathCopy getstate"},
    {const_cast<char*>("__setstate__"), PyOp(FastpathCopy)::setstate, METH_VARARGS, "FastpathCopy setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(FastpathCopy)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(FastpathCopy)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(FastpathCopy)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(FastpathCopy)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_FastpathCopy(py::module m) {
    using py_op = PyOp(FastpathCopy);
    auto& py_type = PyOpType(FastpathCopy);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.FastpathCopy";
    py_type.tp_basicsize = sizeof(PyOp(FastpathCopy));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "FastpathCopy";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(FastpathCopy), &PyOp(FastpathCopy)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("FastpathCopy", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(FastpathCopy::typeinfo(), &py_type).second);
}

PyOpDefBegin(GammaRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(GammaRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(GammaRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(GammaRNG)

int PyOp(GammaRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "handle", "scope", NULL};
    PyObject *seed = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &seed, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GammaRNG)*>(self)->inst().seed =
                    py::cast<decltype(GammaRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GammaRNG)*>(self)->inst().handle =
                    py::cast<decltype(GammaRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(GammaRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(GammaRNG, seed), py_set_generic(GammaRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("handle"), py_get_generic(GammaRNG, handle), py_set_generic(GammaRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(GammaRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(GammaRNG)::getstate, METH_NOARGS, "GammaRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(GammaRNG)::setstate, METH_VARARGS, "GammaRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(GammaRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(GammaRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(GammaRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(GammaRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., handle: int = ...) -> None\n"
};

void _init_py_GammaRNG(py::module m) {
    using py_op = PyOp(GammaRNG);
    auto& py_type = PyOpType(GammaRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.GammaRNG";
    py_type.tp_basicsize = sizeof(PyOp(GammaRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "GammaRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(GammaRNG), &PyOp(GammaRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("GammaRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(GammaRNG::typeinfo(), &py_type).second);
}

PyOpDefBegin(GaussianRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"mean", serialization<decltype(opdef.mean)>::dump(opdef.mean)},
            {"std", serialization<decltype(opdef.std)>::dump(opdef.std)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("mean");
        if (iter != state.end()) {
            opdef.mean = serialization<decltype(opdef.mean)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("std");
        if (iter != state.end()) {
            opdef.std = serialization<decltype(opdef.std)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(GaussianRNG)

int PyOp(GaussianRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "mean", "std", "dtype", "handle", "scope", NULL};
    PyObject *seed = NULL, *mean = NULL, *std = NULL, *dtype = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOO", const_cast<char**>(kwlist), &seed, &mean, &std, &dtype, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst().seed =
                    py::cast<decltype(GaussianRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (mean) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst().mean =
                    py::cast<decltype(GaussianRNG::mean)>(py::handle(mean));
        } CATCH_ALL(-1)
    }

    if (std) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst().std =
                    py::cast<decltype(GaussianRNG::std)>(py::handle(std));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst().dtype =
                    py::cast<decltype(GaussianRNG::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GaussianRNG)*>(self)->inst().handle =
                    py::cast<decltype(GaussianRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(GaussianRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(GaussianRNG, seed), py_set_generic(GaussianRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("mean"), py_get_generic(GaussianRNG, mean), py_set_generic(GaussianRNG, mean), const_cast<char*>("mean"), NULL},
    {const_cast<char*>("std"), py_get_generic(GaussianRNG, std), py_set_generic(GaussianRNG, std), const_cast<char*>("std"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(GaussianRNG, dtype), py_set_generic(GaussianRNG, dtype), const_cast<char*>("dtype"), NULL},
    {const_cast<char*>("handle"), py_get_generic(GaussianRNG, handle), py_set_generic(GaussianRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(GaussianRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(GaussianRNG)::getstate, METH_NOARGS, "GaussianRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(GaussianRNG)::setstate, METH_VARARGS, "GaussianRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(GaussianRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(GaussianRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(GaussianRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(GaussianRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., mean: float = ..., std: float = ..., dtype: str = ..., handle: int = ...) -> None\n"
};

void _init_py_GaussianRNG(py::module m) {
    using py_op = PyOp(GaussianRNG);
    auto& py_type = PyOpType(GaussianRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.GaussianRNG";
    py_type.tp_basicsize = sizeof(PyOp(GaussianRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "GaussianRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(GaussianRNG), &PyOp(GaussianRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("GaussianRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(GaussianRNG::typeinfo(), &py_type).second);
}

PyOpDefBegin(GetVarShape) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(GetVarShape)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(GetVarShape)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(GetVarShape)

int PyOp(GetVarShape)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "scope", NULL};
    PyObject *axis = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &axis, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GetVarShape)*>(self)->inst().axis =
                    py::cast<decltype(GetVarShape::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(GetVarShape)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(GetVarShape, axis), py_set_generic(GetVarShape, axis), const_cast<char*>("axis"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(GetVarShape)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(GetVarShape)::getstate, METH_NOARGS, "GetVarShape getstate"},
    {const_cast<char*>("__setstate__"), PyOp(GetVarShape)::setstate, METH_VARARGS, "GetVarShape setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(GetVarShape)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(GetVarShape)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(GetVarShape)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(GetVarShape)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ...) -> None\n"
};

void _init_py_GetVarShape(py::module m) {
    using py_op = PyOp(GetVarShape);
    auto& py_type = PyOpType(GetVarShape);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.GetVarShape";
    py_type.tp_basicsize = sizeof(PyOp(GetVarShape));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "GetVarShape";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(GetVarShape), &PyOp(GetVarShape)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("GetVarShape", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(GetVarShape::typeinfo(), &py_type).second);
}

void _init_py_GroupLocal_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<GroupLocal::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_GroupLocal_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<GroupLocal::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_GroupLocal_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<GroupLocal::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_GroupLocal_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<GroupLocal::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(GroupLocal) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(GroupLocal)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(GroupLocal)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(GroupLocal)

int PyOp(GroupLocal)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().mode =
                    py::cast<decltype(GroupLocal::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().pad_h =
                    py::cast<decltype(GroupLocal::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().pad_w =
                    py::cast<decltype(GroupLocal::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().stride_h =
                    py::cast<decltype(GroupLocal::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().stride_w =
                    py::cast<decltype(GroupLocal::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().dilate_h =
                    py::cast<decltype(GroupLocal::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().dilate_w =
                    py::cast<decltype(GroupLocal::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().sparse =
                    py::cast<decltype(GroupLocal::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().format =
                    py::cast<decltype(GroupLocal::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupLocal)*>(self)->inst().compute_mode =
                    py::cast<decltype(GroupLocal::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(GroupLocal)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(GroupLocal, mode), py_set_generic(GroupLocal, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(GroupLocal, pad_h), py_set_generic(GroupLocal, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(GroupLocal, pad_w), py_set_generic(GroupLocal, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(GroupLocal, stride_h), py_set_generic(GroupLocal, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(GroupLocal, stride_w), py_set_generic(GroupLocal, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(GroupLocal, dilate_h), py_set_generic(GroupLocal, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(GroupLocal, dilate_w), py_set_generic(GroupLocal, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(GroupLocal, sparse), py_set_generic(GroupLocal, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(GroupLocal, format), py_set_generic(GroupLocal, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(GroupLocal, compute_mode), py_set_generic(GroupLocal, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(GroupLocal)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(GroupLocal)::getstate, METH_NOARGS, "GroupLocal getstate"},
    {const_cast<char*>("__setstate__"), PyOp(GroupLocal)::setstate, METH_VARARGS, "GroupLocal setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(GroupLocal)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(GroupLocal)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(GroupLocal)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(GroupLocal)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ...) -> None\n"
};

void _init_py_GroupLocal(py::module m) {
    using py_op = PyOp(GroupLocal);
    auto& py_type = PyOpType(GroupLocal);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.GroupLocal";
    py_type.tp_basicsize = sizeof(PyOp(GroupLocal));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "GroupLocal";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(GroupLocal), &PyOp(GroupLocal)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_GroupLocal_Mode(py_type);
    _init_py_GroupLocal_Sparse(py_type);
    _init_py_GroupLocal_Format(py_type);
    _init_py_GroupLocal_ComputeMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("GroupLocal", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(GroupLocal::typeinfo(), &py_type).second);
}

void _init_py_GroupNorm_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<GroupNorm::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(GroupNorm) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(GroupNorm)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"affine", serialization<decltype(opdef.affine)>::dump(opdef.affine)},
            {"eps", serialization<decltype(opdef.eps)>::dump(opdef.eps)},
            {"group", serialization<decltype(opdef.group)>::dump(opdef.group)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(GroupNorm)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("affine");
        if (iter != state.end()) {
            opdef.affine = serialization<decltype(opdef.affine)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("eps");
        if (iter != state.end()) {
            opdef.eps = serialization<decltype(opdef.eps)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("group");
        if (iter != state.end()) {
            opdef.group = serialization<decltype(opdef.group)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(GroupNorm)

int PyOp(GroupNorm)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"affine", "eps", "group", "format", "scope", NULL};
    PyObject *affine = NULL, *eps = NULL, *group = NULL, *format = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &affine, &eps, &group, &format, &scope))
    return -1;

    if (affine) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupNorm)*>(self)->inst().affine =
                    py::cast<decltype(GroupNorm::affine)>(py::handle(affine));
        } CATCH_ALL(-1)
    }

    if (eps) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupNorm)*>(self)->inst().eps =
                    py::cast<decltype(GroupNorm::eps)>(py::handle(eps));
        } CATCH_ALL(-1)
    }

    if (group) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupNorm)*>(self)->inst().group =
                    py::cast<decltype(GroupNorm::group)>(py::handle(group));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(GroupNorm)*>(self)->inst().format =
                    py::cast<decltype(GroupNorm::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(GroupNorm)::py_getsetters[] = {
    {const_cast<char*>("affine"), py_get_generic(GroupNorm, affine), py_set_generic(GroupNorm, affine), const_cast<char*>("affine"), NULL},
    {const_cast<char*>("eps"), py_get_generic(GroupNorm, eps), py_set_generic(GroupNorm, eps), const_cast<char*>("eps"), NULL},
    {const_cast<char*>("group"), py_get_generic(GroupNorm, group), py_set_generic(GroupNorm, group), const_cast<char*>("group"), NULL},
    {const_cast<char*>("format"), py_get_generic(GroupNorm, format), py_set_generic(GroupNorm, format), const_cast<char*>("format"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(GroupNorm)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(GroupNorm)::getstate, METH_NOARGS, "GroupNorm getstate"},
    {const_cast<char*>("__setstate__"), PyOp(GroupNorm)::setstate, METH_VARARGS, "GroupNorm setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(GroupNorm)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(GroupNorm)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(GroupNorm)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(GroupNorm)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, affine: bool = ..., eps: float = ..., group: int = ..., format: Union[str, Format] = ...) -> None\n"
};

void _init_py_GroupNorm(py::module m) {
    using py_op = PyOp(GroupNorm);
    auto& py_type = PyOpType(GroupNorm);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.GroupNorm";
    py_type.tp_basicsize = sizeof(PyOp(GroupNorm));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "GroupNorm";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(GroupNorm), &PyOp(GroupNorm)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_GroupNorm_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("GroupNorm", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(GroupNorm::typeinfo(), &py_type).second);
}

PyOpDefBegin(Identity) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Identity)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Identity)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Identity)

int PyOp(Identity)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(Identity)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Identity)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Identity)::getstate, METH_NOARGS, "Identity getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Identity)::setstate, METH_VARARGS, "Identity setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Identity)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Identity)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Identity)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Identity)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_Identity(py::module m) {
    using py_op = PyOp(Identity);
    auto& py_type = PyOpType(Identity);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Identity";
    py_type.tp_basicsize = sizeof(PyOp(Identity));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Identity";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Identity), &PyOp(Identity)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Identity", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Identity::typeinfo(), &py_type).second);
}

PyOpDefBegin(Images2Neibs) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"window_h", serialization<decltype(opdef.window_h)>::dump(opdef.window_h)},
            {"window_w", serialization<decltype(opdef.window_w)>::dump(opdef.window_w)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("window_h");
        if (iter != state.end()) {
            opdef.window_h = serialization<decltype(opdef.window_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("window_w");
        if (iter != state.end()) {
            opdef.window_w = serialization<decltype(opdef.window_w)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Images2Neibs)

int PyOp(Images2Neibs)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "window_h", "window_w", "scope", NULL};
    PyObject *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *window_h = NULL, *window_w = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", const_cast<char**>(kwlist), &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &window_h, &window_w, &scope))
    return -1;

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().pad_h =
                    py::cast<decltype(Images2Neibs::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().pad_w =
                    py::cast<decltype(Images2Neibs::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().stride_h =
                    py::cast<decltype(Images2Neibs::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().stride_w =
                    py::cast<decltype(Images2Neibs::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().dilate_h =
                    py::cast<decltype(Images2Neibs::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().dilate_w =
                    py::cast<decltype(Images2Neibs::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (window_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().window_h =
                    py::cast<decltype(Images2Neibs::window_h)>(py::handle(window_h));
        } CATCH_ALL(-1)
    }

    if (window_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Images2Neibs)*>(self)->inst().window_w =
                    py::cast<decltype(Images2Neibs::window_w)>(py::handle(window_w));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Images2Neibs)::py_getsetters[] = {
    {const_cast<char*>("pad_h"), py_get_generic(Images2Neibs, pad_h), py_set_generic(Images2Neibs, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(Images2Neibs, pad_w), py_set_generic(Images2Neibs, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(Images2Neibs, stride_h), py_set_generic(Images2Neibs, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(Images2Neibs, stride_w), py_set_generic(Images2Neibs, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(Images2Neibs, dilate_h), py_set_generic(Images2Neibs, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(Images2Neibs, dilate_w), py_set_generic(Images2Neibs, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("window_h"), py_get_generic(Images2Neibs, window_h), py_set_generic(Images2Neibs, window_h), const_cast<char*>("window_h"), NULL},
    {const_cast<char*>("window_w"), py_get_generic(Images2Neibs, window_w), py_set_generic(Images2Neibs, window_w), const_cast<char*>("window_w"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Images2Neibs)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Images2Neibs)::getstate, METH_NOARGS, "Images2Neibs getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Images2Neibs)::setstate, METH_VARARGS, "Images2Neibs setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Images2Neibs)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Images2Neibs)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Images2Neibs)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Images2Neibs)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., window_h: int = ..., window_w: int = ...) -> None\n"
};

void _init_py_Images2Neibs(py::module m) {
    using py_op = PyOp(Images2Neibs);
    auto& py_type = PyOpType(Images2Neibs);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Images2Neibs";
    py_type.tp_basicsize = sizeof(PyOp(Images2Neibs));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Images2Neibs";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Images2Neibs), &PyOp(Images2Neibs)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Images2Neibs", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Images2Neibs::typeinfo(), &py_type).second);
}

PyOpDefBegin(IncrMeshIndexing) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IncrMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IncrMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IncrMeshIndexing)

int PyOp(IncrMeshIndexing)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IncrMeshIndexing)*>(self)->inst().items =
                    py::cast<decltype(IncrMeshIndexing::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IncrMeshIndexing)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(IncrMeshIndexing, items), py_set_generic(IncrMeshIndexing, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IncrMeshIndexing)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IncrMeshIndexing)::getstate, METH_NOARGS, "IncrMeshIndexing getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IncrMeshIndexing)::setstate, METH_VARARGS, "IncrMeshIndexing setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IncrMeshIndexing)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IncrMeshIndexing)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IncrMeshIndexing)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IncrMeshIndexing)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_IncrMeshIndexing(py::module m) {
    using py_op = PyOp(IncrMeshIndexing);
    auto& py_type = PyOpType(IncrMeshIndexing);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IncrMeshIndexing";
    py_type.tp_basicsize = sizeof(PyOp(IncrMeshIndexing));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IncrMeshIndexing";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IncrMeshIndexing), &PyOp(IncrMeshIndexing)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IncrMeshIndexing", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IncrMeshIndexing::typeinfo(), &py_type).second);
}

PyOpDefBegin(IncrSubtensor) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IncrSubtensor)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IncrSubtensor)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IncrSubtensor)

int PyOp(IncrSubtensor)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IncrSubtensor)*>(self)->inst().items =
                    py::cast<decltype(IncrSubtensor::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IncrSubtensor)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(IncrSubtensor, items), py_set_generic(IncrSubtensor, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IncrSubtensor)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IncrSubtensor)::getstate, METH_NOARGS, "IncrSubtensor getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IncrSubtensor)::setstate, METH_VARARGS, "IncrSubtensor setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IncrSubtensor)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IncrSubtensor)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IncrSubtensor)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IncrSubtensor)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_IncrSubtensor(py::module m) {
    using py_op = PyOp(IncrSubtensor);
    auto& py_type = PyOpType(IncrSubtensor);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IncrSubtensor";
    py_type.tp_basicsize = sizeof(PyOp(IncrSubtensor));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IncrSubtensor";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IncrSubtensor), &PyOp(IncrSubtensor)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IncrSubtensor", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IncrSubtensor::typeinfo(), &py_type).second);
}

PyOpDefBegin(IndexingIncrMultiAxisVec) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IndexingIncrMultiAxisVec)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IndexingIncrMultiAxisVec)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IndexingIncrMultiAxisVec)

int PyOp(IndexingIncrMultiAxisVec)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingIncrMultiAxisVec)*>(self)->inst().items =
                    py::cast<decltype(IndexingIncrMultiAxisVec::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IndexingIncrMultiAxisVec)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(IndexingIncrMultiAxisVec, items), py_set_generic(IndexingIncrMultiAxisVec, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IndexingIncrMultiAxisVec)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IndexingIncrMultiAxisVec)::getstate, METH_NOARGS, "IndexingIncrMultiAxisVec getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IndexingIncrMultiAxisVec)::setstate, METH_VARARGS, "IndexingIncrMultiAxisVec setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IndexingIncrMultiAxisVec)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IndexingIncrMultiAxisVec)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IndexingIncrMultiAxisVec)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IndexingIncrMultiAxisVec)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_IndexingIncrMultiAxisVec(py::module m) {
    using py_op = PyOp(IndexingIncrMultiAxisVec);
    auto& py_type = PyOpType(IndexingIncrMultiAxisVec);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IndexingIncrMultiAxisVec";
    py_type.tp_basicsize = sizeof(PyOp(IndexingIncrMultiAxisVec));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IndexingIncrMultiAxisVec";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IndexingIncrMultiAxisVec), &PyOp(IndexingIncrMultiAxisVec)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IndexingIncrMultiAxisVec", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IndexingIncrMultiAxisVec::typeinfo(), &py_type).second);
}

PyOpDefBegin(IndexingMultiAxisVec) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IndexingMultiAxisVec)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IndexingMultiAxisVec)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IndexingMultiAxisVec)

int PyOp(IndexingMultiAxisVec)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingMultiAxisVec)*>(self)->inst().items =
                    py::cast<decltype(IndexingMultiAxisVec::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IndexingMultiAxisVec)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(IndexingMultiAxisVec, items), py_set_generic(IndexingMultiAxisVec, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IndexingMultiAxisVec)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IndexingMultiAxisVec)::getstate, METH_NOARGS, "IndexingMultiAxisVec getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IndexingMultiAxisVec)::setstate, METH_VARARGS, "IndexingMultiAxisVec setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IndexingMultiAxisVec)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IndexingMultiAxisVec)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IndexingMultiAxisVec)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IndexingMultiAxisVec)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_IndexingMultiAxisVec(py::module m) {
    using py_op = PyOp(IndexingMultiAxisVec);
    auto& py_type = PyOpType(IndexingMultiAxisVec);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IndexingMultiAxisVec";
    py_type.tp_basicsize = sizeof(PyOp(IndexingMultiAxisVec));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IndexingMultiAxisVec";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IndexingMultiAxisVec), &PyOp(IndexingMultiAxisVec)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IndexingMultiAxisVec", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IndexingMultiAxisVec::typeinfo(), &py_type).second);
}

PyOpDefBegin(IndexingOneHot) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IndexingOneHot)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"ndim", serialization<decltype(opdef.ndim)>::dump(opdef.ndim)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IndexingOneHot)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("ndim");
        if (iter != state.end()) {
            opdef.ndim = serialization<decltype(opdef.ndim)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IndexingOneHot)

int PyOp(IndexingOneHot)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "ndim", "scope", NULL};
    PyObject *axis = NULL, *ndim = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &axis, &ndim, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingOneHot)*>(self)->inst().axis =
                    py::cast<decltype(IndexingOneHot::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (ndim) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingOneHot)*>(self)->inst().ndim =
                    py::cast<decltype(IndexingOneHot::ndim)>(py::handle(ndim));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IndexingOneHot)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(IndexingOneHot, axis), py_set_generic(IndexingOneHot, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("ndim"), py_get_generic(IndexingOneHot, ndim), py_set_generic(IndexingOneHot, ndim), const_cast<char*>("ndim"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IndexingOneHot)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IndexingOneHot)::getstate, METH_NOARGS, "IndexingOneHot getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IndexingOneHot)::setstate, METH_VARARGS, "IndexingOneHot setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IndexingOneHot)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IndexingOneHot)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IndexingOneHot)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IndexingOneHot)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ..., ndim: int = ...) -> None\n"
};

void _init_py_IndexingOneHot(py::module m) {
    using py_op = PyOp(IndexingOneHot);
    auto& py_type = PyOpType(IndexingOneHot);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IndexingOneHot";
    py_type.tp_basicsize = sizeof(PyOp(IndexingOneHot));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IndexingOneHot";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IndexingOneHot), &PyOp(IndexingOneHot)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IndexingOneHot", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IndexingOneHot::typeinfo(), &py_type).second);
}

PyOpDefBegin(IndexingSetMultiAxisVec) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IndexingSetMultiAxisVec)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IndexingSetMultiAxisVec)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IndexingSetMultiAxisVec)

int PyOp(IndexingSetMultiAxisVec)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingSetMultiAxisVec)*>(self)->inst().items =
                    py::cast<decltype(IndexingSetMultiAxisVec::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IndexingSetMultiAxisVec)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(IndexingSetMultiAxisVec, items), py_set_generic(IndexingSetMultiAxisVec, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IndexingSetMultiAxisVec)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IndexingSetMultiAxisVec)::getstate, METH_NOARGS, "IndexingSetMultiAxisVec getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IndexingSetMultiAxisVec)::setstate, METH_VARARGS, "IndexingSetMultiAxisVec setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IndexingSetMultiAxisVec)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IndexingSetMultiAxisVec)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IndexingSetMultiAxisVec)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IndexingSetMultiAxisVec)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_IndexingSetMultiAxisVec(py::module m) {
    using py_op = PyOp(IndexingSetMultiAxisVec);
    auto& py_type = PyOpType(IndexingSetMultiAxisVec);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IndexingSetMultiAxisVec";
    py_type.tp_basicsize = sizeof(PyOp(IndexingSetMultiAxisVec));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IndexingSetMultiAxisVec";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IndexingSetMultiAxisVec), &PyOp(IndexingSetMultiAxisVec)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IndexingSetMultiAxisVec", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IndexingSetMultiAxisVec::typeinfo(), &py_type).second);
}

PyOpDefBegin(IndexingSetOneHot) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(IndexingSetOneHot)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"ndim", serialization<decltype(opdef.ndim)>::dump(opdef.ndim)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(IndexingSetOneHot)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("ndim");
        if (iter != state.end()) {
            opdef.ndim = serialization<decltype(opdef.ndim)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(IndexingSetOneHot)

int PyOp(IndexingSetOneHot)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "ndim", "scope", NULL};
    PyObject *axis = NULL, *ndim = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &axis, &ndim, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingSetOneHot)*>(self)->inst().axis =
                    py::cast<decltype(IndexingSetOneHot::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (ndim) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(IndexingSetOneHot)*>(self)->inst().ndim =
                    py::cast<decltype(IndexingSetOneHot::ndim)>(py::handle(ndim));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(IndexingSetOneHot)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(IndexingSetOneHot, axis), py_set_generic(IndexingSetOneHot, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("ndim"), py_get_generic(IndexingSetOneHot, ndim), py_set_generic(IndexingSetOneHot, ndim), const_cast<char*>("ndim"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(IndexingSetOneHot)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(IndexingSetOneHot)::getstate, METH_NOARGS, "IndexingSetOneHot getstate"},
    {const_cast<char*>("__setstate__"), PyOp(IndexingSetOneHot)::setstate, METH_VARARGS, "IndexingSetOneHot setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(IndexingSetOneHot)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(IndexingSetOneHot)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(IndexingSetOneHot)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(IndexingSetOneHot)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ..., ndim: int = ...) -> None\n"
};

void _init_py_IndexingSetOneHot(py::module m) {
    using py_op = PyOp(IndexingSetOneHot);
    auto& py_type = PyOpType(IndexingSetOneHot);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.IndexingSetOneHot";
    py_type.tp_basicsize = sizeof(PyOp(IndexingSetOneHot));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "IndexingSetOneHot";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(IndexingSetOneHot), &PyOp(IndexingSetOneHot)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("IndexingSetOneHot", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(IndexingSetOneHot::typeinfo(), &py_type).second);
}

PyOpDefBegin(InplaceAdd) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(InplaceAdd)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(InplaceAdd)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(InplaceAdd)

int PyOp(InplaceAdd)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(InplaceAdd)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(InplaceAdd)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(InplaceAdd)::getstate, METH_NOARGS, "InplaceAdd getstate"},
    {const_cast<char*>("__setstate__"), PyOp(InplaceAdd)::setstate, METH_VARARGS, "InplaceAdd setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(InplaceAdd)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(InplaceAdd)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(InplaceAdd)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(InplaceAdd)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_InplaceAdd(py::module m) {
    using py_op = PyOp(InplaceAdd);
    auto& py_type = PyOpType(InplaceAdd);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.InplaceAdd";
    py_type.tp_basicsize = sizeof(PyOp(InplaceAdd));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "InplaceAdd";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(InplaceAdd), &PyOp(InplaceAdd)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("InplaceAdd", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(InplaceAdd::typeinfo(), &py_type).second);
}

PyOpDefBegin(LAMBUpdate) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"beta_1", serialization<decltype(opdef.beta_1)>::dump(opdef.beta_1)},
            {"beta_2", serialization<decltype(opdef.beta_2)>::dump(opdef.beta_2)},
            {"step", serialization<decltype(opdef.step)>::dump(opdef.step)},
            {"lr", serialization<decltype(opdef.lr)>::dump(opdef.lr)},
            {"weight_decay", serialization<decltype(opdef.weight_decay)>::dump(opdef.weight_decay)},
            {"eps", serialization<decltype(opdef.eps)>::dump(opdef.eps)},
            {"bias_correction", serialization<decltype(opdef.bias_correction)>::dump(opdef.bias_correction)},
            {"always_adapt", serialization<decltype(opdef.always_adapt)>::dump(opdef.always_adapt)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("beta_1");
        if (iter != state.end()) {
            opdef.beta_1 = serialization<decltype(opdef.beta_1)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("beta_2");
        if (iter != state.end()) {
            opdef.beta_2 = serialization<decltype(opdef.beta_2)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("step");
        if (iter != state.end()) {
            opdef.step = serialization<decltype(opdef.step)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("lr");
        if (iter != state.end()) {
            opdef.lr = serialization<decltype(opdef.lr)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("weight_decay");
        if (iter != state.end()) {
            opdef.weight_decay = serialization<decltype(opdef.weight_decay)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("eps");
        if (iter != state.end()) {
            opdef.eps = serialization<decltype(opdef.eps)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bias_correction");
        if (iter != state.end()) {
            opdef.bias_correction = serialization<decltype(opdef.bias_correction)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("always_adapt");
        if (iter != state.end()) {
            opdef.always_adapt = serialization<decltype(opdef.always_adapt)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(LAMBUpdate)

int PyOp(LAMBUpdate)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"beta_1", "beta_2", "step", "lr", "weight_decay", "eps", "bias_correction", "always_adapt", "scope", NULL};
    PyObject *beta_1 = NULL, *beta_2 = NULL, *step = NULL, *lr = NULL, *weight_decay = NULL, *eps = NULL, *bias_correction = NULL, *always_adapt = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", const_cast<char**>(kwlist), &beta_1, &beta_2, &step, &lr, &weight_decay, &eps, &bias_correction, &always_adapt, &scope))
    return -1;

    if (beta_1) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().beta_1 =
                    py::cast<decltype(LAMBUpdate::beta_1)>(py::handle(beta_1));
        } CATCH_ALL(-1)
    }

    if (beta_2) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().beta_2 =
                    py::cast<decltype(LAMBUpdate::beta_2)>(py::handle(beta_2));
        } CATCH_ALL(-1)
    }

    if (step) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().step =
                    py::cast<decltype(LAMBUpdate::step)>(py::handle(step));
        } CATCH_ALL(-1)
    }

    if (lr) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().lr =
                    py::cast<decltype(LAMBUpdate::lr)>(py::handle(lr));
        } CATCH_ALL(-1)
    }

    if (weight_decay) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().weight_decay =
                    py::cast<decltype(LAMBUpdate::weight_decay)>(py::handle(weight_decay));
        } CATCH_ALL(-1)
    }

    if (eps) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().eps =
                    py::cast<decltype(LAMBUpdate::eps)>(py::handle(eps));
        } CATCH_ALL(-1)
    }

    if (bias_correction) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().bias_correction =
                    py::cast<decltype(LAMBUpdate::bias_correction)>(py::handle(bias_correction));
        } CATCH_ALL(-1)
    }

    if (always_adapt) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LAMBUpdate)*>(self)->inst().always_adapt =
                    py::cast<decltype(LAMBUpdate::always_adapt)>(py::handle(always_adapt));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(LAMBUpdate)::py_getsetters[] = {
    {const_cast<char*>("beta_1"), py_get_generic(LAMBUpdate, beta_1), py_set_generic(LAMBUpdate, beta_1), const_cast<char*>("beta_1"), NULL},
    {const_cast<char*>("beta_2"), py_get_generic(LAMBUpdate, beta_2), py_set_generic(LAMBUpdate, beta_2), const_cast<char*>("beta_2"), NULL},
    {const_cast<char*>("step"), py_get_generic(LAMBUpdate, step), py_set_generic(LAMBUpdate, step), const_cast<char*>("step"), NULL},
    {const_cast<char*>("lr"), py_get_generic(LAMBUpdate, lr), py_set_generic(LAMBUpdate, lr), const_cast<char*>("lr"), NULL},
    {const_cast<char*>("weight_decay"), py_get_generic(LAMBUpdate, weight_decay), py_set_generic(LAMBUpdate, weight_decay), const_cast<char*>("weight_decay"), NULL},
    {const_cast<char*>("eps"), py_get_generic(LAMBUpdate, eps), py_set_generic(LAMBUpdate, eps), const_cast<char*>("eps"), NULL},
    {const_cast<char*>("bias_correction"), py_get_generic(LAMBUpdate, bias_correction), py_set_generic(LAMBUpdate, bias_correction), const_cast<char*>("bias_correction"), NULL},
    {const_cast<char*>("always_adapt"), py_get_generic(LAMBUpdate, always_adapt), py_set_generic(LAMBUpdate, always_adapt), const_cast<char*>("always_adapt"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(LAMBUpdate)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(LAMBUpdate)::getstate, METH_NOARGS, "LAMBUpdate getstate"},
    {const_cast<char*>("__setstate__"), PyOp(LAMBUpdate)::setstate, METH_VARARGS, "LAMBUpdate setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(LAMBUpdate)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(LAMBUpdate)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(LAMBUpdate)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(LAMBUpdate)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, beta_1: float = ..., beta_2: float = ..., step: float = ..., lr: float = ..., weight_decay: float = ..., eps: float = ..., bias_correction: bool = ..., always_adapt: bool = ...) -> None\n"
};

void _init_py_LAMBUpdate(py::module m) {
    using py_op = PyOp(LAMBUpdate);
    auto& py_type = PyOpType(LAMBUpdate);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.LAMBUpdate";
    py_type.tp_basicsize = sizeof(PyOp(LAMBUpdate));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "LAMBUpdate";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(LAMBUpdate), &PyOp(LAMBUpdate)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("LAMBUpdate", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(LAMBUpdate::typeinfo(), &py_type).second);
}

PyOpDefBegin(LRN) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(LRN)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"n", serialization<decltype(opdef.n)>::dump(opdef.n)},
            {"k", serialization<decltype(opdef.k)>::dump(opdef.k)},
            {"alpha", serialization<decltype(opdef.alpha)>::dump(opdef.alpha)},
            {"beta", serialization<decltype(opdef.beta)>::dump(opdef.beta)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(LRN)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("n");
        if (iter != state.end()) {
            opdef.n = serialization<decltype(opdef.n)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("k");
        if (iter != state.end()) {
            opdef.k = serialization<decltype(opdef.k)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("alpha");
        if (iter != state.end()) {
            opdef.alpha = serialization<decltype(opdef.alpha)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("beta");
        if (iter != state.end()) {
            opdef.beta = serialization<decltype(opdef.beta)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(LRN)

int PyOp(LRN)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"n", "k", "alpha", "beta", "scope", NULL};
    PyObject *n = NULL, *k = NULL, *alpha = NULL, *beta = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &n, &k, &alpha, &beta, &scope))
    return -1;

    if (n) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LRN)*>(self)->inst().n =
                    py::cast<decltype(LRN::n)>(py::handle(n));
        } CATCH_ALL(-1)
    }

    if (k) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LRN)*>(self)->inst().k =
                    py::cast<decltype(LRN::k)>(py::handle(k));
        } CATCH_ALL(-1)
    }

    if (alpha) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LRN)*>(self)->inst().alpha =
                    py::cast<decltype(LRN::alpha)>(py::handle(alpha));
        } CATCH_ALL(-1)
    }

    if (beta) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LRN)*>(self)->inst().beta =
                    py::cast<decltype(LRN::beta)>(py::handle(beta));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(LRN)::py_getsetters[] = {
    {const_cast<char*>("n"), py_get_generic(LRN, n), py_set_generic(LRN, n), const_cast<char*>("n"), NULL},
    {const_cast<char*>("k"), py_get_generic(LRN, k), py_set_generic(LRN, k), const_cast<char*>("k"), NULL},
    {const_cast<char*>("alpha"), py_get_generic(LRN, alpha), py_set_generic(LRN, alpha), const_cast<char*>("alpha"), NULL},
    {const_cast<char*>("beta"), py_get_generic(LRN, beta), py_set_generic(LRN, beta), const_cast<char*>("beta"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(LRN)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(LRN)::getstate, METH_NOARGS, "LRN getstate"},
    {const_cast<char*>("__setstate__"), PyOp(LRN)::setstate, METH_VARARGS, "LRN setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(LRN)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(LRN)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(LRN)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(LRN)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, n: int = ..., k: float = ..., alpha: float = ..., beta: float = ...) -> None\n"
};

void _init_py_LRN(py::module m) {
    using py_op = PyOp(LRN);
    auto& py_type = PyOpType(LRN);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.LRN";
    py_type.tp_basicsize = sizeof(PyOp(LRN));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "LRN";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(LRN), &PyOp(LRN)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("LRN", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(LRN::typeinfo(), &py_type).second);
}

PyOpDefBegin(LSQ) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(LSQ)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"qmin", serialization<decltype(opdef.qmin)>::dump(opdef.qmin)},
            {"qmax", serialization<decltype(opdef.qmax)>::dump(opdef.qmax)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(LSQ)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("qmin");
        if (iter != state.end()) {
            opdef.qmin = serialization<decltype(opdef.qmin)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("qmax");
        if (iter != state.end()) {
            opdef.qmax = serialization<decltype(opdef.qmax)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(LSQ)

int PyOp(LSQ)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"qmin", "qmax", "scope", NULL};
    PyObject *qmin = NULL, *qmax = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &qmin, &qmax, &scope))
    return -1;

    if (qmin) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSQ)*>(self)->inst().qmin =
                    py::cast<decltype(LSQ::qmin)>(py::handle(qmin));
        } CATCH_ALL(-1)
    }

    if (qmax) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSQ)*>(self)->inst().qmax =
                    py::cast<decltype(LSQ::qmax)>(py::handle(qmax));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(LSQ)::py_getsetters[] = {
    {const_cast<char*>("qmin"), py_get_generic(LSQ, qmin), py_set_generic(LSQ, qmin), const_cast<char*>("qmin"), NULL},
    {const_cast<char*>("qmax"), py_get_generic(LSQ, qmax), py_set_generic(LSQ, qmax), const_cast<char*>("qmax"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(LSQ)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(LSQ)::getstate, METH_NOARGS, "LSQ getstate"},
    {const_cast<char*>("__setstate__"), PyOp(LSQ)::setstate, METH_VARARGS, "LSQ setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(LSQ)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(LSQ)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(LSQ)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(LSQ)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, qmin: int = ..., qmax: int = ...) -> None\n"
};

void _init_py_LSQ(py::module m) {
    using py_op = PyOp(LSQ);
    auto& py_type = PyOpType(LSQ);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.LSQ";
    py_type.tp_basicsize = sizeof(PyOp(LSQ));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "LSQ";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(LSQ), &PyOp(LSQ)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("LSQ", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(LSQ::typeinfo(), &py_type).second);
}

void _init_py_LSTM_FwdMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<LSTM::FwdMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "FwdMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(LSTM) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(LSTM)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"num_layers", serialization<decltype(opdef.num_layers)>::dump(opdef.num_layers)},
            {"bidirectional", serialization<decltype(opdef.bidirectional)>::dump(opdef.bidirectional)},
            {"bias", serialization<decltype(opdef.bias)>::dump(opdef.bias)},
            {"hidden_size", serialization<decltype(opdef.hidden_size)>::dump(opdef.hidden_size)},
            {"proj_size", serialization<decltype(opdef.proj_size)>::dump(opdef.proj_size)},
            {"dropout", serialization<decltype(opdef.dropout)>::dump(opdef.dropout)},
            {"fwd_mode", serialization<decltype(opdef.fwd_mode)>::dump(opdef.fwd_mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(LSTM)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("num_layers");
        if (iter != state.end()) {
            opdef.num_layers = serialization<decltype(opdef.num_layers)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bidirectional");
        if (iter != state.end()) {
            opdef.bidirectional = serialization<decltype(opdef.bidirectional)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bias");
        if (iter != state.end()) {
            opdef.bias = serialization<decltype(opdef.bias)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("hidden_size");
        if (iter != state.end()) {
            opdef.hidden_size = serialization<decltype(opdef.hidden_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("proj_size");
        if (iter != state.end()) {
            opdef.proj_size = serialization<decltype(opdef.proj_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dropout");
        if (iter != state.end()) {
            opdef.dropout = serialization<decltype(opdef.dropout)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("fwd_mode");
        if (iter != state.end()) {
            opdef.fwd_mode = serialization<decltype(opdef.fwd_mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(LSTM)

int PyOp(LSTM)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"num_layers", "bidirectional", "bias", "hidden_size", "proj_size", "dropout", "fwd_mode", "scope", NULL};
    PyObject *num_layers = NULL, *bidirectional = NULL, *bias = NULL, *hidden_size = NULL, *proj_size = NULL, *dropout = NULL, *fwd_mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOO", const_cast<char**>(kwlist), &num_layers, &bidirectional, &bias, &hidden_size, &proj_size, &dropout, &fwd_mode, &scope))
    return -1;

    if (num_layers) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().num_layers =
                    py::cast<decltype(LSTM::num_layers)>(py::handle(num_layers));
        } CATCH_ALL(-1)
    }

    if (bidirectional) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().bidirectional =
                    py::cast<decltype(LSTM::bidirectional)>(py::handle(bidirectional));
        } CATCH_ALL(-1)
    }

    if (bias) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().bias =
                    py::cast<decltype(LSTM::bias)>(py::handle(bias));
        } CATCH_ALL(-1)
    }

    if (hidden_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().hidden_size =
                    py::cast<decltype(LSTM::hidden_size)>(py::handle(hidden_size));
        } CATCH_ALL(-1)
    }

    if (proj_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().proj_size =
                    py::cast<decltype(LSTM::proj_size)>(py::handle(proj_size));
        } CATCH_ALL(-1)
    }

    if (dropout) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().dropout =
                    py::cast<decltype(LSTM::dropout)>(py::handle(dropout));
        } CATCH_ALL(-1)
    }

    if (fwd_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LSTM)*>(self)->inst().fwd_mode =
                    py::cast<decltype(LSTM::fwd_mode)>(py::handle(fwd_mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(LSTM)::py_getsetters[] = {
    {const_cast<char*>("num_layers"), py_get_generic(LSTM, num_layers), py_set_generic(LSTM, num_layers), const_cast<char*>("num_layers"), NULL},
    {const_cast<char*>("bidirectional"), py_get_generic(LSTM, bidirectional), py_set_generic(LSTM, bidirectional), const_cast<char*>("bidirectional"), NULL},
    {const_cast<char*>("bias"), py_get_generic(LSTM, bias), py_set_generic(LSTM, bias), const_cast<char*>("bias"), NULL},
    {const_cast<char*>("hidden_size"), py_get_generic(LSTM, hidden_size), py_set_generic(LSTM, hidden_size), const_cast<char*>("hidden_size"), NULL},
    {const_cast<char*>("proj_size"), py_get_generic(LSTM, proj_size), py_set_generic(LSTM, proj_size), const_cast<char*>("proj_size"), NULL},
    {const_cast<char*>("dropout"), py_get_generic(LSTM, dropout), py_set_generic(LSTM, dropout), const_cast<char*>("dropout"), NULL},
    {const_cast<char*>("fwd_mode"), py_get_generic(LSTM, fwd_mode), py_set_generic(LSTM, fwd_mode), const_cast<char*>("fwd_mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(LSTM)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(LSTM)::getstate, METH_NOARGS, "LSTM getstate"},
    {const_cast<char*>("__setstate__"), PyOp(LSTM)::setstate, METH_VARARGS, "LSTM setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(LSTM)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(LSTM)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(LSTM)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(LSTM)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, num_layers: int = ..., bidirectional: bool = ..., bias: bool = ..., hidden_size: int = ..., proj_size: int = ..., dropout: float = ..., fwd_mode: Union[str, FwdMode] = ...) -> None\n"
};

void _init_py_LSTM(py::module m) {
    using py_op = PyOp(LSTM);
    auto& py_type = PyOpType(LSTM);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.LSTM";
    py_type.tp_basicsize = sizeof(PyOp(LSTM));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "LSTM";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(LSTM), &PyOp(LSTM)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_LSTM_FwdMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("LSTM", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(LSTM::typeinfo(), &py_type).second);
}

PyOpDefBegin(LSTMCell) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(LSTMCell)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(LSTMCell)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(LSTMCell)

int PyOp(LSTMCell)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(LSTMCell)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(LSTMCell)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(LSTMCell)::getstate, METH_NOARGS, "LSTMCell getstate"},
    {const_cast<char*>("__setstate__"), PyOp(LSTMCell)::setstate, METH_VARARGS, "LSTMCell setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(LSTMCell)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(LSTMCell)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(LSTMCell)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(LSTMCell)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_LSTMCell(py::module m) {
    using py_op = PyOp(LSTMCell);
    auto& py_type = PyOpType(LSTMCell);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.LSTMCell";
    py_type.tp_basicsize = sizeof(PyOp(LSTMCell));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "LSTMCell";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(LSTMCell), &PyOp(LSTMCell)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("LSTMCell", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(LSTMCell::typeinfo(), &py_type).second);
}

PyOpDefBegin(LayerNorm) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(LayerNorm)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"affine", serialization<decltype(opdef.affine)>::dump(opdef.affine)},
            {"eps", serialization<decltype(opdef.eps)>::dump(opdef.eps)},
            {"normalized_dim", serialization<decltype(opdef.normalized_dim)>::dump(opdef.normalized_dim)},
            {"normalized_size", serialization<decltype(opdef.normalized_size)>::dump(opdef.normalized_size)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(LayerNorm)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("affine");
        if (iter != state.end()) {
            opdef.affine = serialization<decltype(opdef.affine)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("eps");
        if (iter != state.end()) {
            opdef.eps = serialization<decltype(opdef.eps)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("normalized_dim");
        if (iter != state.end()) {
            opdef.normalized_dim = serialization<decltype(opdef.normalized_dim)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("normalized_size");
        if (iter != state.end()) {
            opdef.normalized_size = serialization<decltype(opdef.normalized_size)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(LayerNorm)

int PyOp(LayerNorm)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"affine", "eps", "normalized_dim", "normalized_size", "scope", NULL};
    PyObject *affine = NULL, *eps = NULL, *normalized_dim = NULL, *normalized_size = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &affine, &eps, &normalized_dim, &normalized_size, &scope))
    return -1;

    if (affine) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LayerNorm)*>(self)->inst().affine =
                    py::cast<decltype(LayerNorm::affine)>(py::handle(affine));
        } CATCH_ALL(-1)
    }

    if (eps) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LayerNorm)*>(self)->inst().eps =
                    py::cast<decltype(LayerNorm::eps)>(py::handle(eps));
        } CATCH_ALL(-1)
    }

    if (normalized_dim) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LayerNorm)*>(self)->inst().normalized_dim =
                    py::cast<decltype(LayerNorm::normalized_dim)>(py::handle(normalized_dim));
        } CATCH_ALL(-1)
    }

    if (normalized_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(LayerNorm)*>(self)->inst().normalized_size =
                    py::cast<decltype(LayerNorm::normalized_size)>(py::handle(normalized_size));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(LayerNorm)::py_getsetters[] = {
    {const_cast<char*>("affine"), py_get_generic(LayerNorm, affine), py_set_generic(LayerNorm, affine), const_cast<char*>("affine"), NULL},
    {const_cast<char*>("eps"), py_get_generic(LayerNorm, eps), py_set_generic(LayerNorm, eps), const_cast<char*>("eps"), NULL},
    {const_cast<char*>("normalized_dim"), py_get_generic(LayerNorm, normalized_dim), py_set_generic(LayerNorm, normalized_dim), const_cast<char*>("normalized_dim"), NULL},
    {const_cast<char*>("normalized_size"), py_get_generic(LayerNorm, normalized_size), py_set_generic(LayerNorm, normalized_size), const_cast<char*>("normalized_size"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(LayerNorm)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(LayerNorm)::getstate, METH_NOARGS, "LayerNorm getstate"},
    {const_cast<char*>("__setstate__"), PyOp(LayerNorm)::setstate, METH_VARARGS, "LayerNorm setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(LayerNorm)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(LayerNorm)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(LayerNorm)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(LayerNorm)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, affine: bool = ..., eps: float = ..., normalized_dim: int = ..., normalized_size: int = ...) -> None\n"
};

void _init_py_LayerNorm(py::module m) {
    using py_op = PyOp(LayerNorm);
    auto& py_type = PyOpType(LayerNorm);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.LayerNorm";
    py_type.tp_basicsize = sizeof(PyOp(LayerNorm));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "LayerNorm";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(LayerNorm), &PyOp(LayerNorm)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("LayerNorm", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(LayerNorm::typeinfo(), &py_type).second);
}

PyOpDefBegin(Linspace) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Linspace)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"endpoint", serialization<decltype(opdef.endpoint)>::dump(opdef.endpoint)},
            {"comp_node", serialization<decltype(opdef.comp_node)>::dump(opdef.comp_node)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Linspace)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("endpoint");
        if (iter != state.end()) {
            opdef.endpoint = serialization<decltype(opdef.endpoint)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("comp_node");
        if (iter != state.end()) {
            opdef.comp_node = serialization<decltype(opdef.comp_node)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Linspace)

int PyOp(Linspace)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"endpoint", "comp_node", "scope", NULL};
    PyObject *endpoint = NULL, *comp_node = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &endpoint, &comp_node, &scope))
    return -1;

    if (endpoint) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Linspace)*>(self)->inst().endpoint =
                    py::cast<decltype(Linspace::endpoint)>(py::handle(endpoint));
        } CATCH_ALL(-1)
    }

    if (comp_node) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Linspace)*>(self)->inst().comp_node =
                    py::cast<decltype(Linspace::comp_node)>(py::handle(comp_node));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Linspace)::py_getsetters[] = {
    {const_cast<char*>("endpoint"), py_get_generic(Linspace, endpoint), py_set_generic(Linspace, endpoint), const_cast<char*>("endpoint"), NULL},
    {const_cast<char*>("comp_node"), py_get_generic(Linspace, comp_node), py_set_generic(Linspace, comp_node), const_cast<char*>("comp_node"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Linspace)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Linspace)::getstate, METH_NOARGS, "Linspace getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Linspace)::setstate, METH_VARARGS, "Linspace setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Linspace)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Linspace)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Linspace)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Linspace)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, endpoint: bool = ..., comp_node: str = ...) -> None\n"
};

void _init_py_Linspace(py::module m) {
    using py_op = PyOp(Linspace);
    auto& py_type = PyOpType(Linspace);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Linspace";
    py_type.tp_basicsize = sizeof(PyOp(Linspace));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Linspace";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Linspace), &PyOp(Linspace)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Linspace", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Linspace::typeinfo(), &py_type).second);
}

PyOpDefBegin(MagicMindRuntime) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(MagicMindRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"buf", serialization<decltype(opdef.buf)>::dump(opdef.buf)},
            {"buf_size", serialization<decltype(opdef.buf_size)>::dump(opdef.buf_size)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(MagicMindRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("buf");
        if (iter != state.end()) {
            opdef.buf = serialization<decltype(opdef.buf)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("buf_size");
        if (iter != state.end()) {
            opdef.buf_size = serialization<decltype(opdef.buf_size)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(MagicMindRuntime)

int PyOp(MagicMindRuntime)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"buf", "buf_size", "scope", NULL};
    PyObject *buf = NULL, *buf_size = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &buf, &buf_size, &scope))
    return -1;

    if (buf) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MagicMindRuntime)*>(self)->inst().buf =
                    py::cast<decltype(MagicMindRuntime::buf)>(py::handle(buf));
        } CATCH_ALL(-1)
    }

    if (buf_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MagicMindRuntime)*>(self)->inst().buf_size =
                    py::cast<decltype(MagicMindRuntime::buf_size)>(py::handle(buf_size));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(MagicMindRuntime)::py_getsetters[] = {
    {const_cast<char*>("buf"), py_get_generic(MagicMindRuntime, buf), py_set_generic(MagicMindRuntime, buf), const_cast<char*>("buf"), NULL},
    {const_cast<char*>("buf_size"), py_get_generic(MagicMindRuntime, buf_size), py_set_generic(MagicMindRuntime, buf_size), const_cast<char*>("buf_size"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(MagicMindRuntime)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(MagicMindRuntime)::getstate, METH_NOARGS, "MagicMindRuntime getstate"},
    {const_cast<char*>("__setstate__"), PyOp(MagicMindRuntime)::setstate, METH_VARARGS, "MagicMindRuntime setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(MagicMindRuntime)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(MagicMindRuntime)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(MagicMindRuntime)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(MagicMindRuntime)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, buf: str = ..., buf_size: int = ...) -> None\n"
};

void _init_py_MagicMindRuntime(py::module m) {
    using py_op = PyOp(MagicMindRuntime);
    auto& py_type = PyOpType(MagicMindRuntime);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.MagicMindRuntime";
    py_type.tp_basicsize = sizeof(PyOp(MagicMindRuntime));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "MagicMindRuntime";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(MagicMindRuntime), &PyOp(MagicMindRuntime)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("MagicMindRuntime", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(MagicMindRuntime::typeinfo(), &py_type).second);
}

PyOpDefBegin(MatrixInverse) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(MatrixInverse)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(MatrixInverse)*>(self)->inst();
        static_cast<void>(opdef);
        
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(MatrixInverse)

int PyOp(MatrixInverse)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    
    return 0;
}

PyGetSetDef PyOp(MatrixInverse)::py_getsetters[] = {
    
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(MatrixInverse)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(MatrixInverse)::getstate, METH_NOARGS, "MatrixInverse getstate"},
    {const_cast<char*>("__setstate__"), PyOp(MatrixInverse)::setstate, METH_VARARGS, "MatrixInverse setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(MatrixInverse)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(MatrixInverse)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(MatrixInverse)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(MatrixInverse)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self) -> None\n"
};

void _init_py_MatrixInverse(py::module m) {
    using py_op = PyOp(MatrixInverse);
    auto& py_type = PyOpType(MatrixInverse);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.MatrixInverse";
    py_type.tp_basicsize = sizeof(PyOp(MatrixInverse));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "MatrixInverse";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(MatrixInverse), &PyOp(MatrixInverse)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("MatrixInverse", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(MatrixInverse::typeinfo(), &py_type).second);
}

void _init_py_MatrixMul_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<MatrixMul::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_MatrixMul_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<MatrixMul::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_MatrixMul_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<MatrixMul::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(MatrixMul) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(MatrixMul)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"transposeA", serialization<decltype(opdef.transposeA)>::dump(opdef.transposeA)},
            {"transposeB", serialization<decltype(opdef.transposeB)>::dump(opdef.transposeB)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)},
            {"dimA", serialization<decltype(opdef.dimA)>::dump(opdef.dimA)},
            {"dimB", serialization<decltype(opdef.dimB)>::dump(opdef.dimB)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(MatrixMul)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("transposeA");
        if (iter != state.end()) {
            opdef.transposeA = serialization<decltype(opdef.transposeA)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("transposeB");
        if (iter != state.end()) {
            opdef.transposeB = serialization<decltype(opdef.transposeB)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dimA");
        if (iter != state.end()) {
            opdef.dimA = serialization<decltype(opdef.dimA)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dimB");
        if (iter != state.end()) {
            opdef.dimB = serialization<decltype(opdef.dimB)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(MatrixMul)

int PyOp(MatrixMul)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"transposeA", "transposeB", "compute_mode", "format", "strategy", "workspace_limit", "dimA", "dimB", "scope", NULL};
    PyObject *transposeA = NULL, *transposeB = NULL, *compute_mode = NULL, *format = NULL, *strategy = NULL, *workspace_limit = NULL, *dimA = NULL, *dimB = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", const_cast<char**>(kwlist), &transposeA, &transposeB, &compute_mode, &format, &strategy, &workspace_limit, &dimA, &dimB, &scope))
    return -1;

    if (transposeA) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().transposeA =
                    py::cast<decltype(MatrixMul::transposeA)>(py::handle(transposeA));
        } CATCH_ALL(-1)
    }

    if (transposeB) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().transposeB =
                    py::cast<decltype(MatrixMul::transposeB)>(py::handle(transposeB));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().compute_mode =
                    py::cast<decltype(MatrixMul::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().format =
                    py::cast<decltype(MatrixMul::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().strategy =
                    py::cast<decltype(MatrixMul::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().workspace_limit =
                    py::cast<decltype(MatrixMul::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (dimA) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().dimA =
                    py::cast<decltype(MatrixMul::dimA)>(py::handle(dimA));
        } CATCH_ALL(-1)
    }

    if (dimB) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MatrixMul)*>(self)->inst().dimB =
                    py::cast<decltype(MatrixMul::dimB)>(py::handle(dimB));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(MatrixMul)::py_getsetters[] = {
    {const_cast<char*>("transposeA"), py_get_generic(MatrixMul, transposeA), py_set_generic(MatrixMul, transposeA), const_cast<char*>("transposeA"), NULL},
    {const_cast<char*>("transposeB"), py_get_generic(MatrixMul, transposeB), py_set_generic(MatrixMul, transposeB), const_cast<char*>("transposeB"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(MatrixMul, compute_mode), py_set_generic(MatrixMul, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {const_cast<char*>("format"), py_get_generic(MatrixMul, format), py_set_generic(MatrixMul, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(MatrixMul, strategy), py_set_generic(MatrixMul, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(MatrixMul, workspace_limit), py_set_generic(MatrixMul, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {const_cast<char*>("dimA"), py_get_generic(MatrixMul, dimA), py_set_generic(MatrixMul, dimA), const_cast<char*>("dimA"), NULL},
    {const_cast<char*>("dimB"), py_get_generic(MatrixMul, dimB), py_set_generic(MatrixMul, dimB), const_cast<char*>("dimB"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(MatrixMul)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(MatrixMul)::getstate, METH_NOARGS, "MatrixMul getstate"},
    {const_cast<char*>("__setstate__"), PyOp(MatrixMul)::setstate, METH_VARARGS, "MatrixMul setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(MatrixMul)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(MatrixMul)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(MatrixMul)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(MatrixMul)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, transposeA: bool = ..., transposeB: bool = ..., compute_mode: Union[str, ComputeMode] = ..., format: Union[str, Format] = ..., strategy: Union[str, Strategy] = ..., dimA: int = ..., dimB: int = ...) -> None\n"
};

void _init_py_MatrixMul(py::module m) {
    using py_op = PyOp(MatrixMul);
    auto& py_type = PyOpType(MatrixMul);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.MatrixMul";
    py_type.tp_basicsize = sizeof(PyOp(MatrixMul));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "MatrixMul";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(MatrixMul), &PyOp(MatrixMul)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_MatrixMul_ComputeMode(py_type);
    _init_py_MatrixMul_Format(py_type);
    _init_py_MatrixMul_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("MatrixMul", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(MatrixMul::typeinfo(), &py_type).second);
}

PyOpDefBegin(MeshGrid) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(MeshGrid)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"indexing", serialization<decltype(opdef.indexing)>::dump(opdef.indexing)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(MeshGrid)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("indexing");
        if (iter != state.end()) {
            opdef.indexing = serialization<decltype(opdef.indexing)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(MeshGrid)

int PyOp(MeshGrid)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"indexing", "scope", NULL};
    PyObject *indexing = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &indexing, &scope))
    return -1;

    if (indexing) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MeshGrid)*>(self)->inst().indexing =
                    py::cast<decltype(MeshGrid::indexing)>(py::handle(indexing));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(MeshGrid)::py_getsetters[] = {
    {const_cast<char*>("indexing"), py_get_generic(MeshGrid, indexing), py_set_generic(MeshGrid, indexing), const_cast<char*>("indexing"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(MeshGrid)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(MeshGrid)::getstate, METH_NOARGS, "MeshGrid getstate"},
    {const_cast<char*>("__setstate__"), PyOp(MeshGrid)::setstate, METH_VARARGS, "MeshGrid setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(MeshGrid)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(MeshGrid)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(MeshGrid)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(MeshGrid)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, indexing: str = ...) -> None\n"
};

void _init_py_MeshGrid(py::module m) {
    using py_op = PyOp(MeshGrid);
    auto& py_type = PyOpType(MeshGrid);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.MeshGrid";
    py_type.tp_basicsize = sizeof(PyOp(MeshGrid));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "MeshGrid";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(MeshGrid), &PyOp(MeshGrid)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("MeshGrid", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(MeshGrid::typeinfo(), &py_type).second);
}

PyOpDefBegin(MeshIndexing) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(MeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(MeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(MeshIndexing)

int PyOp(MeshIndexing)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(MeshIndexing)*>(self)->inst().items =
                    py::cast<decltype(MeshIndexing::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(MeshIndexing)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(MeshIndexing, items), py_set_generic(MeshIndexing, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(MeshIndexing)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(MeshIndexing)::getstate, METH_NOARGS, "MeshIndexing getstate"},
    {const_cast<char*>("__setstate__"), PyOp(MeshIndexing)::setstate, METH_VARARGS, "MeshIndexing setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(MeshIndexing)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(MeshIndexing)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(MeshIndexing)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(MeshIndexing)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_MeshIndexing(py::module m) {
    using py_op = PyOp(MeshIndexing);
    auto& py_type = PyOpType(MeshIndexing);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.MeshIndexing";
    py_type.tp_basicsize = sizeof(PyOp(MeshIndexing));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "MeshIndexing";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(MeshIndexing), &PyOp(MeshIndexing)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("MeshIndexing", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(MeshIndexing::typeinfo(), &py_type).second);
}

PyOpDefBegin(NMSKeep) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(NMSKeep)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"iou_thresh", serialization<decltype(opdef.iou_thresh)>::dump(opdef.iou_thresh)},
            {"max_output", serialization<decltype(opdef.max_output)>::dump(opdef.max_output)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(NMSKeep)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("iou_thresh");
        if (iter != state.end()) {
            opdef.iou_thresh = serialization<decltype(opdef.iou_thresh)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("max_output");
        if (iter != state.end()) {
            opdef.max_output = serialization<decltype(opdef.max_output)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(NMSKeep)

int PyOp(NMSKeep)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"iou_thresh", "max_output", "scope", NULL};
    PyObject *iou_thresh = NULL, *max_output = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &iou_thresh, &max_output, &scope))
    return -1;

    if (iou_thresh) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(NMSKeep)*>(self)->inst().iou_thresh =
                    py::cast<decltype(NMSKeep::iou_thresh)>(py::handle(iou_thresh));
        } CATCH_ALL(-1)
    }

    if (max_output) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(NMSKeep)*>(self)->inst().max_output =
                    py::cast<decltype(NMSKeep::max_output)>(py::handle(max_output));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(NMSKeep)::py_getsetters[] = {
    {const_cast<char*>("iou_thresh"), py_get_generic(NMSKeep, iou_thresh), py_set_generic(NMSKeep, iou_thresh), const_cast<char*>("iou_thresh"), NULL},
    {const_cast<char*>("max_output"), py_get_generic(NMSKeep, max_output), py_set_generic(NMSKeep, max_output), const_cast<char*>("max_output"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(NMSKeep)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(NMSKeep)::getstate, METH_NOARGS, "NMSKeep getstate"},
    {const_cast<char*>("__setstate__"), PyOp(NMSKeep)::setstate, METH_VARARGS, "NMSKeep setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(NMSKeep)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(NMSKeep)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(NMSKeep)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(NMSKeep)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, iou_thresh: float = ..., max_output: int = ...) -> None\n"
};

void _init_py_NMSKeep(py::module m) {
    using py_op = PyOp(NMSKeep);
    auto& py_type = PyOpType(NMSKeep);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.NMSKeep";
    py_type.tp_basicsize = sizeof(PyOp(NMSKeep));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "NMSKeep";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(NMSKeep), &PyOp(NMSKeep)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("NMSKeep", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(NMSKeep::typeinfo(), &py_type).second);
}

PyOpDefBegin(NvOf) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(NvOf)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"precision", serialization<decltype(opdef.precision)>::dump(opdef.precision)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(NvOf)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("precision");
        if (iter != state.end()) {
            opdef.precision = serialization<decltype(opdef.precision)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(NvOf)

int PyOp(NvOf)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"precision", "scope", NULL};
    PyObject *precision = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &precision, &scope))
    return -1;

    if (precision) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(NvOf)*>(self)->inst().precision =
                    py::cast<decltype(NvOf::precision)>(py::handle(precision));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(NvOf)::py_getsetters[] = {
    {const_cast<char*>("precision"), py_get_generic(NvOf, precision), py_set_generic(NvOf, precision), const_cast<char*>("precision"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(NvOf)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(NvOf)::getstate, METH_NOARGS, "NvOf getstate"},
    {const_cast<char*>("__setstate__"), PyOp(NvOf)::setstate, METH_VARARGS, "NvOf setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(NvOf)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(NvOf)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(NvOf)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(NvOf)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, precision: int = ...) -> None\n"
};

void _init_py_NvOf(py::module m) {
    using py_op = PyOp(NvOf);
    auto& py_type = PyOpType(NvOf);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.NvOf";
    py_type.tp_basicsize = sizeof(PyOp(NvOf));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "NvOf";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(NvOf), &PyOp(NvOf)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("NvOf", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(NvOf::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Padding::PaddingMode> {
    static constexpr const char *name = "Padding.PaddingMode";
    static constexpr std::underlying_type_t<Padding::PaddingMode> max = 3 - 1;
};
template<> PyTypeObject* EnumWrapper<Padding::PaddingMode>::type = nullptr;

template<> const char*
EnumWrapper<Padding::PaddingMode>::members[] = {"REPLICATE", "REFLECT", "CONSTANT"};

template<> std::unordered_map<std::string, Padding::PaddingMode>
EnumWrapper<Padding::PaddingMode>::mem2value = {{normalize_enum("REPLICATE"), Padding::PaddingMode::REPLICATE}, {normalize_enum("REFLECT"), Padding::PaddingMode::REFLECT}, {normalize_enum("CONSTANT"), Padding::PaddingMode::CONSTANT}};
template<> PyObject* EnumWrapper<Padding::PaddingMode>::pyobj_insts[3] = {nullptr};

void _init_py_Padding_PaddingMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Padding::PaddingMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Padding::PaddingMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Padding::PaddingMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Padding::PaddingMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Padding.PaddingMode",
        // basicsize
        sizeof(EnumWrapper<Padding::PaddingMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("PaddingMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Padding.PaddingMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Padding::PaddingMode>*>(inst)->value = Padding::PaddingMode::REPLICATE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REPLICATE", inst) >= 0);
    EnumWrapper<Padding::PaddingMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Padding::PaddingMode>*>(inst)->value = Padding::PaddingMode::REFLECT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REFLECT", inst) >= 0);
    EnumWrapper<Padding::PaddingMode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Padding::PaddingMode>*>(inst)->value = Padding::PaddingMode::CONSTANT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CONSTANT", inst) >= 0);
    EnumWrapper<Padding::PaddingMode>::pyobj_insts[2] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "PaddingMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Padding) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Padding)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"front_offset_dim0", serialization<decltype(opdef.front_offset_dim0)>::dump(opdef.front_offset_dim0)},
            {"front_offset_dim1", serialization<decltype(opdef.front_offset_dim1)>::dump(opdef.front_offset_dim1)},
            {"front_offset_dim2", serialization<decltype(opdef.front_offset_dim2)>::dump(opdef.front_offset_dim2)},
            {"front_offset_dim3", serialization<decltype(opdef.front_offset_dim3)>::dump(opdef.front_offset_dim3)},
            {"front_offset_dim4", serialization<decltype(opdef.front_offset_dim4)>::dump(opdef.front_offset_dim4)},
            {"front_offset_dim5", serialization<decltype(opdef.front_offset_dim5)>::dump(opdef.front_offset_dim5)},
            {"front_offset_dim6", serialization<decltype(opdef.front_offset_dim6)>::dump(opdef.front_offset_dim6)},
            {"back_offset_dim0", serialization<decltype(opdef.back_offset_dim0)>::dump(opdef.back_offset_dim0)},
            {"back_offset_dim1", serialization<decltype(opdef.back_offset_dim1)>::dump(opdef.back_offset_dim1)},
            {"back_offset_dim2", serialization<decltype(opdef.back_offset_dim2)>::dump(opdef.back_offset_dim2)},
            {"back_offset_dim3", serialization<decltype(opdef.back_offset_dim3)>::dump(opdef.back_offset_dim3)},
            {"back_offset_dim4", serialization<decltype(opdef.back_offset_dim4)>::dump(opdef.back_offset_dim4)},
            {"back_offset_dim5", serialization<decltype(opdef.back_offset_dim5)>::dump(opdef.back_offset_dim5)},
            {"back_offset_dim6", serialization<decltype(opdef.back_offset_dim6)>::dump(opdef.back_offset_dim6)},
            {"padding_val", serialization<decltype(opdef.padding_val)>::dump(opdef.padding_val)},
            {"padding_mode", serialization<decltype(opdef.padding_mode)>::dump(opdef.padding_mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Padding)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("front_offset_dim0");
        if (iter != state.end()) {
            opdef.front_offset_dim0 = serialization<decltype(opdef.front_offset_dim0)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("front_offset_dim1");
        if (iter != state.end()) {
            opdef.front_offset_dim1 = serialization<decltype(opdef.front_offset_dim1)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("front_offset_dim2");
        if (iter != state.end()) {
            opdef.front_offset_dim2 = serialization<decltype(opdef.front_offset_dim2)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("front_offset_dim3");
        if (iter != state.end()) {
            opdef.front_offset_dim3 = serialization<decltype(opdef.front_offset_dim3)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("front_offset_dim4");
        if (iter != state.end()) {
            opdef.front_offset_dim4 = serialization<decltype(opdef.front_offset_dim4)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("front_offset_dim5");
        if (iter != state.end()) {
            opdef.front_offset_dim5 = serialization<decltype(opdef.front_offset_dim5)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("front_offset_dim6");
        if (iter != state.end()) {
            opdef.front_offset_dim6 = serialization<decltype(opdef.front_offset_dim6)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim0");
        if (iter != state.end()) {
            opdef.back_offset_dim0 = serialization<decltype(opdef.back_offset_dim0)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim1");
        if (iter != state.end()) {
            opdef.back_offset_dim1 = serialization<decltype(opdef.back_offset_dim1)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim2");
        if (iter != state.end()) {
            opdef.back_offset_dim2 = serialization<decltype(opdef.back_offset_dim2)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim3");
        if (iter != state.end()) {
            opdef.back_offset_dim3 = serialization<decltype(opdef.back_offset_dim3)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim4");
        if (iter != state.end()) {
            opdef.back_offset_dim4 = serialization<decltype(opdef.back_offset_dim4)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim5");
        if (iter != state.end()) {
            opdef.back_offset_dim5 = serialization<decltype(opdef.back_offset_dim5)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("back_offset_dim6");
        if (iter != state.end()) {
            opdef.back_offset_dim6 = serialization<decltype(opdef.back_offset_dim6)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("padding_val");
        if (iter != state.end()) {
            opdef.padding_val = serialization<decltype(opdef.padding_val)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("padding_mode");
        if (iter != state.end()) {
            opdef.padding_mode = serialization<decltype(opdef.padding_mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Padding)

int PyOp(Padding)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"front_offset_dim0", "front_offset_dim1", "front_offset_dim2", "front_offset_dim3", "front_offset_dim4", "front_offset_dim5", "front_offset_dim6", "back_offset_dim0", "back_offset_dim1", "back_offset_dim2", "back_offset_dim3", "back_offset_dim4", "back_offset_dim5", "back_offset_dim6", "padding_val", "padding_mode", "scope", NULL};
    PyObject *front_offset_dim0 = NULL, *front_offset_dim1 = NULL, *front_offset_dim2 = NULL, *front_offset_dim3 = NULL, *front_offset_dim4 = NULL, *front_offset_dim5 = NULL, *front_offset_dim6 = NULL, *back_offset_dim0 = NULL, *back_offset_dim1 = NULL, *back_offset_dim2 = NULL, *back_offset_dim3 = NULL, *back_offset_dim4 = NULL, *back_offset_dim5 = NULL, *back_offset_dim6 = NULL, *padding_val = NULL, *padding_mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOOOOOOOO", const_cast<char**>(kwlist), &front_offset_dim0, &front_offset_dim1, &front_offset_dim2, &front_offset_dim3, &front_offset_dim4, &front_offset_dim5, &front_offset_dim6, &back_offset_dim0, &back_offset_dim1, &back_offset_dim2, &back_offset_dim3, &back_offset_dim4, &back_offset_dim5, &back_offset_dim6, &padding_val, &padding_mode, &scope))
    return -1;

    if (front_offset_dim0) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim0 =
                    py::cast<decltype(Padding::front_offset_dim0)>(py::handle(front_offset_dim0));
        } CATCH_ALL(-1)
    }

    if (front_offset_dim1) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim1 =
                    py::cast<decltype(Padding::front_offset_dim1)>(py::handle(front_offset_dim1));
        } CATCH_ALL(-1)
    }

    if (front_offset_dim2) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim2 =
                    py::cast<decltype(Padding::front_offset_dim2)>(py::handle(front_offset_dim2));
        } CATCH_ALL(-1)
    }

    if (front_offset_dim3) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim3 =
                    py::cast<decltype(Padding::front_offset_dim3)>(py::handle(front_offset_dim3));
        } CATCH_ALL(-1)
    }

    if (front_offset_dim4) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim4 =
                    py::cast<decltype(Padding::front_offset_dim4)>(py::handle(front_offset_dim4));
        } CATCH_ALL(-1)
    }

    if (front_offset_dim5) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim5 =
                    py::cast<decltype(Padding::front_offset_dim5)>(py::handle(front_offset_dim5));
        } CATCH_ALL(-1)
    }

    if (front_offset_dim6) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().front_offset_dim6 =
                    py::cast<decltype(Padding::front_offset_dim6)>(py::handle(front_offset_dim6));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim0) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim0 =
                    py::cast<decltype(Padding::back_offset_dim0)>(py::handle(back_offset_dim0));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim1) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim1 =
                    py::cast<decltype(Padding::back_offset_dim1)>(py::handle(back_offset_dim1));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim2) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim2 =
                    py::cast<decltype(Padding::back_offset_dim2)>(py::handle(back_offset_dim2));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim3) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim3 =
                    py::cast<decltype(Padding::back_offset_dim3)>(py::handle(back_offset_dim3));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim4) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim4 =
                    py::cast<decltype(Padding::back_offset_dim4)>(py::handle(back_offset_dim4));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim5) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim5 =
                    py::cast<decltype(Padding::back_offset_dim5)>(py::handle(back_offset_dim5));
        } CATCH_ALL(-1)
    }

    if (back_offset_dim6) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().back_offset_dim6 =
                    py::cast<decltype(Padding::back_offset_dim6)>(py::handle(back_offset_dim6));
        } CATCH_ALL(-1)
    }

    if (padding_val) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().padding_val =
                    py::cast<decltype(Padding::padding_val)>(py::handle(padding_val));
        } CATCH_ALL(-1)
    }

    if (padding_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Padding)*>(self)->inst().padding_mode =
                    py::cast<decltype(Padding::padding_mode)>(py::handle(padding_mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Padding)::py_getsetters[] = {
    {const_cast<char*>("front_offset_dim0"), py_get_generic(Padding, front_offset_dim0), py_set_generic(Padding, front_offset_dim0), const_cast<char*>("front_offset_dim0"), NULL},
    {const_cast<char*>("front_offset_dim1"), py_get_generic(Padding, front_offset_dim1), py_set_generic(Padding, front_offset_dim1), const_cast<char*>("front_offset_dim1"), NULL},
    {const_cast<char*>("front_offset_dim2"), py_get_generic(Padding, front_offset_dim2), py_set_generic(Padding, front_offset_dim2), const_cast<char*>("front_offset_dim2"), NULL},
    {const_cast<char*>("front_offset_dim3"), py_get_generic(Padding, front_offset_dim3), py_set_generic(Padding, front_offset_dim3), const_cast<char*>("front_offset_dim3"), NULL},
    {const_cast<char*>("front_offset_dim4"), py_get_generic(Padding, front_offset_dim4), py_set_generic(Padding, front_offset_dim4), const_cast<char*>("front_offset_dim4"), NULL},
    {const_cast<char*>("front_offset_dim5"), py_get_generic(Padding, front_offset_dim5), py_set_generic(Padding, front_offset_dim5), const_cast<char*>("front_offset_dim5"), NULL},
    {const_cast<char*>("front_offset_dim6"), py_get_generic(Padding, front_offset_dim6), py_set_generic(Padding, front_offset_dim6), const_cast<char*>("front_offset_dim6"), NULL},
    {const_cast<char*>("back_offset_dim0"), py_get_generic(Padding, back_offset_dim0), py_set_generic(Padding, back_offset_dim0), const_cast<char*>("back_offset_dim0"), NULL},
    {const_cast<char*>("back_offset_dim1"), py_get_generic(Padding, back_offset_dim1), py_set_generic(Padding, back_offset_dim1), const_cast<char*>("back_offset_dim1"), NULL},
    {const_cast<char*>("back_offset_dim2"), py_get_generic(Padding, back_offset_dim2), py_set_generic(Padding, back_offset_dim2), const_cast<char*>("back_offset_dim2"), NULL},
    {const_cast<char*>("back_offset_dim3"), py_get_generic(Padding, back_offset_dim3), py_set_generic(Padding, back_offset_dim3), const_cast<char*>("back_offset_dim3"), NULL},
    {const_cast<char*>("back_offset_dim4"), py_get_generic(Padding, back_offset_dim4), py_set_generic(Padding, back_offset_dim4), const_cast<char*>("back_offset_dim4"), NULL},
    {const_cast<char*>("back_offset_dim5"), py_get_generic(Padding, back_offset_dim5), py_set_generic(Padding, back_offset_dim5), const_cast<char*>("back_offset_dim5"), NULL},
    {const_cast<char*>("back_offset_dim6"), py_get_generic(Padding, back_offset_dim6), py_set_generic(Padding, back_offset_dim6), const_cast<char*>("back_offset_dim6"), NULL},
    {const_cast<char*>("padding_val"), py_get_generic(Padding, padding_val), py_set_generic(Padding, padding_val), const_cast<char*>("padding_val"), NULL},
    {const_cast<char*>("padding_mode"), py_get_generic(Padding, padding_mode), py_set_generic(Padding, padding_mode), const_cast<char*>("padding_mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Padding)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Padding)::getstate, METH_NOARGS, "Padding getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Padding)::setstate, METH_VARARGS, "Padding setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Padding)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Padding)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Padding)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Padding)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, front_offset_dim0: int = ..., front_offset_dim1: int = ..., front_offset_dim2: int = ..., front_offset_dim3: int = ..., front_offset_dim4: int = ..., front_offset_dim5: int = ..., front_offset_dim6: int = ..., back_offset_dim0: int = ..., back_offset_dim1: int = ..., back_offset_dim2: int = ..., back_offset_dim3: int = ..., back_offset_dim4: int = ..., back_offset_dim5: int = ..., back_offset_dim6: int = ..., padding_val: float = ..., padding_mode: Union[str, PaddingMode] = ...) -> None\n"
};

void _init_py_Padding(py::module m) {
    using py_op = PyOp(Padding);
    auto& py_type = PyOpType(Padding);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Padding";
    py_type.tp_basicsize = sizeof(PyOp(Padding));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Padding";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Padding), &PyOp(Padding)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Padding_PaddingMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("Padding", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Padding::typeinfo(), &py_type).second);
}

PyOpDefBegin(ParamPackConcat) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ParamPackConcat)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"offsets", serialization<decltype(opdef.offsets)>::dump(opdef.offsets)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ParamPackConcat)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("offsets");
        if (iter != state.end()) {
            opdef.offsets = serialization<decltype(opdef.offsets)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ParamPackConcat)

int PyOp(ParamPackConcat)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"offsets", "scope", NULL};
    PyObject *offsets = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &offsets, &scope))
    return -1;

    if (offsets) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ParamPackConcat)*>(self)->inst().offsets =
                    py::cast<decltype(ParamPackConcat::offsets)>(py::handle(offsets));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ParamPackConcat)::py_getsetters[] = {
    {const_cast<char*>("offsets"), py_get_generic(ParamPackConcat, offsets), py_set_generic(ParamPackConcat, offsets), const_cast<char*>("offsets"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ParamPackConcat)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ParamPackConcat)::getstate, METH_NOARGS, "ParamPackConcat getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ParamPackConcat)::setstate, METH_VARARGS, "ParamPackConcat setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ParamPackConcat)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ParamPackConcat)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ParamPackConcat)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ParamPackConcat)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, offsets: list[int] = ...) -> None\n"
};

void _init_py_ParamPackConcat(py::module m) {
    using py_op = PyOp(ParamPackConcat);
    auto& py_type = PyOpType(ParamPackConcat);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ParamPackConcat";
    py_type.tp_basicsize = sizeof(PyOp(ParamPackConcat));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ParamPackConcat";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ParamPackConcat), &PyOp(ParamPackConcat)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("ParamPackConcat", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ParamPackConcat::typeinfo(), &py_type).second);
}

PyOpDefBegin(ParamPackSplit) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ParamPackSplit)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"offsets", serialization<decltype(opdef.offsets)>::dump(opdef.offsets)},
            {"shapes", serialization<decltype(opdef.shapes)>::dump(opdef.shapes)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ParamPackSplit)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("offsets");
        if (iter != state.end()) {
            opdef.offsets = serialization<decltype(opdef.offsets)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("shapes");
        if (iter != state.end()) {
            opdef.shapes = serialization<decltype(opdef.shapes)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ParamPackSplit)

int PyOp(ParamPackSplit)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"offsets", "shapes", "scope", NULL};
    PyObject *offsets = NULL, *shapes = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &offsets, &shapes, &scope))
    return -1;

    if (offsets) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ParamPackSplit)*>(self)->inst().offsets =
                    py::cast<decltype(ParamPackSplit::offsets)>(py::handle(offsets));
        } CATCH_ALL(-1)
    }

    if (shapes) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ParamPackSplit)*>(self)->inst().shapes =
                    py::cast<decltype(ParamPackSplit::shapes)>(py::handle(shapes));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ParamPackSplit)::py_getsetters[] = {
    {const_cast<char*>("offsets"), py_get_generic(ParamPackSplit, offsets), py_set_generic(ParamPackSplit, offsets), const_cast<char*>("offsets"), NULL},
    {const_cast<char*>("shapes"), py_get_generic(ParamPackSplit, shapes), py_set_generic(ParamPackSplit, shapes), const_cast<char*>("shapes"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ParamPackSplit)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ParamPackSplit)::getstate, METH_NOARGS, "ParamPackSplit getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ParamPackSplit)::setstate, METH_VARARGS, "ParamPackSplit setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ParamPackSplit)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ParamPackSplit)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ParamPackSplit)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ParamPackSplit)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, offsets: list[int] = ..., shapes: list[list[int]] = ...) -> None\n"
};

void _init_py_ParamPackSplit(py::module m) {
    using py_op = PyOp(ParamPackSplit);
    auto& py_type = PyOpType(ParamPackSplit);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ParamPackSplit";
    py_type.tp_basicsize = sizeof(PyOp(ParamPackSplit));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ParamPackSplit";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ParamPackSplit), &PyOp(ParamPackSplit)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("ParamPackSplit", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ParamPackSplit::typeinfo(), &py_type).second);
}

PyOpDefBegin(PermutationRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(PermutationRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(PermutationRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(PermutationRNG)

int PyOp(PermutationRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "dtype", "handle", "scope", NULL};
    PyObject *seed = NULL, *dtype = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", const_cast<char**>(kwlist), &seed, &dtype, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PermutationRNG)*>(self)->inst().seed =
                    py::cast<decltype(PermutationRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PermutationRNG)*>(self)->inst().dtype =
                    py::cast<decltype(PermutationRNG::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PermutationRNG)*>(self)->inst().handle =
                    py::cast<decltype(PermutationRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(PermutationRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(PermutationRNG, seed), py_set_generic(PermutationRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(PermutationRNG, dtype), py_set_generic(PermutationRNG, dtype), const_cast<char*>("dtype"), NULL},
    {const_cast<char*>("handle"), py_get_generic(PermutationRNG, handle), py_set_generic(PermutationRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(PermutationRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(PermutationRNG)::getstate, METH_NOARGS, "PermutationRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(PermutationRNG)::setstate, METH_VARARGS, "PermutationRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(PermutationRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(PermutationRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(PermutationRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(PermutationRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., dtype: str = ..., handle: int = ...) -> None\n"
};

void _init_py_PermutationRNG(py::module m) {
    using py_op = PyOp(PermutationRNG);
    auto& py_type = PyOpType(PermutationRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.PermutationRNG";
    py_type.tp_basicsize = sizeof(PyOp(PermutationRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "PermutationRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(PermutationRNG), &PyOp(PermutationRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("PermutationRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(PermutationRNG::typeinfo(), &py_type).second);
}

PyOpDefBegin(PixelShuffle) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(PixelShuffle)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"factor", serialization<decltype(opdef.factor)>::dump(opdef.factor)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(PixelShuffle)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("factor");
        if (iter != state.end()) {
            opdef.factor = serialization<decltype(opdef.factor)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(PixelShuffle)

int PyOp(PixelShuffle)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"factor", "scope", NULL};
    PyObject *factor = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &factor, &scope))
    return -1;

    if (factor) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PixelShuffle)*>(self)->inst().factor =
                    py::cast<decltype(PixelShuffle::factor)>(py::handle(factor));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(PixelShuffle)::py_getsetters[] = {
    {const_cast<char*>("factor"), py_get_generic(PixelShuffle, factor), py_set_generic(PixelShuffle, factor), const_cast<char*>("factor"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(PixelShuffle)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(PixelShuffle)::getstate, METH_NOARGS, "PixelShuffle getstate"},
    {const_cast<char*>("__setstate__"), PyOp(PixelShuffle)::setstate, METH_VARARGS, "PixelShuffle setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(PixelShuffle)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(PixelShuffle)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(PixelShuffle)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(PixelShuffle)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, factor: int = ...) -> None\n"
};

void _init_py_PixelShuffle(py::module m) {
    using py_op = PyOp(PixelShuffle);
    auto& py_type = PyOpType(PixelShuffle);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.PixelShuffle";
    py_type.tp_basicsize = sizeof(PyOp(PixelShuffle));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "PixelShuffle";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(PixelShuffle), &PyOp(PixelShuffle)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("PixelShuffle", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(PixelShuffle::typeinfo(), &py_type).second);
}

PyOpDefBegin(PixelShuffleBackward) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(PixelShuffleBackward)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"factor", serialization<decltype(opdef.factor)>::dump(opdef.factor)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(PixelShuffleBackward)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("factor");
        if (iter != state.end()) {
            opdef.factor = serialization<decltype(opdef.factor)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(PixelShuffleBackward)

int PyOp(PixelShuffleBackward)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"factor", "scope", NULL};
    PyObject *factor = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &factor, &scope))
    return -1;

    if (factor) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PixelShuffleBackward)*>(self)->inst().factor =
                    py::cast<decltype(PixelShuffleBackward::factor)>(py::handle(factor));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(PixelShuffleBackward)::py_getsetters[] = {
    {const_cast<char*>("factor"), py_get_generic(PixelShuffleBackward, factor), py_set_generic(PixelShuffleBackward, factor), const_cast<char*>("factor"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(PixelShuffleBackward)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(PixelShuffleBackward)::getstate, METH_NOARGS, "PixelShuffleBackward getstate"},
    {const_cast<char*>("__setstate__"), PyOp(PixelShuffleBackward)::setstate, METH_VARARGS, "PixelShuffleBackward setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(PixelShuffleBackward)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(PixelShuffleBackward)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(PixelShuffleBackward)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(PixelShuffleBackward)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, factor: int = ...) -> None\n"
};

void _init_py_PixelShuffleBackward(py::module m) {
    using py_op = PyOp(PixelShuffleBackward);
    auto& py_type = PyOpType(PixelShuffleBackward);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.PixelShuffleBackward";
    py_type.tp_basicsize = sizeof(PyOp(PixelShuffleBackward));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "PixelShuffleBackward";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(PixelShuffleBackward), &PyOp(PixelShuffleBackward)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("PixelShuffleBackward", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(PixelShuffleBackward::typeinfo(), &py_type).second);
}

PyOpDefBegin(PoissonRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(PoissonRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(PoissonRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(PoissonRNG)

int PyOp(PoissonRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "handle", "scope", NULL};
    PyObject *seed = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &seed, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PoissonRNG)*>(self)->inst().seed =
                    py::cast<decltype(PoissonRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(PoissonRNG)*>(self)->inst().handle =
                    py::cast<decltype(PoissonRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(PoissonRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(PoissonRNG, seed), py_set_generic(PoissonRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("handle"), py_get_generic(PoissonRNG, handle), py_set_generic(PoissonRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(PoissonRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(PoissonRNG)::getstate, METH_NOARGS, "PoissonRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(PoissonRNG)::setstate, METH_VARARGS, "PoissonRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(PoissonRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(PoissonRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(PoissonRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(PoissonRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., handle: int = ...) -> None\n"
};

void _init_py_PoissonRNG(py::module m) {
    using py_op = PyOp(PoissonRNG);
    auto& py_type = PyOpType(PoissonRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.PoissonRNG";
    py_type.tp_basicsize = sizeof(PyOp(PoissonRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "PoissonRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(PoissonRNG), &PyOp(PoissonRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("PoissonRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(PoissonRNG::typeinfo(), &py_type).second);
}

void _init_py_Pooling_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Pooling::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Pooling_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Pooling::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Pooling_Strategy(PyTypeObject& py_type) {
    auto& e_type = BitCombinedEnumWrapper<Pooling::Strategy>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Strategy", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Pooling) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Pooling)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"window_h", serialization<decltype(opdef.window_h)>::dump(opdef.window_h)},
            {"window_w", serialization<decltype(opdef.window_w)>::dump(opdef.window_w)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"strategy", serialization<decltype(opdef.strategy)>::dump(opdef.strategy)},
            {"workspace_limit", serialization<decltype(opdef.workspace_limit)>::dump(opdef.workspace_limit)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Pooling)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("window_h");
        if (iter != state.end()) {
            opdef.window_h = serialization<decltype(opdef.window_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("window_w");
        if (iter != state.end()) {
            opdef.window_w = serialization<decltype(opdef.window_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("strategy");
        if (iter != state.end()) {
            opdef.strategy = serialization<decltype(opdef.strategy)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("workspace_limit");
        if (iter != state.end()) {
            opdef.workspace_limit = serialization<decltype(opdef.workspace_limit)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Pooling)

int PyOp(Pooling)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "window_h", "window_w", "format", "strategy", "workspace_limit", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *window_h = NULL, *window_w = NULL, *format = NULL, *strategy = NULL, *workspace_limit = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &window_h, &window_w, &format, &strategy, &workspace_limit, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().mode =
                    py::cast<decltype(Pooling::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().pad_h =
                    py::cast<decltype(Pooling::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().pad_w =
                    py::cast<decltype(Pooling::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().stride_h =
                    py::cast<decltype(Pooling::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().stride_w =
                    py::cast<decltype(Pooling::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (window_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().window_h =
                    py::cast<decltype(Pooling::window_h)>(py::handle(window_h));
        } CATCH_ALL(-1)
    }

    if (window_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().window_w =
                    py::cast<decltype(Pooling::window_w)>(py::handle(window_w));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().format =
                    py::cast<decltype(Pooling::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (strategy) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().strategy =
                    py::cast<decltype(Pooling::strategy)>(py::handle(strategy));
        } CATCH_ALL(-1)
    }

    if (workspace_limit) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Pooling)*>(self)->inst().workspace_limit =
                    py::cast<decltype(Pooling::workspace_limit)>(py::handle(workspace_limit));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Pooling)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(Pooling, mode), py_set_generic(Pooling, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(Pooling, pad_h), py_set_generic(Pooling, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(Pooling, pad_w), py_set_generic(Pooling, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(Pooling, stride_h), py_set_generic(Pooling, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(Pooling, stride_w), py_set_generic(Pooling, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("window_h"), py_get_generic(Pooling, window_h), py_set_generic(Pooling, window_h), const_cast<char*>("window_h"), NULL},
    {const_cast<char*>("window_w"), py_get_generic(Pooling, window_w), py_set_generic(Pooling, window_w), const_cast<char*>("window_w"), NULL},
    {const_cast<char*>("format"), py_get_generic(Pooling, format), py_set_generic(Pooling, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("strategy"), py_get_generic(Pooling, strategy), py_set_generic(Pooling, strategy), const_cast<char*>("strategy"), NULL},
    {const_cast<char*>("workspace_limit"), py_get_generic(Pooling, workspace_limit), py_set_generic(Pooling, workspace_limit), const_cast<char*>("workspace_limit"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Pooling)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Pooling)::getstate, METH_NOARGS, "Pooling getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Pooling)::setstate, METH_VARARGS, "Pooling setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Pooling)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Pooling)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Pooling)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Pooling)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., window_h: int = ..., window_w: int = ..., format: Union[str, Format] = ..., strategy: Union[str, Strategy] = ...) -> None\n"
};

void _init_py_Pooling(py::module m) {
    using py_op = PyOp(Pooling);
    auto& py_type = PyOpType(Pooling);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Pooling";
    py_type.tp_basicsize = sizeof(PyOp(Pooling));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Pooling";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Pooling), &PyOp(Pooling)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Pooling_Mode(py_type);
    _init_py_Pooling_Format(py_type);
    _init_py_Pooling_Strategy(py_type);

    PyType_Modified(&py_type);
    m.add_object("Pooling", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Pooling::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<RNN::NonlineMode> {
    static constexpr const char *name = "RNN.NonlineMode";
    static constexpr std::underlying_type_t<RNN::NonlineMode> max = 3 - 1;
};
template<> PyTypeObject* EnumWrapper<RNN::NonlineMode>::type = nullptr;

template<> const char*
EnumWrapper<RNN::NonlineMode>::members[] = {"IDENTITY", "RELU", "TANH"};

template<> std::unordered_map<std::string, RNN::NonlineMode>
EnumWrapper<RNN::NonlineMode>::mem2value = {{normalize_enum("IDENTITY"), RNN::NonlineMode::IDENTITY}, {normalize_enum("RELU"), RNN::NonlineMode::RELU}, {normalize_enum("TANH"), RNN::NonlineMode::TANH}};
template<> PyObject* EnumWrapper<RNN::NonlineMode>::pyobj_insts[3] = {nullptr};

void _init_py_RNN_NonlineMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RNN::NonlineMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<RNN::NonlineMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<RNN::NonlineMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<RNN::NonlineMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.RNN.NonlineMode",
        // basicsize
        sizeof(EnumWrapper<RNN::NonlineMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("NonlineMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("RNN.NonlineMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<RNN::NonlineMode>*>(inst)->value = RNN::NonlineMode::IDENTITY;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "IDENTITY", inst) >= 0);
    EnumWrapper<RNN::NonlineMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<RNN::NonlineMode>*>(inst)->value = RNN::NonlineMode::RELU;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "RELU", inst) >= 0);
    EnumWrapper<RNN::NonlineMode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<RNN::NonlineMode>*>(inst)->value = RNN::NonlineMode::TANH;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TANH", inst) >= 0);
    EnumWrapper<RNN::NonlineMode>::pyobj_insts[2] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "NonlineMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RNN_FwdMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RNN::FwdMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "FwdMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(RNN) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RNN)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"num_layers", serialization<decltype(opdef.num_layers)>::dump(opdef.num_layers)},
            {"bidirectional", serialization<decltype(opdef.bidirectional)>::dump(opdef.bidirectional)},
            {"bias", serialization<decltype(opdef.bias)>::dump(opdef.bias)},
            {"hidden_size", serialization<decltype(opdef.hidden_size)>::dump(opdef.hidden_size)},
            {"dropout", serialization<decltype(opdef.dropout)>::dump(opdef.dropout)},
            {"nonlineMode", serialization<decltype(opdef.nonlineMode)>::dump(opdef.nonlineMode)},
            {"fwd_mode", serialization<decltype(opdef.fwd_mode)>::dump(opdef.fwd_mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RNN)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("num_layers");
        if (iter != state.end()) {
            opdef.num_layers = serialization<decltype(opdef.num_layers)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bidirectional");
        if (iter != state.end()) {
            opdef.bidirectional = serialization<decltype(opdef.bidirectional)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bias");
        if (iter != state.end()) {
            opdef.bias = serialization<decltype(opdef.bias)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("hidden_size");
        if (iter != state.end()) {
            opdef.hidden_size = serialization<decltype(opdef.hidden_size)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dropout");
        if (iter != state.end()) {
            opdef.dropout = serialization<decltype(opdef.dropout)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("nonlineMode");
        if (iter != state.end()) {
            opdef.nonlineMode = serialization<decltype(opdef.nonlineMode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("fwd_mode");
        if (iter != state.end()) {
            opdef.fwd_mode = serialization<decltype(opdef.fwd_mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RNN)

int PyOp(RNN)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"num_layers", "bidirectional", "bias", "hidden_size", "dropout", "nonlineMode", "fwd_mode", "scope", NULL};
    PyObject *num_layers = NULL, *bidirectional = NULL, *bias = NULL, *hidden_size = NULL, *dropout = NULL, *nonlineMode = NULL, *fwd_mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOO", const_cast<char**>(kwlist), &num_layers, &bidirectional, &bias, &hidden_size, &dropout, &nonlineMode, &fwd_mode, &scope))
    return -1;

    if (num_layers) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().num_layers =
                    py::cast<decltype(RNN::num_layers)>(py::handle(num_layers));
        } CATCH_ALL(-1)
    }

    if (bidirectional) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().bidirectional =
                    py::cast<decltype(RNN::bidirectional)>(py::handle(bidirectional));
        } CATCH_ALL(-1)
    }

    if (bias) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().bias =
                    py::cast<decltype(RNN::bias)>(py::handle(bias));
        } CATCH_ALL(-1)
    }

    if (hidden_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().hidden_size =
                    py::cast<decltype(RNN::hidden_size)>(py::handle(hidden_size));
        } CATCH_ALL(-1)
    }

    if (dropout) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().dropout =
                    py::cast<decltype(RNN::dropout)>(py::handle(dropout));
        } CATCH_ALL(-1)
    }

    if (nonlineMode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().nonlineMode =
                    py::cast<decltype(RNN::nonlineMode)>(py::handle(nonlineMode));
        } CATCH_ALL(-1)
    }

    if (fwd_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNN)*>(self)->inst().fwd_mode =
                    py::cast<decltype(RNN::fwd_mode)>(py::handle(fwd_mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RNN)::py_getsetters[] = {
    {const_cast<char*>("num_layers"), py_get_generic(RNN, num_layers), py_set_generic(RNN, num_layers), const_cast<char*>("num_layers"), NULL},
    {const_cast<char*>("bidirectional"), py_get_generic(RNN, bidirectional), py_set_generic(RNN, bidirectional), const_cast<char*>("bidirectional"), NULL},
    {const_cast<char*>("bias"), py_get_generic(RNN, bias), py_set_generic(RNN, bias), const_cast<char*>("bias"), NULL},
    {const_cast<char*>("hidden_size"), py_get_generic(RNN, hidden_size), py_set_generic(RNN, hidden_size), const_cast<char*>("hidden_size"), NULL},
    {const_cast<char*>("dropout"), py_get_generic(RNN, dropout), py_set_generic(RNN, dropout), const_cast<char*>("dropout"), NULL},
    {const_cast<char*>("nonlineMode"), py_get_generic(RNN, nonlineMode), py_set_generic(RNN, nonlineMode), const_cast<char*>("nonlineMode"), NULL},
    {const_cast<char*>("fwd_mode"), py_get_generic(RNN, fwd_mode), py_set_generic(RNN, fwd_mode), const_cast<char*>("fwd_mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RNN)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RNN)::getstate, METH_NOARGS, "RNN getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RNN)::setstate, METH_VARARGS, "RNN setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RNN)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RNN)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RNN)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RNN)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, num_layers: int = ..., bidirectional: bool = ..., bias: bool = ..., hidden_size: int = ..., dropout: float = ..., nonlineMode: Union[str, NonlineMode] = ..., fwd_mode: Union[str, FwdMode] = ...) -> None\n"
};

void _init_py_RNN(py::module m) {
    using py_op = PyOp(RNN);
    auto& py_type = PyOpType(RNN);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RNN";
    py_type.tp_basicsize = sizeof(PyOp(RNN));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RNN";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RNN), &PyOp(RNN)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_RNN_NonlineMode(py_type);
    _init_py_RNN_FwdMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("RNN", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RNN::typeinfo(), &py_type).second);
}

void _init_py_RNNCell_NonlineMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RNNCell::NonlineMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "NonlineMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(RNNCell) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RNNCell)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"nonlineMode", serialization<decltype(opdef.nonlineMode)>::dump(opdef.nonlineMode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RNNCell)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("nonlineMode");
        if (iter != state.end()) {
            opdef.nonlineMode = serialization<decltype(opdef.nonlineMode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RNNCell)

int PyOp(RNNCell)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"nonlineMode", "scope", NULL};
    PyObject *nonlineMode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &nonlineMode, &scope))
    return -1;

    if (nonlineMode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RNNCell)*>(self)->inst().nonlineMode =
                    py::cast<decltype(RNNCell::nonlineMode)>(py::handle(nonlineMode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RNNCell)::py_getsetters[] = {
    {const_cast<char*>("nonlineMode"), py_get_generic(RNNCell, nonlineMode), py_set_generic(RNNCell, nonlineMode), const_cast<char*>("nonlineMode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RNNCell)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RNNCell)::getstate, METH_NOARGS, "RNNCell getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RNNCell)::setstate, METH_VARARGS, "RNNCell setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RNNCell)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RNNCell)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RNNCell)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RNNCell)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, nonlineMode: Union[str, NonlineMode] = ...) -> None\n"
};

void _init_py_RNNCell(py::module m) {
    using py_op = PyOp(RNNCell);
    auto& py_type = PyOpType(RNNCell);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RNNCell";
    py_type.tp_basicsize = sizeof(PyOp(RNNCell));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RNNCell";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RNNCell), &PyOp(RNNCell)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_RNNCell_NonlineMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("RNNCell", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RNNCell::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<ROIAlign::Mode> {
    static constexpr const char *name = "ROIAlign.Mode";
    static constexpr std::underlying_type_t<ROIAlign::Mode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<ROIAlign::Mode>::type = nullptr;

template<> const char*
EnumWrapper<ROIAlign::Mode>::members[] = {"MAX", "AVERAGE"};

template<> std::unordered_map<std::string, ROIAlign::Mode>
EnumWrapper<ROIAlign::Mode>::mem2value = {{normalize_enum("MAX"), ROIAlign::Mode::MAX}, {normalize_enum("AVERAGE"), ROIAlign::Mode::AVERAGE}};
template<> PyObject* EnumWrapper<ROIAlign::Mode>::pyobj_insts[2] = {nullptr};

void _init_py_ROIAlign_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ROIAlign::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<ROIAlign::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<ROIAlign::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<ROIAlign::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.ROIAlign.Mode",
        // basicsize
        sizeof(EnumWrapper<ROIAlign::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("ROIAlign.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ROIAlign::Mode>*>(inst)->value = ROIAlign::Mode::MAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MAX", inst) >= 0);
    EnumWrapper<ROIAlign::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ROIAlign::Mode>*>(inst)->value = ROIAlign::Mode::AVERAGE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "AVERAGE", inst) >= 0);
    EnumWrapper<ROIAlign::Mode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_ROIAlign_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ROIAlign::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(ROIAlign) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ROIAlign)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"spatial_scale", serialization<decltype(opdef.spatial_scale)>::dump(opdef.spatial_scale)},
            {"offset", serialization<decltype(opdef.offset)>::dump(opdef.offset)},
            {"pooled_height", serialization<decltype(opdef.pooled_height)>::dump(opdef.pooled_height)},
            {"pooled_width", serialization<decltype(opdef.pooled_width)>::dump(opdef.pooled_width)},
            {"sample_height", serialization<decltype(opdef.sample_height)>::dump(opdef.sample_height)},
            {"sample_width", serialization<decltype(opdef.sample_width)>::dump(opdef.sample_width)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ROIAlign)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("spatial_scale");
        if (iter != state.end()) {
            opdef.spatial_scale = serialization<decltype(opdef.spatial_scale)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("offset");
        if (iter != state.end()) {
            opdef.offset = serialization<decltype(opdef.offset)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pooled_height");
        if (iter != state.end()) {
            opdef.pooled_height = serialization<decltype(opdef.pooled_height)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pooled_width");
        if (iter != state.end()) {
            opdef.pooled_width = serialization<decltype(opdef.pooled_width)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sample_height");
        if (iter != state.end()) {
            opdef.sample_height = serialization<decltype(opdef.sample_height)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sample_width");
        if (iter != state.end()) {
            opdef.sample_width = serialization<decltype(opdef.sample_width)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ROIAlign)

int PyOp(ROIAlign)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "format", "spatial_scale", "offset", "pooled_height", "pooled_width", "sample_height", "sample_width", "scope", NULL};
    PyObject *mode = NULL, *format = NULL, *spatial_scale = NULL, *offset = NULL, *pooled_height = NULL, *pooled_width = NULL, *sample_height = NULL, *sample_width = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", const_cast<char**>(kwlist), &mode, &format, &spatial_scale, &offset, &pooled_height, &pooled_width, &sample_height, &sample_width, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().mode =
                    py::cast<decltype(ROIAlign::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().format =
                    py::cast<decltype(ROIAlign::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (spatial_scale) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().spatial_scale =
                    py::cast<decltype(ROIAlign::spatial_scale)>(py::handle(spatial_scale));
        } CATCH_ALL(-1)
    }

    if (offset) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().offset =
                    py::cast<decltype(ROIAlign::offset)>(py::handle(offset));
        } CATCH_ALL(-1)
    }

    if (pooled_height) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().pooled_height =
                    py::cast<decltype(ROIAlign::pooled_height)>(py::handle(pooled_height));
        } CATCH_ALL(-1)
    }

    if (pooled_width) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().pooled_width =
                    py::cast<decltype(ROIAlign::pooled_width)>(py::handle(pooled_width));
        } CATCH_ALL(-1)
    }

    if (sample_height) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().sample_height =
                    py::cast<decltype(ROIAlign::sample_height)>(py::handle(sample_height));
        } CATCH_ALL(-1)
    }

    if (sample_width) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIAlign)*>(self)->inst().sample_width =
                    py::cast<decltype(ROIAlign::sample_width)>(py::handle(sample_width));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ROIAlign)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(ROIAlign, mode), py_set_generic(ROIAlign, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("format"), py_get_generic(ROIAlign, format), py_set_generic(ROIAlign, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("spatial_scale"), py_get_generic(ROIAlign, spatial_scale), py_set_generic(ROIAlign, spatial_scale), const_cast<char*>("spatial_scale"), NULL},
    {const_cast<char*>("offset"), py_get_generic(ROIAlign, offset), py_set_generic(ROIAlign, offset), const_cast<char*>("offset"), NULL},
    {const_cast<char*>("pooled_height"), py_get_generic(ROIAlign, pooled_height), py_set_generic(ROIAlign, pooled_height), const_cast<char*>("pooled_height"), NULL},
    {const_cast<char*>("pooled_width"), py_get_generic(ROIAlign, pooled_width), py_set_generic(ROIAlign, pooled_width), const_cast<char*>("pooled_width"), NULL},
    {const_cast<char*>("sample_height"), py_get_generic(ROIAlign, sample_height), py_set_generic(ROIAlign, sample_height), const_cast<char*>("sample_height"), NULL},
    {const_cast<char*>("sample_width"), py_get_generic(ROIAlign, sample_width), py_set_generic(ROIAlign, sample_width), const_cast<char*>("sample_width"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ROIAlign)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ROIAlign)::getstate, METH_NOARGS, "ROIAlign getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ROIAlign)::setstate, METH_VARARGS, "ROIAlign setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ROIAlign)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ROIAlign)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ROIAlign)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ROIAlign)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., format: Union[str, Format] = ..., spatial_scale: float = ..., offset: float = ..., pooled_height: int = ..., pooled_width: int = ..., sample_height: int = ..., sample_width: int = ...) -> None\n"
};

void _init_py_ROIAlign(py::module m) {
    using py_op = PyOp(ROIAlign);
    auto& py_type = PyOpType(ROIAlign);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ROIAlign";
    py_type.tp_basicsize = sizeof(PyOp(ROIAlign));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ROIAlign";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ROIAlign), &PyOp(ROIAlign)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_ROIAlign_Mode(py_type);
    _init_py_ROIAlign_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("ROIAlign", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ROIAlign::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<ROIPooling::Mode> {
    static constexpr const char *name = "ROIPooling.Mode";
    static constexpr std::underlying_type_t<ROIPooling::Mode> max = 2 - 1;
};
template<> PyTypeObject* EnumWrapper<ROIPooling::Mode>::type = nullptr;

template<> const char*
EnumWrapper<ROIPooling::Mode>::members[] = {"MAX", "AVERAGE"};

template<> std::unordered_map<std::string, ROIPooling::Mode>
EnumWrapper<ROIPooling::Mode>::mem2value = {{normalize_enum("MAX"), ROIPooling::Mode::MAX}, {normalize_enum("AVERAGE"), ROIPooling::Mode::AVERAGE}};
template<> PyObject* EnumWrapper<ROIPooling::Mode>::pyobj_insts[2] = {nullptr};

void _init_py_ROIPooling_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<ROIPooling::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<ROIPooling::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<ROIPooling::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<ROIPooling::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.ROIPooling.Mode",
        // basicsize
        sizeof(EnumWrapper<ROIPooling::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("ROIPooling.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ROIPooling::Mode>*>(inst)->value = ROIPooling::Mode::MAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MAX", inst) >= 0);
    EnumWrapper<ROIPooling::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<ROIPooling::Mode>*>(inst)->value = ROIPooling::Mode::AVERAGE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "AVERAGE", inst) >= 0);
    EnumWrapper<ROIPooling::Mode>::pyobj_insts[1] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(ROIPooling) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ROIPooling)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"scale", serialization<decltype(opdef.scale)>::dump(opdef.scale)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ROIPooling)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("scale");
        if (iter != state.end()) {
            opdef.scale = serialization<decltype(opdef.scale)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ROIPooling)

int PyOp(ROIPooling)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "scale", "scope", NULL};
    PyObject *mode = NULL, *scale = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &mode, &scale, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIPooling)*>(self)->inst().mode =
                    py::cast<decltype(ROIPooling::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (scale) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ROIPooling)*>(self)->inst().scale =
                    py::cast<decltype(ROIPooling::scale)>(py::handle(scale));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ROIPooling)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(ROIPooling, mode), py_set_generic(ROIPooling, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("scale"), py_get_generic(ROIPooling, scale), py_set_generic(ROIPooling, scale), const_cast<char*>("scale"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ROIPooling)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ROIPooling)::getstate, METH_NOARGS, "ROIPooling getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ROIPooling)::setstate, METH_VARARGS, "ROIPooling setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ROIPooling)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ROIPooling)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ROIPooling)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ROIPooling)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., scale: float = ...) -> None\n"
};

void _init_py_ROIPooling(py::module m) {
    using py_op = PyOp(ROIPooling);
    auto& py_type = PyOpType(ROIPooling);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ROIPooling";
    py_type.tp_basicsize = sizeof(PyOp(ROIPooling));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ROIPooling";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ROIPooling), &PyOp(ROIPooling)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_ROIPooling_Mode(py_type);

    PyType_Modified(&py_type);
    m.add_object("ROIPooling", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ROIPooling::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Reduce::Mode> {
    static constexpr const char *name = "Reduce.Mode";
    static constexpr std::underlying_type_t<Reduce::Mode> max = 6 - 1;
};
template<> PyTypeObject* EnumWrapper<Reduce::Mode>::type = nullptr;

template<> const char*
EnumWrapper<Reduce::Mode>::members[] = {"SUM", "SUM_SQR", "PRODUCT", "MIN", "MAX", "MEAN"};

template<> std::unordered_map<std::string, Reduce::Mode>
EnumWrapper<Reduce::Mode>::mem2value = {{normalize_enum("SUM"), Reduce::Mode::SUM}, {normalize_enum("SUM_SQR"), Reduce::Mode::SUM_SQR}, {normalize_enum("PRODUCT"), Reduce::Mode::PRODUCT}, {normalize_enum("MIN"), Reduce::Mode::MIN}, {normalize_enum("MAX"), Reduce::Mode::MAX}, {normalize_enum("MEAN"), Reduce::Mode::MEAN}};
template<> PyObject* EnumWrapper<Reduce::Mode>::pyobj_insts[6] = {nullptr};

void _init_py_Reduce_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Reduce::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Reduce::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Reduce::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Reduce::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Reduce.Mode",
        // basicsize
        sizeof(EnumWrapper<Reduce::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Reduce.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::Mode>*>(inst)->value = Reduce::Mode::SUM;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SUM", inst) >= 0);
    EnumWrapper<Reduce::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::Mode>*>(inst)->value = Reduce::Mode::SUM_SQR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "SUM_SQR", inst) >= 0);
    EnumWrapper<Reduce::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::Mode>*>(inst)->value = Reduce::Mode::PRODUCT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "PRODUCT", inst) >= 0);
    EnumWrapper<Reduce::Mode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::Mode>*>(inst)->value = Reduce::Mode::MIN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MIN", inst) >= 0);
    EnumWrapper<Reduce::Mode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::Mode>*>(inst)->value = Reduce::Mode::MAX;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MAX", inst) >= 0);
    EnumWrapper<Reduce::Mode>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::Mode>*>(inst)->value = Reduce::Mode::MEAN;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "MEAN", inst) >= 0);
    EnumWrapper<Reduce::Mode>::pyobj_insts[5] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<Reduce::DataType> {
    static constexpr const char *name = "Reduce.DataType";
    static constexpr std::underlying_type_t<Reduce::DataType> max = 6 - 1;
};
template<> PyTypeObject* EnumWrapper<Reduce::DataType>::type = nullptr;

template<> const char*
EnumWrapper<Reduce::DataType>::members[] = {"DEFAULT", "FLOAT_IO16xC32", "FLOAT_O32xC32", "FLOAT_O16xC32", "QUINT_I8xO32", "QINT_I8xO32"};

template<> std::unordered_map<std::string, Reduce::DataType>
EnumWrapper<Reduce::DataType>::mem2value = {{normalize_enum("DEFAULT"), Reduce::DataType::DEFAULT}, {normalize_enum("FLOAT_IO16xC32"), Reduce::DataType::FLOAT_IO16xC32}, {normalize_enum("FLOAT_O32xC32"), Reduce::DataType::FLOAT_O32xC32}, {normalize_enum("FLOAT_O16xC32"), Reduce::DataType::FLOAT_O16xC32}, {normalize_enum("QUINT_I8xO32"), Reduce::DataType::QUINT_I8xO32}, {normalize_enum("QINT_I8xO32"), Reduce::DataType::QINT_I8xO32}};
template<> PyObject* EnumWrapper<Reduce::DataType>::pyobj_insts[6] = {nullptr};

void _init_py_Reduce_DataType(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Reduce::DataType>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Reduce::DataType>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Reduce::DataType>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Reduce::DataType>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Reduce.DataType",
        // basicsize
        sizeof(EnumWrapper<Reduce::DataType>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("DataType").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Reduce.DataType").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::DataType>*>(inst)->value = Reduce::DataType::DEFAULT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "DEFAULT", inst) >= 0);
    EnumWrapper<Reduce::DataType>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::DataType>*>(inst)->value = Reduce::DataType::FLOAT_IO16xC32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT_IO16xC32", inst) >= 0);
    EnumWrapper<Reduce::DataType>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::DataType>*>(inst)->value = Reduce::DataType::FLOAT_O32xC32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT_O32xC32", inst) >= 0);
    EnumWrapper<Reduce::DataType>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::DataType>*>(inst)->value = Reduce::DataType::FLOAT_O16xC32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "FLOAT_O16xC32", inst) >= 0);
    EnumWrapper<Reduce::DataType>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::DataType>*>(inst)->value = Reduce::DataType::QUINT_I8xO32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QUINT_I8xO32", inst) >= 0);
    EnumWrapper<Reduce::DataType>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Reduce::DataType>*>(inst)->value = Reduce::DataType::QINT_I8xO32;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "QINT_I8xO32", inst) >= 0);
    EnumWrapper<Reduce::DataType>::pyobj_insts[5] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "DataType", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Reduce) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Reduce)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"data_type", serialization<decltype(opdef.data_type)>::dump(opdef.data_type)},
            {"keepdim", serialization<decltype(opdef.keepdim)>::dump(opdef.keepdim)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Reduce)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("data_type");
        if (iter != state.end()) {
            opdef.data_type = serialization<decltype(opdef.data_type)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("keepdim");
        if (iter != state.end()) {
            opdef.keepdim = serialization<decltype(opdef.keepdim)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Reduce)

int PyOp(Reduce)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "axis", "data_type", "keepdim", "scope", NULL};
    PyObject *mode = NULL, *axis = NULL, *data_type = NULL, *keepdim = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &mode, &axis, &data_type, &keepdim, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Reduce)*>(self)->inst().mode =
                    py::cast<decltype(Reduce::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Reduce)*>(self)->inst().axis =
                    py::cast<decltype(Reduce::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (data_type) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Reduce)*>(self)->inst().data_type =
                    py::cast<decltype(Reduce::data_type)>(py::handle(data_type));
        } CATCH_ALL(-1)
    }

    if (keepdim) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Reduce)*>(self)->inst().keepdim =
                    py::cast<decltype(Reduce::keepdim)>(py::handle(keepdim));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Reduce)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(Reduce, mode), py_set_generic(Reduce, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("axis"), py_get_generic(Reduce, axis), py_set_generic(Reduce, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("data_type"), py_get_generic(Reduce, data_type), py_set_generic(Reduce, data_type), const_cast<char*>("data_type"), NULL},
    {const_cast<char*>("keepdim"), py_get_generic(Reduce, keepdim), py_set_generic(Reduce, keepdim), const_cast<char*>("keepdim"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Reduce)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Reduce)::getstate, METH_NOARGS, "Reduce getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Reduce)::setstate, METH_VARARGS, "Reduce setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Reduce)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Reduce)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Reduce)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Reduce)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., axis: int = ..., data_type: Union[str, DataType] = ..., keepdim: bool = ...) -> None\n"
};

void _init_py_Reduce(py::module m) {
    using py_op = PyOp(Reduce);
    auto& py_type = PyOpType(Reduce);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Reduce";
    py_type.tp_basicsize = sizeof(PyOp(Reduce));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Reduce";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Reduce), &PyOp(Reduce)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Reduce_Mode(py_type);
    _init_py_Reduce_DataType(py_type);

    PyType_Modified(&py_type);
    m.add_object("Reduce", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Reduce::typeinfo(), &py_type).second);
}

void _init_py_RegionRestrictedConvolution_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolution::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RegionRestrictedConvolution_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolution::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RegionRestrictedConvolution_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolution::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RegionRestrictedConvolution_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolution::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(RegionRestrictedConvolution) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RegionRestrictedConvolution)

int PyOp(RegionRestrictedConvolution)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().mode =
                    py::cast<decltype(RegionRestrictedConvolution::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().pad_h =
                    py::cast<decltype(RegionRestrictedConvolution::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().pad_w =
                    py::cast<decltype(RegionRestrictedConvolution::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().stride_h =
                    py::cast<decltype(RegionRestrictedConvolution::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().stride_w =
                    py::cast<decltype(RegionRestrictedConvolution::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().dilate_h =
                    py::cast<decltype(RegionRestrictedConvolution::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().dilate_w =
                    py::cast<decltype(RegionRestrictedConvolution::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().sparse =
                    py::cast<decltype(RegionRestrictedConvolution::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().format =
                    py::cast<decltype(RegionRestrictedConvolution::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolution)*>(self)->inst().compute_mode =
                    py::cast<decltype(RegionRestrictedConvolution::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RegionRestrictedConvolution)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(RegionRestrictedConvolution, mode), py_set_generic(RegionRestrictedConvolution, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(RegionRestrictedConvolution, pad_h), py_set_generic(RegionRestrictedConvolution, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(RegionRestrictedConvolution, pad_w), py_set_generic(RegionRestrictedConvolution, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(RegionRestrictedConvolution, stride_h), py_set_generic(RegionRestrictedConvolution, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(RegionRestrictedConvolution, stride_w), py_set_generic(RegionRestrictedConvolution, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(RegionRestrictedConvolution, dilate_h), py_set_generic(RegionRestrictedConvolution, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(RegionRestrictedConvolution, dilate_w), py_set_generic(RegionRestrictedConvolution, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(RegionRestrictedConvolution, sparse), py_set_generic(RegionRestrictedConvolution, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(RegionRestrictedConvolution, format), py_set_generic(RegionRestrictedConvolution, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(RegionRestrictedConvolution, compute_mode), py_set_generic(RegionRestrictedConvolution, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RegionRestrictedConvolution)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RegionRestrictedConvolution)::getstate, METH_NOARGS, "RegionRestrictedConvolution getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RegionRestrictedConvolution)::setstate, METH_VARARGS, "RegionRestrictedConvolution setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RegionRestrictedConvolution)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RegionRestrictedConvolution)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RegionRestrictedConvolution)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RegionRestrictedConvolution)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ...) -> None\n"
};

void _init_py_RegionRestrictedConvolution(py::module m) {
    using py_op = PyOp(RegionRestrictedConvolution);
    auto& py_type = PyOpType(RegionRestrictedConvolution);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RegionRestrictedConvolution";
    py_type.tp_basicsize = sizeof(PyOp(RegionRestrictedConvolution));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RegionRestrictedConvolution";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RegionRestrictedConvolution), &PyOp(RegionRestrictedConvolution)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_RegionRestrictedConvolution_Mode(py_type);
    _init_py_RegionRestrictedConvolution_Sparse(py_type);
    _init_py_RegionRestrictedConvolution_Format(py_type);
    _init_py_RegionRestrictedConvolution_ComputeMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("RegionRestrictedConvolution", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RegionRestrictedConvolution::typeinfo(), &py_type).second);
}

void _init_py_RegionRestrictedConvolutionBackwardData_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolutionBackwardData::Mode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RegionRestrictedConvolutionBackwardData_Sparse(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolutionBackwardData::Sparse>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Sparse", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RegionRestrictedConvolutionBackwardData_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolutionBackwardData::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_RegionRestrictedConvolutionBackwardData_ComputeMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<RegionRestrictedConvolutionBackwardData::ComputeMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "ComputeMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(RegionRestrictedConvolutionBackwardData) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"sparse", serialization<decltype(opdef.sparse)>::dump(opdef.sparse)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"compute_mode", serialization<decltype(opdef.compute_mode)>::dump(opdef.compute_mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("sparse");
        if (iter != state.end()) {
            opdef.sparse = serialization<decltype(opdef.sparse)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_mode");
        if (iter != state.end()) {
            opdef.compute_mode = serialization<decltype(opdef.compute_mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RegionRestrictedConvolutionBackwardData)

int PyOp(RegionRestrictedConvolutionBackwardData)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "sparse", "format", "compute_mode", "scope", NULL};
    PyObject *mode = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *sparse = NULL, *format = NULL, *compute_mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOO", const_cast<char**>(kwlist), &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &sparse, &format, &compute_mode, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().mode =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().pad_h =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().pad_w =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().stride_h =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().stride_w =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().dilate_h =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().dilate_w =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (sparse) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().sparse =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::sparse)>(py::handle(sparse));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().format =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (compute_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RegionRestrictedConvolutionBackwardData)*>(self)->inst().compute_mode =
                    py::cast<decltype(RegionRestrictedConvolutionBackwardData::compute_mode)>(py::handle(compute_mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RegionRestrictedConvolutionBackwardData)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(RegionRestrictedConvolutionBackwardData, mode), py_set_generic(RegionRestrictedConvolutionBackwardData, mode), const_cast<char*>("mode"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(RegionRestrictedConvolutionBackwardData, pad_h), py_set_generic(RegionRestrictedConvolutionBackwardData, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(RegionRestrictedConvolutionBackwardData, pad_w), py_set_generic(RegionRestrictedConvolutionBackwardData, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(RegionRestrictedConvolutionBackwardData, stride_h), py_set_generic(RegionRestrictedConvolutionBackwardData, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(RegionRestrictedConvolutionBackwardData, stride_w), py_set_generic(RegionRestrictedConvolutionBackwardData, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(RegionRestrictedConvolutionBackwardData, dilate_h), py_set_generic(RegionRestrictedConvolutionBackwardData, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(RegionRestrictedConvolutionBackwardData, dilate_w), py_set_generic(RegionRestrictedConvolutionBackwardData, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("sparse"), py_get_generic(RegionRestrictedConvolutionBackwardData, sparse), py_set_generic(RegionRestrictedConvolutionBackwardData, sparse), const_cast<char*>("sparse"), NULL},
    {const_cast<char*>("format"), py_get_generic(RegionRestrictedConvolutionBackwardData, format), py_set_generic(RegionRestrictedConvolutionBackwardData, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("compute_mode"), py_get_generic(RegionRestrictedConvolutionBackwardData, compute_mode), py_set_generic(RegionRestrictedConvolutionBackwardData, compute_mode), const_cast<char*>("compute_mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RegionRestrictedConvolutionBackwardData)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RegionRestrictedConvolutionBackwardData)::getstate, METH_NOARGS, "RegionRestrictedConvolutionBackwardData getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RegionRestrictedConvolutionBackwardData)::setstate, METH_VARARGS, "RegionRestrictedConvolutionBackwardData setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RegionRestrictedConvolutionBackwardData)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RegionRestrictedConvolutionBackwardData)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RegionRestrictedConvolutionBackwardData)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RegionRestrictedConvolutionBackwardData)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., sparse: Union[str, Sparse] = ..., format: Union[str, Format] = ..., compute_mode: Union[str, ComputeMode] = ...) -> None\n"
};

void _init_py_RegionRestrictedConvolutionBackwardData(py::module m) {
    using py_op = PyOp(RegionRestrictedConvolutionBackwardData);
    auto& py_type = PyOpType(RegionRestrictedConvolutionBackwardData);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RegionRestrictedConvolutionBackwardData";
    py_type.tp_basicsize = sizeof(PyOp(RegionRestrictedConvolutionBackwardData));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RegionRestrictedConvolutionBackwardData";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RegionRestrictedConvolutionBackwardData), &PyOp(RegionRestrictedConvolutionBackwardData)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_RegionRestrictedConvolutionBackwardData_Mode(py_type);
    _init_py_RegionRestrictedConvolutionBackwardData_Sparse(py_type);
    _init_py_RegionRestrictedConvolutionBackwardData_Format(py_type);
    _init_py_RegionRestrictedConvolutionBackwardData_ComputeMode(py_type);

    PyType_Modified(&py_type);
    m.add_object("RegionRestrictedConvolutionBackwardData", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RegionRestrictedConvolutionBackwardData::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<Remap::InterpolationMode> {
    static constexpr const char *name = "Remap.InterpolationMode";
    static constexpr std::underlying_type_t<Remap::InterpolationMode> max = 5 - 1;
};
template<> PyTypeObject* EnumWrapper<Remap::InterpolationMode>::type = nullptr;

template<> const char*
EnumWrapper<Remap::InterpolationMode>::members[] = {"NEAREST", "LINEAR", "AREA", "CUBIC", "LANCZOS4"};

template<> std::unordered_map<std::string, Remap::InterpolationMode>
EnumWrapper<Remap::InterpolationMode>::mem2value = {{normalize_enum("NEAREST"), Remap::InterpolationMode::NEAREST}, {normalize_enum("LINEAR"), Remap::InterpolationMode::LINEAR}, {normalize_enum("AREA"), Remap::InterpolationMode::AREA}, {normalize_enum("CUBIC"), Remap::InterpolationMode::CUBIC}, {normalize_enum("LANCZOS4"), Remap::InterpolationMode::LANCZOS4}};
template<> PyObject* EnumWrapper<Remap::InterpolationMode>::pyobj_insts[5] = {nullptr};

void _init_py_Remap_InterpolationMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Remap::InterpolationMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Remap::InterpolationMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Remap::InterpolationMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Remap::InterpolationMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Remap.InterpolationMode",
        // basicsize
        sizeof(EnumWrapper<Remap::InterpolationMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("InterpolationMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Remap.InterpolationMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::InterpolationMode>*>(inst)->value = Remap::InterpolationMode::NEAREST;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "NEAREST", inst) >= 0);
    EnumWrapper<Remap::InterpolationMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::InterpolationMode>*>(inst)->value = Remap::InterpolationMode::LINEAR;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LINEAR", inst) >= 0);
    EnumWrapper<Remap::InterpolationMode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::InterpolationMode>*>(inst)->value = Remap::InterpolationMode::AREA;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "AREA", inst) >= 0);
    EnumWrapper<Remap::InterpolationMode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::InterpolationMode>*>(inst)->value = Remap::InterpolationMode::CUBIC;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CUBIC", inst) >= 0);
    EnumWrapper<Remap::InterpolationMode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::InterpolationMode>*>(inst)->value = Remap::InterpolationMode::LANCZOS4;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "LANCZOS4", inst) >= 0);
    EnumWrapper<Remap::InterpolationMode>::pyobj_insts[4] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "InterpolationMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

template<> struct EnumTrait<Remap::BorderMode> {
    static constexpr const char *name = "Remap.BorderMode";
    static constexpr std::underlying_type_t<Remap::BorderMode> max = 7 - 1;
};
template<> PyTypeObject* EnumWrapper<Remap::BorderMode>::type = nullptr;

template<> const char*
EnumWrapper<Remap::BorderMode>::members[] = {"REPLICATE", "REFLECT", "REFLECT_101", "WRAP", "CONSTANT", "TRANSPARENT", "ISOLATED"};

template<> std::unordered_map<std::string, Remap::BorderMode>
EnumWrapper<Remap::BorderMode>::mem2value = {{normalize_enum("REPLICATE"), Remap::BorderMode::REPLICATE}, {normalize_enum("REFLECT"), Remap::BorderMode::REFLECT}, {normalize_enum("REFLECT_101"), Remap::BorderMode::REFLECT_101}, {normalize_enum("WRAP"), Remap::BorderMode::WRAP}, {normalize_enum("CONSTANT"), Remap::BorderMode::CONSTANT}, {normalize_enum("TRANSPARENT"), Remap::BorderMode::TRANSPARENT}, {normalize_enum("ISOLATED"), Remap::BorderMode::ISOLATED}};
template<> PyObject* EnumWrapper<Remap::BorderMode>::pyobj_insts[7] = {nullptr};

void _init_py_Remap_BorderMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Remap::BorderMode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<Remap::BorderMode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<Remap::BorderMode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<Remap::BorderMode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.Remap.BorderMode",
        // basicsize
        sizeof(EnumWrapper<Remap::BorderMode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("BorderMode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("Remap.BorderMode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::REPLICATE;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REPLICATE", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::REFLECT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REFLECT", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::REFLECT_101;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "REFLECT_101", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[2] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::WRAP;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "WRAP", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[3] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::CONSTANT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "CONSTANT", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[4] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::TRANSPARENT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "TRANSPARENT", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[5] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<Remap::BorderMode>*>(inst)->value = Remap::BorderMode::ISOLATED;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "ISOLATED", inst) >= 0);
    EnumWrapper<Remap::BorderMode>::pyobj_insts[6] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "BorderMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Remap_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Remap::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Remap) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Remap)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"imode", serialization<decltype(opdef.imode)>::dump(opdef.imode)},
            {"border_type", serialization<decltype(opdef.border_type)>::dump(opdef.border_type)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"scalar", serialization<decltype(opdef.scalar)>::dump(opdef.scalar)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Remap)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("imode");
        if (iter != state.end()) {
            opdef.imode = serialization<decltype(opdef.imode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("border_type");
        if (iter != state.end()) {
            opdef.border_type = serialization<decltype(opdef.border_type)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("scalar");
        if (iter != state.end()) {
            opdef.scalar = serialization<decltype(opdef.scalar)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Remap)

int PyOp(Remap)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"imode", "border_type", "format", "scalar", "scope", NULL};
    PyObject *imode = NULL, *border_type = NULL, *format = NULL, *scalar = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &imode, &border_type, &format, &scalar, &scope))
    return -1;

    if (imode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Remap)*>(self)->inst().imode =
                    py::cast<decltype(Remap::imode)>(py::handle(imode));
        } CATCH_ALL(-1)
    }

    if (border_type) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Remap)*>(self)->inst().border_type =
                    py::cast<decltype(Remap::border_type)>(py::handle(border_type));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Remap)*>(self)->inst().format =
                    py::cast<decltype(Remap::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (scalar) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Remap)*>(self)->inst().scalar =
                    py::cast<decltype(Remap::scalar)>(py::handle(scalar));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Remap)::py_getsetters[] = {
    {const_cast<char*>("imode"), py_get_generic(Remap, imode), py_set_generic(Remap, imode), const_cast<char*>("imode"), NULL},
    {const_cast<char*>("border_type"), py_get_generic(Remap, border_type), py_set_generic(Remap, border_type), const_cast<char*>("border_type"), NULL},
    {const_cast<char*>("format"), py_get_generic(Remap, format), py_set_generic(Remap, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("scalar"), py_get_generic(Remap, scalar), py_set_generic(Remap, scalar), const_cast<char*>("scalar"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Remap)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Remap)::getstate, METH_NOARGS, "Remap getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Remap)::setstate, METH_VARARGS, "Remap setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Remap)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Remap)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Remap)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Remap)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, imode: Union[str, InterpolationMode] = ..., border_type: Union[str, BorderMode] = ..., format: Union[str, Format] = ..., scalar: float = ...) -> None\n"
};

void _init_py_Remap(py::module m) {
    using py_op = PyOp(Remap);
    auto& py_type = PyOpType(Remap);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Remap";
    py_type.tp_basicsize = sizeof(PyOp(Remap));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Remap";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Remap), &PyOp(Remap)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Remap_InterpolationMode(py_type);
    _init_py_Remap_BorderMode(py_type);
    _init_py_Remap_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("Remap", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Remap::typeinfo(), &py_type).second);
}

PyOpDefBegin(RemoteRecv) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"key", serialization<decltype(opdef.key)>::dump(opdef.key)},
            {"addr", serialization<decltype(opdef.addr)>::dump(opdef.addr)},
            {"port", serialization<decltype(opdef.port)>::dump(opdef.port)},
            {"rank_from", serialization<decltype(opdef.rank_from)>::dump(opdef.rank_from)},
            {"cn", serialization<decltype(opdef.cn)>::dump(opdef.cn)},
            {"shape", serialization<decltype(opdef.shape)>::dump(opdef.shape)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)},
            {"backend", serialization<decltype(opdef.backend)>::dump(opdef.backend)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("key");
        if (iter != state.end()) {
            opdef.key = serialization<decltype(opdef.key)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("addr");
        if (iter != state.end()) {
            opdef.addr = serialization<decltype(opdef.addr)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("port");
        if (iter != state.end()) {
            opdef.port = serialization<decltype(opdef.port)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("rank_from");
        if (iter != state.end()) {
            opdef.rank_from = serialization<decltype(opdef.rank_from)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("cn");
        if (iter != state.end()) {
            opdef.cn = serialization<decltype(opdef.cn)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("shape");
        if (iter != state.end()) {
            opdef.shape = serialization<decltype(opdef.shape)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("backend");
        if (iter != state.end()) {
            opdef.backend = serialization<decltype(opdef.backend)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RemoteRecv)

int PyOp(RemoteRecv)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"key", "addr", "port", "rank_from", "cn", "shape", "dtype", "backend", "scope", NULL};
    PyObject *key = NULL, *addr = NULL, *port = NULL, *rank_from = NULL, *cn = NULL, *shape = NULL, *dtype = NULL, *backend = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", const_cast<char**>(kwlist), &key, &addr, &port, &rank_from, &cn, &shape, &dtype, &backend, &scope))
    return -1;

    if (key) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().key =
                    py::cast<decltype(RemoteRecv::key)>(py::handle(key));
        } CATCH_ALL(-1)
    }

    if (addr) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().addr =
                    py::cast<decltype(RemoteRecv::addr)>(py::handle(addr));
        } CATCH_ALL(-1)
    }

    if (port) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().port =
                    py::cast<decltype(RemoteRecv::port)>(py::handle(port));
        } CATCH_ALL(-1)
    }

    if (rank_from) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().rank_from =
                    py::cast<decltype(RemoteRecv::rank_from)>(py::handle(rank_from));
        } CATCH_ALL(-1)
    }

    if (cn) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().cn =
                    py::cast<decltype(RemoteRecv::cn)>(py::handle(cn));
        } CATCH_ALL(-1)
    }

    if (shape) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().shape =
                    py::cast<decltype(RemoteRecv::shape)>(py::handle(shape));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().dtype =
                    py::cast<decltype(RemoteRecv::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (backend) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteRecv)*>(self)->inst().backend =
                    py::cast<decltype(RemoteRecv::backend)>(py::handle(backend));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RemoteRecv)::py_getsetters[] = {
    {const_cast<char*>("key"), py_get_generic(RemoteRecv, key), py_set_generic(RemoteRecv, key), const_cast<char*>("key"), NULL},
    {const_cast<char*>("addr"), py_get_generic(RemoteRecv, addr), py_set_generic(RemoteRecv, addr), const_cast<char*>("addr"), NULL},
    {const_cast<char*>("port"), py_get_generic(RemoteRecv, port), py_set_generic(RemoteRecv, port), const_cast<char*>("port"), NULL},
    {const_cast<char*>("rank_from"), py_get_generic(RemoteRecv, rank_from), py_set_generic(RemoteRecv, rank_from), const_cast<char*>("rank_from"), NULL},
    {const_cast<char*>("cn"), py_get_generic(RemoteRecv, cn), py_set_generic(RemoteRecv, cn), const_cast<char*>("cn"), NULL},
    {const_cast<char*>("shape"), py_get_generic(RemoteRecv, shape), py_set_generic(RemoteRecv, shape), const_cast<char*>("shape"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(RemoteRecv, dtype), py_set_generic(RemoteRecv, dtype), const_cast<char*>("dtype"), NULL},
    {const_cast<char*>("backend"), py_get_generic(RemoteRecv, backend), py_set_generic(RemoteRecv, backend), const_cast<char*>("backend"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RemoteRecv)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RemoteRecv)::getstate, METH_NOARGS, "RemoteRecv getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RemoteRecv)::setstate, METH_VARARGS, "RemoteRecv setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RemoteRecv)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RemoteRecv)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RemoteRecv)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RemoteRecv)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, key: str = ..., addr: str = ..., port: int = ..., rank_from: int = ..., cn: str = ..., shape: list[int] = ..., dtype: str = ..., backend: str = ...) -> None\n"
};

void _init_py_RemoteRecv(py::module m) {
    using py_op = PyOp(RemoteRecv);
    auto& py_type = PyOpType(RemoteRecv);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RemoteRecv";
    py_type.tp_basicsize = sizeof(PyOp(RemoteRecv));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RemoteRecv";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RemoteRecv), &PyOp(RemoteRecv)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("RemoteRecv", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RemoteRecv::typeinfo(), &py_type).second);
}

PyOpDefBegin(RemoteSend) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RemoteSend)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"key", serialization<decltype(opdef.key)>::dump(opdef.key)},
            {"addr", serialization<decltype(opdef.addr)>::dump(opdef.addr)},
            {"port", serialization<decltype(opdef.port)>::dump(opdef.port)},
            {"rank_to", serialization<decltype(opdef.rank_to)>::dump(opdef.rank_to)},
            {"backend", serialization<decltype(opdef.backend)>::dump(opdef.backend)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RemoteSend)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("key");
        if (iter != state.end()) {
            opdef.key = serialization<decltype(opdef.key)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("addr");
        if (iter != state.end()) {
            opdef.addr = serialization<decltype(opdef.addr)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("port");
        if (iter != state.end()) {
            opdef.port = serialization<decltype(opdef.port)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("rank_to");
        if (iter != state.end()) {
            opdef.rank_to = serialization<decltype(opdef.rank_to)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("backend");
        if (iter != state.end()) {
            opdef.backend = serialization<decltype(opdef.backend)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RemoteSend)

int PyOp(RemoteSend)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"key", "addr", "port", "rank_to", "backend", "scope", NULL};
    PyObject *key = NULL, *addr = NULL, *port = NULL, *rank_to = NULL, *backend = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOO", const_cast<char**>(kwlist), &key, &addr, &port, &rank_to, &backend, &scope))
    return -1;

    if (key) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteSend)*>(self)->inst().key =
                    py::cast<decltype(RemoteSend::key)>(py::handle(key));
        } CATCH_ALL(-1)
    }

    if (addr) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteSend)*>(self)->inst().addr =
                    py::cast<decltype(RemoteSend::addr)>(py::handle(addr));
        } CATCH_ALL(-1)
    }

    if (port) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteSend)*>(self)->inst().port =
                    py::cast<decltype(RemoteSend::port)>(py::handle(port));
        } CATCH_ALL(-1)
    }

    if (rank_to) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteSend)*>(self)->inst().rank_to =
                    py::cast<decltype(RemoteSend::rank_to)>(py::handle(rank_to));
        } CATCH_ALL(-1)
    }

    if (backend) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoteSend)*>(self)->inst().backend =
                    py::cast<decltype(RemoteSend::backend)>(py::handle(backend));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RemoteSend)::py_getsetters[] = {
    {const_cast<char*>("key"), py_get_generic(RemoteSend, key), py_set_generic(RemoteSend, key), const_cast<char*>("key"), NULL},
    {const_cast<char*>("addr"), py_get_generic(RemoteSend, addr), py_set_generic(RemoteSend, addr), const_cast<char*>("addr"), NULL},
    {const_cast<char*>("port"), py_get_generic(RemoteSend, port), py_set_generic(RemoteSend, port), const_cast<char*>("port"), NULL},
    {const_cast<char*>("rank_to"), py_get_generic(RemoteSend, rank_to), py_set_generic(RemoteSend, rank_to), const_cast<char*>("rank_to"), NULL},
    {const_cast<char*>("backend"), py_get_generic(RemoteSend, backend), py_set_generic(RemoteSend, backend), const_cast<char*>("backend"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RemoteSend)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RemoteSend)::getstate, METH_NOARGS, "RemoteSend getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RemoteSend)::setstate, METH_VARARGS, "RemoteSend setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RemoteSend)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RemoteSend)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RemoteSend)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RemoteSend)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, key: str = ..., addr: str = ..., port: int = ..., rank_to: int = ..., backend: str = ...) -> None\n"
};

void _init_py_RemoteSend(py::module m) {
    using py_op = PyOp(RemoteSend);
    auto& py_type = PyOpType(RemoteSend);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RemoteSend";
    py_type.tp_basicsize = sizeof(PyOp(RemoteSend));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RemoteSend";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RemoteSend), &PyOp(RemoteSend)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("RemoteSend", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RemoteSend::typeinfo(), &py_type).second);
}

PyOpDefBegin(RemoveAxis) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(RemoveAxis)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(RemoveAxis)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(RemoveAxis)

int PyOp(RemoveAxis)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "scope", NULL};
    PyObject *axis = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &axis, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(RemoveAxis)*>(self)->inst().axis =
                    py::cast<decltype(RemoveAxis::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(RemoveAxis)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(RemoveAxis, axis), py_set_generic(RemoveAxis, axis), const_cast<char*>("axis"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(RemoveAxis)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(RemoveAxis)::getstate, METH_NOARGS, "RemoveAxis getstate"},
    {const_cast<char*>("__setstate__"), PyOp(RemoveAxis)::setstate, METH_VARARGS, "RemoveAxis setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(RemoveAxis)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(RemoveAxis)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(RemoveAxis)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(RemoveAxis)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: list[int] = ...) -> None\n"
};

void _init_py_RemoveAxis(py::module m) {
    using py_op = PyOp(RemoveAxis);
    auto& py_type = PyOpType(RemoveAxis);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.RemoveAxis";
    py_type.tp_basicsize = sizeof(PyOp(RemoveAxis));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "RemoveAxis";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(RemoveAxis), &PyOp(RemoveAxis)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("RemoveAxis", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(RemoveAxis::typeinfo(), &py_type).second);
}

PyOpDefBegin(Reshape) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Reshape)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"shape", serialization<decltype(opdef.shape)>::dump(opdef.shape)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Reshape)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("shape");
        if (iter != state.end()) {
            opdef.shape = serialization<decltype(opdef.shape)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Reshape)

int PyOp(Reshape)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "shape", "scope", NULL};
    PyObject *axis = NULL, *shape = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &axis, &shape, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Reshape)*>(self)->inst().axis =
                    py::cast<decltype(Reshape::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (shape) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Reshape)*>(self)->inst().shape =
                    py::cast<decltype(Reshape::shape)>(py::handle(shape));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Reshape)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Reshape, axis), py_set_generic(Reshape, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("shape"), py_get_generic(Reshape, shape), py_set_generic(Reshape, shape), const_cast<char*>("shape"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Reshape)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Reshape)::getstate, METH_NOARGS, "Reshape getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Reshape)::setstate, METH_VARARGS, "Reshape setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Reshape)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Reshape)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Reshape)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Reshape)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ..., shape: list[int] = ...) -> None\n"
};

void _init_py_Reshape(py::module m) {
    using py_op = PyOp(Reshape);
    auto& py_type = PyOpType(Reshape);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Reshape";
    py_type.tp_basicsize = sizeof(PyOp(Reshape));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Reshape";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Reshape), &PyOp(Reshape)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Reshape", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Reshape::typeinfo(), &py_type).second);
}

void _init_py_Resize_InterpolationMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Resize::InterpolationMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "InterpolationMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_Resize_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<Resize::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(Resize) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Resize)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"imode", serialization<decltype(opdef.imode)>::dump(opdef.imode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Resize)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("imode");
        if (iter != state.end()) {
            opdef.imode = serialization<decltype(opdef.imode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Resize)

int PyOp(Resize)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"imode", "format", "scope", NULL};
    PyObject *imode = NULL, *format = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &imode, &format, &scope))
    return -1;

    if (imode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Resize)*>(self)->inst().imode =
                    py::cast<decltype(Resize::imode)>(py::handle(imode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Resize)*>(self)->inst().format =
                    py::cast<decltype(Resize::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Resize)::py_getsetters[] = {
    {const_cast<char*>("imode"), py_get_generic(Resize, imode), py_set_generic(Resize, imode), const_cast<char*>("imode"), NULL},
    {const_cast<char*>("format"), py_get_generic(Resize, format), py_set_generic(Resize, format), const_cast<char*>("format"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Resize)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Resize)::getstate, METH_NOARGS, "Resize getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Resize)::setstate, METH_VARARGS, "Resize setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Resize)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Resize)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Resize)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Resize)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, imode: Union[str, InterpolationMode] = ..., format: Union[str, Format] = ...) -> None\n"
};

void _init_py_Resize(py::module m) {
    using py_op = PyOp(Resize);
    auto& py_type = PyOpType(Resize);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Resize";
    py_type.tp_basicsize = sizeof(PyOp(Resize));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Resize";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Resize), &PyOp(Resize)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_Resize_InterpolationMode(py_type);
    _init_py_Resize_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("Resize", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Resize::typeinfo(), &py_type).second);
}

PyOpDefBegin(SVD) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(SVD)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"full_matrices", serialization<decltype(opdef.full_matrices)>::dump(opdef.full_matrices)},
            {"compute_uv", serialization<decltype(opdef.compute_uv)>::dump(opdef.compute_uv)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(SVD)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("full_matrices");
        if (iter != state.end()) {
            opdef.full_matrices = serialization<decltype(opdef.full_matrices)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("compute_uv");
        if (iter != state.end()) {
            opdef.compute_uv = serialization<decltype(opdef.compute_uv)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(SVD)

int PyOp(SVD)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"full_matrices", "compute_uv", "scope", NULL};
    PyObject *full_matrices = NULL, *compute_uv = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &full_matrices, &compute_uv, &scope))
    return -1;

    if (full_matrices) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SVD)*>(self)->inst().full_matrices =
                    py::cast<decltype(SVD::full_matrices)>(py::handle(full_matrices));
        } CATCH_ALL(-1)
    }

    if (compute_uv) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SVD)*>(self)->inst().compute_uv =
                    py::cast<decltype(SVD::compute_uv)>(py::handle(compute_uv));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(SVD)::py_getsetters[] = {
    {const_cast<char*>("full_matrices"), py_get_generic(SVD, full_matrices), py_set_generic(SVD, full_matrices), const_cast<char*>("full_matrices"), NULL},
    {const_cast<char*>("compute_uv"), py_get_generic(SVD, compute_uv), py_set_generic(SVD, compute_uv), const_cast<char*>("compute_uv"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(SVD)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(SVD)::getstate, METH_NOARGS, "SVD getstate"},
    {const_cast<char*>("__setstate__"), PyOp(SVD)::setstate, METH_VARARGS, "SVD setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(SVD)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(SVD)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(SVD)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(SVD)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, full_matrices: bool = ..., compute_uv: bool = ...) -> None\n"
};

void _init_py_SVD(py::module m) {
    using py_op = PyOp(SVD);
    auto& py_type = PyOpType(SVD);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.SVD";
    py_type.tp_basicsize = sizeof(PyOp(SVD));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "SVD";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(SVD), &PyOp(SVD)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("SVD", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(SVD::typeinfo(), &py_type).second);
}

PyOpDefBegin(SetMeshIndexing) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(SetMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(SetMeshIndexing)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(SetMeshIndexing)

int PyOp(SetMeshIndexing)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SetMeshIndexing)*>(self)->inst().items =
                    py::cast<decltype(SetMeshIndexing::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(SetMeshIndexing)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(SetMeshIndexing, items), py_set_generic(SetMeshIndexing, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(SetMeshIndexing)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(SetMeshIndexing)::getstate, METH_NOARGS, "SetMeshIndexing getstate"},
    {const_cast<char*>("__setstate__"), PyOp(SetMeshIndexing)::setstate, METH_VARARGS, "SetMeshIndexing setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(SetMeshIndexing)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(SetMeshIndexing)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(SetMeshIndexing)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(SetMeshIndexing)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_SetMeshIndexing(py::module m) {
    using py_op = PyOp(SetMeshIndexing);
    auto& py_type = PyOpType(SetMeshIndexing);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.SetMeshIndexing";
    py_type.tp_basicsize = sizeof(PyOp(SetMeshIndexing));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "SetMeshIndexing";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(SetMeshIndexing), &PyOp(SetMeshIndexing)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("SetMeshIndexing", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(SetMeshIndexing::typeinfo(), &py_type).second);
}

PyOpDefBegin(SetSubtensor) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(SetSubtensor)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(SetSubtensor)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(SetSubtensor)

int PyOp(SetSubtensor)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SetSubtensor)*>(self)->inst().items =
                    py::cast<decltype(SetSubtensor::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(SetSubtensor)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(SetSubtensor, items), py_set_generic(SetSubtensor, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(SetSubtensor)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(SetSubtensor)::getstate, METH_NOARGS, "SetSubtensor getstate"},
    {const_cast<char*>("__setstate__"), PyOp(SetSubtensor)::setstate, METH_VARARGS, "SetSubtensor setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(SetSubtensor)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(SetSubtensor)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(SetSubtensor)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(SetSubtensor)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_SetSubtensor(py::module m) {
    using py_op = PyOp(SetSubtensor);
    auto& py_type = PyOpType(SetSubtensor);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.SetSubtensor";
    py_type.tp_basicsize = sizeof(PyOp(SetSubtensor));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "SetSubtensor";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(SetSubtensor), &PyOp(SetSubtensor)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("SetSubtensor", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(SetSubtensor::typeinfo(), &py_type).second);
}

PyOpDefBegin(ShuffleRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(ShuffleRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(ShuffleRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(ShuffleRNG)

int PyOp(ShuffleRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "handle", "scope", NULL};
    PyObject *seed = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &seed, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ShuffleRNG)*>(self)->inst().seed =
                    py::cast<decltype(ShuffleRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(ShuffleRNG)*>(self)->inst().handle =
                    py::cast<decltype(ShuffleRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(ShuffleRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(ShuffleRNG, seed), py_set_generic(ShuffleRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("handle"), py_get_generic(ShuffleRNG, handle), py_set_generic(ShuffleRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(ShuffleRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(ShuffleRNG)::getstate, METH_NOARGS, "ShuffleRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(ShuffleRNG)::setstate, METH_VARARGS, "ShuffleRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(ShuffleRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(ShuffleRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(ShuffleRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(ShuffleRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., handle: int = ...) -> None\n"
};

void _init_py_ShuffleRNG(py::module m) {
    using py_op = PyOp(ShuffleRNG);
    auto& py_type = PyOpType(ShuffleRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.ShuffleRNG";
    py_type.tp_basicsize = sizeof(PyOp(ShuffleRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "ShuffleRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(ShuffleRNG), &PyOp(ShuffleRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("ShuffleRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(ShuffleRNG::typeinfo(), &py_type).second);
}

PyOpDefBegin(SlidingWindowTranspose) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"out_h", serialization<decltype(opdef.out_h)>::dump(opdef.out_h)},
            {"out_w", serialization<decltype(opdef.out_w)>::dump(opdef.out_w)},
            {"pad_h", serialization<decltype(opdef.pad_h)>::dump(opdef.pad_h)},
            {"pad_w", serialization<decltype(opdef.pad_w)>::dump(opdef.pad_w)},
            {"stride_h", serialization<decltype(opdef.stride_h)>::dump(opdef.stride_h)},
            {"stride_w", serialization<decltype(opdef.stride_w)>::dump(opdef.stride_w)},
            {"dilate_h", serialization<decltype(opdef.dilate_h)>::dump(opdef.dilate_h)},
            {"dilate_w", serialization<decltype(opdef.dilate_w)>::dump(opdef.dilate_w)},
            {"window_h", serialization<decltype(opdef.window_h)>::dump(opdef.window_h)},
            {"window_w", serialization<decltype(opdef.window_w)>::dump(opdef.window_w)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("out_h");
        if (iter != state.end()) {
            opdef.out_h = serialization<decltype(opdef.out_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("out_w");
        if (iter != state.end()) {
            opdef.out_w = serialization<decltype(opdef.out_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_h");
        if (iter != state.end()) {
            opdef.pad_h = serialization<decltype(opdef.pad_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("pad_w");
        if (iter != state.end()) {
            opdef.pad_w = serialization<decltype(opdef.pad_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_h");
        if (iter != state.end()) {
            opdef.stride_h = serialization<decltype(opdef.stride_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("stride_w");
        if (iter != state.end()) {
            opdef.stride_w = serialization<decltype(opdef.stride_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_h");
        if (iter != state.end()) {
            opdef.dilate_h = serialization<decltype(opdef.dilate_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dilate_w");
        if (iter != state.end()) {
            opdef.dilate_w = serialization<decltype(opdef.dilate_w)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("window_h");
        if (iter != state.end()) {
            opdef.window_h = serialization<decltype(opdef.window_h)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("window_w");
        if (iter != state.end()) {
            opdef.window_w = serialization<decltype(opdef.window_w)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(SlidingWindowTranspose)

int PyOp(SlidingWindowTranspose)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"out_h", "out_w", "pad_h", "pad_w", "stride_h", "stride_w", "dilate_h", "dilate_w", "window_h", "window_w", "scope", NULL};
    PyObject *out_h = NULL, *out_w = NULL, *pad_h = NULL, *pad_w = NULL, *stride_h = NULL, *stride_w = NULL, *dilate_h = NULL, *dilate_w = NULL, *window_h = NULL, *window_w = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOOOO", const_cast<char**>(kwlist), &out_h, &out_w, &pad_h, &pad_w, &stride_h, &stride_w, &dilate_h, &dilate_w, &window_h, &window_w, &scope))
    return -1;

    if (out_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().out_h =
                    py::cast<decltype(SlidingWindowTranspose::out_h)>(py::handle(out_h));
        } CATCH_ALL(-1)
    }

    if (out_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().out_w =
                    py::cast<decltype(SlidingWindowTranspose::out_w)>(py::handle(out_w));
        } CATCH_ALL(-1)
    }

    if (pad_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().pad_h =
                    py::cast<decltype(SlidingWindowTranspose::pad_h)>(py::handle(pad_h));
        } CATCH_ALL(-1)
    }

    if (pad_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().pad_w =
                    py::cast<decltype(SlidingWindowTranspose::pad_w)>(py::handle(pad_w));
        } CATCH_ALL(-1)
    }

    if (stride_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().stride_h =
                    py::cast<decltype(SlidingWindowTranspose::stride_h)>(py::handle(stride_h));
        } CATCH_ALL(-1)
    }

    if (stride_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().stride_w =
                    py::cast<decltype(SlidingWindowTranspose::stride_w)>(py::handle(stride_w));
        } CATCH_ALL(-1)
    }

    if (dilate_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().dilate_h =
                    py::cast<decltype(SlidingWindowTranspose::dilate_h)>(py::handle(dilate_h));
        } CATCH_ALL(-1)
    }

    if (dilate_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().dilate_w =
                    py::cast<decltype(SlidingWindowTranspose::dilate_w)>(py::handle(dilate_w));
        } CATCH_ALL(-1)
    }

    if (window_h) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().window_h =
                    py::cast<decltype(SlidingWindowTranspose::window_h)>(py::handle(window_h));
        } CATCH_ALL(-1)
    }

    if (window_w) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(SlidingWindowTranspose)*>(self)->inst().window_w =
                    py::cast<decltype(SlidingWindowTranspose::window_w)>(py::handle(window_w));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(SlidingWindowTranspose)::py_getsetters[] = {
    {const_cast<char*>("out_h"), py_get_generic(SlidingWindowTranspose, out_h), py_set_generic(SlidingWindowTranspose, out_h), const_cast<char*>("out_h"), NULL},
    {const_cast<char*>("out_w"), py_get_generic(SlidingWindowTranspose, out_w), py_set_generic(SlidingWindowTranspose, out_w), const_cast<char*>("out_w"), NULL},
    {const_cast<char*>("pad_h"), py_get_generic(SlidingWindowTranspose, pad_h), py_set_generic(SlidingWindowTranspose, pad_h), const_cast<char*>("pad_h"), NULL},
    {const_cast<char*>("pad_w"), py_get_generic(SlidingWindowTranspose, pad_w), py_set_generic(SlidingWindowTranspose, pad_w), const_cast<char*>("pad_w"), NULL},
    {const_cast<char*>("stride_h"), py_get_generic(SlidingWindowTranspose, stride_h), py_set_generic(SlidingWindowTranspose, stride_h), const_cast<char*>("stride_h"), NULL},
    {const_cast<char*>("stride_w"), py_get_generic(SlidingWindowTranspose, stride_w), py_set_generic(SlidingWindowTranspose, stride_w), const_cast<char*>("stride_w"), NULL},
    {const_cast<char*>("dilate_h"), py_get_generic(SlidingWindowTranspose, dilate_h), py_set_generic(SlidingWindowTranspose, dilate_h), const_cast<char*>("dilate_h"), NULL},
    {const_cast<char*>("dilate_w"), py_get_generic(SlidingWindowTranspose, dilate_w), py_set_generic(SlidingWindowTranspose, dilate_w), const_cast<char*>("dilate_w"), NULL},
    {const_cast<char*>("window_h"), py_get_generic(SlidingWindowTranspose, window_h), py_set_generic(SlidingWindowTranspose, window_h), const_cast<char*>("window_h"), NULL},
    {const_cast<char*>("window_w"), py_get_generic(SlidingWindowTranspose, window_w), py_set_generic(SlidingWindowTranspose, window_w), const_cast<char*>("window_w"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(SlidingWindowTranspose)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(SlidingWindowTranspose)::getstate, METH_NOARGS, "SlidingWindowTranspose getstate"},
    {const_cast<char*>("__setstate__"), PyOp(SlidingWindowTranspose)::setstate, METH_VARARGS, "SlidingWindowTranspose setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(SlidingWindowTranspose)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(SlidingWindowTranspose)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(SlidingWindowTranspose)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(SlidingWindowTranspose)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, out_h: int = ..., out_w: int = ..., pad_h: int = ..., pad_w: int = ..., stride_h: int = ..., stride_w: int = ..., dilate_h: int = ..., dilate_w: int = ..., window_h: int = ..., window_w: int = ...) -> None\n"
};

void _init_py_SlidingWindowTranspose(py::module m) {
    using py_op = PyOp(SlidingWindowTranspose);
    auto& py_type = PyOpType(SlidingWindowTranspose);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.SlidingWindowTranspose";
    py_type.tp_basicsize = sizeof(PyOp(SlidingWindowTranspose));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "SlidingWindowTranspose";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(SlidingWindowTranspose), &PyOp(SlidingWindowTranspose)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("SlidingWindowTranspose", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(SlidingWindowTranspose::typeinfo(), &py_type).second);
}

PyOpDefBegin(Softmax) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Softmax)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Softmax)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Softmax)

int PyOp(Softmax)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "scope", NULL};
    PyObject *axis = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &axis, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Softmax)*>(self)->inst().axis =
                    py::cast<decltype(Softmax::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Softmax)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Softmax, axis), py_set_generic(Softmax, axis), const_cast<char*>("axis"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Softmax)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Softmax)::getstate, METH_NOARGS, "Softmax getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Softmax)::setstate, METH_VARARGS, "Softmax setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Softmax)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Softmax)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Softmax)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Softmax)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ...) -> None\n"
};

void _init_py_Softmax(py::module m) {
    using py_op = PyOp(Softmax);
    auto& py_type = PyOpType(Softmax);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Softmax";
    py_type.tp_basicsize = sizeof(PyOp(Softmax));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Softmax";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Softmax), &PyOp(Softmax)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Softmax", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Softmax::typeinfo(), &py_type).second);
}

PyOpDefBegin(Split) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Split)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"axis", serialization<decltype(opdef.axis)>::dump(opdef.axis)},
            {"nsections", serialization<decltype(opdef.nsections)>::dump(opdef.nsections)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Split)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("axis");
        if (iter != state.end()) {
            opdef.axis = serialization<decltype(opdef.axis)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("nsections");
        if (iter != state.end()) {
            opdef.nsections = serialization<decltype(opdef.nsections)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Split)

int PyOp(Split)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"axis", "nsections", "scope", NULL};
    PyObject *axis = NULL, *nsections = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &axis, &nsections, &scope))
    return -1;

    if (axis) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Split)*>(self)->inst().axis =
                    py::cast<decltype(Split::axis)>(py::handle(axis));
        } CATCH_ALL(-1)
    }

    if (nsections) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Split)*>(self)->inst().nsections =
                    py::cast<decltype(Split::nsections)>(py::handle(nsections));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Split)::py_getsetters[] = {
    {const_cast<char*>("axis"), py_get_generic(Split, axis), py_set_generic(Split, axis), const_cast<char*>("axis"), NULL},
    {const_cast<char*>("nsections"), py_get_generic(Split, nsections), py_set_generic(Split, nsections), const_cast<char*>("nsections"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Split)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Split)::getstate, METH_NOARGS, "Split getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Split)::setstate, METH_VARARGS, "Split setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Split)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Split)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Split)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Split)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, axis: int = ..., nsections: int = ...) -> None\n"
};

void _init_py_Split(py::module m) {
    using py_op = PyOp(Split);
    auto& py_type = PyOpType(Split);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Split";
    py_type.tp_basicsize = sizeof(PyOp(Split));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Split";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Split), &PyOp(Split)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Split", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Split::typeinfo(), &py_type).second);
}

PyOpDefBegin(Subtensor) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(Subtensor)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"items", serialization<decltype(opdef.items)>::dump(opdef.items)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(Subtensor)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("items");
        if (iter != state.end()) {
            opdef.items = serialization<decltype(opdef.items)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(Subtensor)

int PyOp(Subtensor)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"items", "scope", NULL};
    PyObject *items = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &items, &scope))
    return -1;

    if (items) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(Subtensor)*>(self)->inst().items =
                    py::cast<decltype(Subtensor::items)>(py::handle(items));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(Subtensor)::py_getsetters[] = {
    {const_cast<char*>("items"), py_get_generic(Subtensor, items), py_set_generic(Subtensor, items), const_cast<char*>("items"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(Subtensor)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(Subtensor)::getstate, METH_NOARGS, "Subtensor getstate"},
    {const_cast<char*>("__setstate__"), PyOp(Subtensor)::setstate, METH_VARARGS, "Subtensor setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(Subtensor)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(Subtensor)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(Subtensor)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(Subtensor)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, items: list[tuple[int, bool, bool, bool, bool]] = ...) -> None\n"
};

void _init_py_Subtensor(py::module m) {
    using py_op = PyOp(Subtensor);
    auto& py_type = PyOpType(Subtensor);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.Subtensor";
    py_type.tp_basicsize = sizeof(PyOp(Subtensor));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "Subtensor";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(Subtensor), &PyOp(Subtensor)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("Subtensor", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(Subtensor::typeinfo(), &py_type).second);
}

PyOpDefBegin(TQT) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(TQT)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"qmin", serialization<decltype(opdef.qmin)>::dump(opdef.qmin)},
            {"qmax", serialization<decltype(opdef.qmax)>::dump(opdef.qmax)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(TQT)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("qmin");
        if (iter != state.end()) {
            opdef.qmin = serialization<decltype(opdef.qmin)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("qmax");
        if (iter != state.end()) {
            opdef.qmax = serialization<decltype(opdef.qmax)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(TQT)

int PyOp(TQT)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"qmin", "qmax", "scope", NULL};
    PyObject *qmin = NULL, *qmax = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &qmin, &qmax, &scope))
    return -1;

    if (qmin) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(TQT)*>(self)->inst().qmin =
                    py::cast<decltype(TQT::qmin)>(py::handle(qmin));
        } CATCH_ALL(-1)
    }

    if (qmax) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(TQT)*>(self)->inst().qmax =
                    py::cast<decltype(TQT::qmax)>(py::handle(qmax));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(TQT)::py_getsetters[] = {
    {const_cast<char*>("qmin"), py_get_generic(TQT, qmin), py_set_generic(TQT, qmin), const_cast<char*>("qmin"), NULL},
    {const_cast<char*>("qmax"), py_get_generic(TQT, qmax), py_set_generic(TQT, qmax), const_cast<char*>("qmax"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(TQT)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(TQT)::getstate, METH_NOARGS, "TQT getstate"},
    {const_cast<char*>("__setstate__"), PyOp(TQT)::setstate, METH_VARARGS, "TQT setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(TQT)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(TQT)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(TQT)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(TQT)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, qmin: int = ..., qmax: int = ...) -> None\n"
};

void _init_py_TQT(py::module m) {
    using py_op = PyOp(TQT);
    auto& py_type = PyOpType(TQT);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.TQT";
    py_type.tp_basicsize = sizeof(PyOp(TQT));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "TQT";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(TQT), &PyOp(TQT)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("TQT", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(TQT::typeinfo(), &py_type).second);
}

PyOpDefBegin(TensorRTRuntime) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(TensorRTRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"buf", serialization<decltype(opdef.buf)>::dump(opdef.buf)},
            {"buf_size", serialization<decltype(opdef.buf_size)>::dump(opdef.buf_size)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(TensorRTRuntime)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("buf");
        if (iter != state.end()) {
            opdef.buf = serialization<decltype(opdef.buf)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("buf_size");
        if (iter != state.end()) {
            opdef.buf_size = serialization<decltype(opdef.buf_size)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(TensorRTRuntime)

int PyOp(TensorRTRuntime)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"buf", "buf_size", "scope", NULL};
    PyObject *buf = NULL, *buf_size = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", const_cast<char**>(kwlist), &buf, &buf_size, &scope))
    return -1;

    if (buf) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(TensorRTRuntime)*>(self)->inst().buf =
                    py::cast<decltype(TensorRTRuntime::buf)>(py::handle(buf));
        } CATCH_ALL(-1)
    }

    if (buf_size) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(TensorRTRuntime)*>(self)->inst().buf_size =
                    py::cast<decltype(TensorRTRuntime::buf_size)>(py::handle(buf_size));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(TensorRTRuntime)::py_getsetters[] = {
    {const_cast<char*>("buf"), py_get_generic(TensorRTRuntime, buf), py_set_generic(TensorRTRuntime, buf), const_cast<char*>("buf"), NULL},
    {const_cast<char*>("buf_size"), py_get_generic(TensorRTRuntime, buf_size), py_set_generic(TensorRTRuntime, buf_size), const_cast<char*>("buf_size"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(TensorRTRuntime)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(TensorRTRuntime)::getstate, METH_NOARGS, "TensorRTRuntime getstate"},
    {const_cast<char*>("__setstate__"), PyOp(TensorRTRuntime)::setstate, METH_VARARGS, "TensorRTRuntime setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(TensorRTRuntime)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(TensorRTRuntime)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(TensorRTRuntime)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(TensorRTRuntime)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, buf: str = ..., buf_size: int = ...) -> None\n"
};

void _init_py_TensorRTRuntime(py::module m) {
    using py_op = PyOp(TensorRTRuntime);
    auto& py_type = PyOpType(TensorRTRuntime);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.TensorRTRuntime";
    py_type.tp_basicsize = sizeof(PyOp(TensorRTRuntime));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "TensorRTRuntime";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(TensorRTRuntime), &PyOp(TensorRTRuntime)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("TensorRTRuntime", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(TensorRTRuntime::typeinfo(), &py_type).second);
}

template<> struct EnumTrait<TopK::Mode> {
    static constexpr const char *name = "TopK.Mode";
    static constexpr std::underlying_type_t<TopK::Mode> max = 3 - 1;
};
template<> PyTypeObject* EnumWrapper<TopK::Mode>::type = nullptr;

template<> const char*
EnumWrapper<TopK::Mode>::members[] = {"KTH_ONLY", "VALUE_IDX_NOSORT", "VALUE_IDX_SORTED"};

template<> std::unordered_map<std::string, TopK::Mode>
EnumWrapper<TopK::Mode>::mem2value = {{normalize_enum("KTH_ONLY"), TopK::Mode::KTH_ONLY}, {normalize_enum("VALUE_IDX_NOSORT"), TopK::Mode::VALUE_IDX_NOSORT}, {normalize_enum("VALUE_IDX_SORTED"), TopK::Mode::VALUE_IDX_SORTED}};
template<> PyObject* EnumWrapper<TopK::Mode>::pyobj_insts[3] = {nullptr};

void _init_py_TopK_Mode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<TopK::Mode>::type;

    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)EnumWrapper<TopK::Mode>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)EnumWrapper<TopK::Mode>::py_repr},
        {Py_tp_richcompare, (void*)EnumWrapper<TopK::Mode>::tp_richcompare},
        {Py_tp_methods, tp_methods},

        {0, NULL}
    };
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.TopK.Mode",
        // basicsize
        sizeof(EnumWrapper<TopK::Mode>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__name__").release().ptr(),
                    py::cast("Mode").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__module__").release().ptr(),
                    py::cast("megengine.core._imperative_rt.ops").release().ptr()) >= 0);

    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("__qualname__").release().ptr(),
                    py::cast("TopK.Mode").release().ptr()) >= 0);
{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<TopK::Mode>*>(inst)->value = TopK::Mode::KTH_ONLY;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "KTH_ONLY", inst) >= 0);
    EnumWrapper<TopK::Mode>::pyobj_insts[0] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<TopK::Mode>*>(inst)->value = TopK::Mode::VALUE_IDX_NOSORT;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "VALUE_IDX_NOSORT", inst) >= 0);
    EnumWrapper<TopK::Mode>::pyobj_insts[1] = inst;
}{
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<EnumWrapper<TopK::Mode>*>(inst)->value = TopK::Mode::VALUE_IDX_SORTED;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "VALUE_IDX_SORTED", inst) >= 0);
    EnumWrapper<TopK::Mode>::pyobj_insts[2] = inst;
}
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Mode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(TopK) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(TopK)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"mode", serialization<decltype(opdef.mode)>::dump(opdef.mode)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(TopK)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("mode");
        if (iter != state.end()) {
            opdef.mode = serialization<decltype(opdef.mode)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(TopK)

int PyOp(TopK)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"mode", "scope", NULL};
    PyObject *mode = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &mode, &scope))
    return -1;

    if (mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(TopK)*>(self)->inst().mode =
                    py::cast<decltype(TopK::mode)>(py::handle(mode));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(TopK)::py_getsetters[] = {
    {const_cast<char*>("mode"), py_get_generic(TopK, mode), py_set_generic(TopK, mode), const_cast<char*>("mode"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(TopK)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(TopK)::getstate, METH_NOARGS, "TopK getstate"},
    {const_cast<char*>("__setstate__"), PyOp(TopK)::setstate, METH_VARARGS, "TopK setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(TopK)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(TopK)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(TopK)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(TopK)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, mode: Union[str, Mode] = ...) -> None\n"
};

void _init_py_TopK(py::module m) {
    using py_op = PyOp(TopK);
    auto& py_type = PyOpType(TopK);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.TopK";
    py_type.tp_basicsize = sizeof(PyOp(TopK));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "TopK";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(TopK), &PyOp(TopK)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_TopK_Mode(py_type);

    PyType_Modified(&py_type);
    m.add_object("TopK", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(TopK::typeinfo(), &py_type).second);
}

PyOpDefBegin(TypeCvt) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(TypeCvt)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(TypeCvt)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(TypeCvt)

int PyOp(TypeCvt)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"dtype", "scope", NULL};
    PyObject *dtype = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &dtype, &scope))
    return -1;

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(TypeCvt)*>(self)->inst().dtype =
                    py::cast<decltype(TypeCvt::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(TypeCvt)::py_getsetters[] = {
    {const_cast<char*>("dtype"), py_get_generic(TypeCvt, dtype), py_set_generic(TypeCvt, dtype), const_cast<char*>("dtype"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(TypeCvt)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(TypeCvt)::getstate, METH_NOARGS, "TypeCvt getstate"},
    {const_cast<char*>("__setstate__"), PyOp(TypeCvt)::setstate, METH_VARARGS, "TypeCvt setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(TypeCvt)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(TypeCvt)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(TypeCvt)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(TypeCvt)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, dtype: str = ...) -> None\n"
};

void _init_py_TypeCvt(py::module m) {
    using py_op = PyOp(TypeCvt);
    auto& py_type = PyOpType(TypeCvt);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.TypeCvt";
    py_type.tp_basicsize = sizeof(PyOp(TypeCvt));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "TypeCvt";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(TypeCvt), &PyOp(TypeCvt)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("TypeCvt", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(TypeCvt::typeinfo(), &py_type).second);
}

PyOpDefBegin(UniformRNG) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(UniformRNG)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"seed", serialization<decltype(opdef.seed)>::dump(opdef.seed)},
            {"dtype", serialization<decltype(opdef.dtype)>::dump(opdef.dtype)},
            {"handle", serialization<decltype(opdef.handle)>::dump(opdef.handle)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(UniformRNG)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("seed");
        if (iter != state.end()) {
            opdef.seed = serialization<decltype(opdef.seed)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("dtype");
        if (iter != state.end()) {
            opdef.dtype = serialization<decltype(opdef.dtype)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("handle");
        if (iter != state.end()) {
            opdef.handle = serialization<decltype(opdef.handle)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(UniformRNG)

int PyOp(UniformRNG)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"seed", "dtype", "handle", "scope", NULL};
    PyObject *seed = NULL, *dtype = NULL, *handle = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", const_cast<char**>(kwlist), &seed, &dtype, &handle, &scope))
    return -1;

    if (seed) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(UniformRNG)*>(self)->inst().seed =
                    py::cast<decltype(UniformRNG::seed)>(py::handle(seed));
        } CATCH_ALL(-1)
    }

    if (dtype) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(UniformRNG)*>(self)->inst().dtype =
                    py::cast<decltype(UniformRNG::dtype)>(py::handle(dtype));
        } CATCH_ALL(-1)
    }

    if (handle) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(UniformRNG)*>(self)->inst().handle =
                    py::cast<decltype(UniformRNG::handle)>(py::handle(handle));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(UniformRNG)::py_getsetters[] = {
    {const_cast<char*>("seed"), py_get_generic(UniformRNG, seed), py_set_generic(UniformRNG, seed), const_cast<char*>("seed"), NULL},
    {const_cast<char*>("dtype"), py_get_generic(UniformRNG, dtype), py_set_generic(UniformRNG, dtype), const_cast<char*>("dtype"), NULL},
    {const_cast<char*>("handle"), py_get_generic(UniformRNG, handle), py_set_generic(UniformRNG, handle), const_cast<char*>("handle"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(UniformRNG)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(UniformRNG)::getstate, METH_NOARGS, "UniformRNG getstate"},
    {const_cast<char*>("__setstate__"), PyOp(UniformRNG)::setstate, METH_VARARGS, "UniformRNG setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(UniformRNG)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(UniformRNG)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(UniformRNG)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(UniformRNG)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, seed: int = ..., dtype: str = ..., handle: int = ...) -> None\n"
};

void _init_py_UniformRNG(py::module m) {
    using py_op = PyOp(UniformRNG);
    auto& py_type = PyOpType(UniformRNG);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.UniformRNG";
    py_type.tp_basicsize = sizeof(PyOp(UniformRNG));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "UniformRNG";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(UniformRNG), &PyOp(UniformRNG)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
    
    PyType_Modified(&py_type);
    m.add_object("UniformRNG", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(UniformRNG::typeinfo(), &py_type).second);
}

void _init_py_WarpAffine_InterpolationMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpAffine::InterpolationMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "InterpolationMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpAffine_BorderMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpAffine::BorderMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "BorderMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpAffine_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpAffine::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(WarpAffine) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(WarpAffine)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"imode", serialization<decltype(opdef.imode)>::dump(opdef.imode)},
            {"border_mode", serialization<decltype(opdef.border_mode)>::dump(opdef.border_mode)},
            {"border_val", serialization<decltype(opdef.border_val)>::dump(opdef.border_val)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(WarpAffine)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("imode");
        if (iter != state.end()) {
            opdef.imode = serialization<decltype(opdef.imode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("border_mode");
        if (iter != state.end()) {
            opdef.border_mode = serialization<decltype(opdef.border_mode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("border_val");
        if (iter != state.end()) {
            opdef.border_val = serialization<decltype(opdef.border_val)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(WarpAffine)

int PyOp(WarpAffine)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"imode", "border_mode", "border_val", "format", "scope", NULL};
    PyObject *imode = NULL, *border_mode = NULL, *border_val = NULL, *format = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &imode, &border_mode, &border_val, &format, &scope))
    return -1;

    if (imode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpAffine)*>(self)->inst().imode =
                    py::cast<decltype(WarpAffine::imode)>(py::handle(imode));
        } CATCH_ALL(-1)
    }

    if (border_mode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpAffine)*>(self)->inst().border_mode =
                    py::cast<decltype(WarpAffine::border_mode)>(py::handle(border_mode));
        } CATCH_ALL(-1)
    }

    if (border_val) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpAffine)*>(self)->inst().border_val =
                    py::cast<decltype(WarpAffine::border_val)>(py::handle(border_val));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpAffine)*>(self)->inst().format =
                    py::cast<decltype(WarpAffine::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(WarpAffine)::py_getsetters[] = {
    {const_cast<char*>("imode"), py_get_generic(WarpAffine, imode), py_set_generic(WarpAffine, imode), const_cast<char*>("imode"), NULL},
    {const_cast<char*>("border_mode"), py_get_generic(WarpAffine, border_mode), py_set_generic(WarpAffine, border_mode), const_cast<char*>("border_mode"), NULL},
    {const_cast<char*>("border_val"), py_get_generic(WarpAffine, border_val), py_set_generic(WarpAffine, border_val), const_cast<char*>("border_val"), NULL},
    {const_cast<char*>("format"), py_get_generic(WarpAffine, format), py_set_generic(WarpAffine, format), const_cast<char*>("format"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(WarpAffine)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(WarpAffine)::getstate, METH_NOARGS, "WarpAffine getstate"},
    {const_cast<char*>("__setstate__"), PyOp(WarpAffine)::setstate, METH_VARARGS, "WarpAffine setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(WarpAffine)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(WarpAffine)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(WarpAffine)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(WarpAffine)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, imode: Union[str, InterpolationMode] = ..., border_mode: Union[str, BorderMode] = ..., border_val: float = ..., format: Union[str, Format] = ...) -> None\n"
};

void _init_py_WarpAffine(py::module m) {
    using py_op = PyOp(WarpAffine);
    auto& py_type = PyOpType(WarpAffine);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.WarpAffine";
    py_type.tp_basicsize = sizeof(PyOp(WarpAffine));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "WarpAffine";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(WarpAffine), &PyOp(WarpAffine)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_WarpAffine_InterpolationMode(py_type);
    _init_py_WarpAffine_BorderMode(py_type);
    _init_py_WarpAffine_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("WarpAffine", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(WarpAffine::typeinfo(), &py_type).second);
}

void _init_py_WarpPerspective_InterpolationMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspective::InterpolationMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "InterpolationMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpPerspective_BorderMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspective::BorderMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "BorderMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpPerspective_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspective::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(WarpPerspective) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(WarpPerspective)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"imode", serialization<decltype(opdef.imode)>::dump(opdef.imode)},
            {"bmode", serialization<decltype(opdef.bmode)>::dump(opdef.bmode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"border_val", serialization<decltype(opdef.border_val)>::dump(opdef.border_val)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(WarpPerspective)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("imode");
        if (iter != state.end()) {
            opdef.imode = serialization<decltype(opdef.imode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bmode");
        if (iter != state.end()) {
            opdef.bmode = serialization<decltype(opdef.bmode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("border_val");
        if (iter != state.end()) {
            opdef.border_val = serialization<decltype(opdef.border_val)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(WarpPerspective)

int PyOp(WarpPerspective)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"imode", "bmode", "format", "border_val", "scope", NULL};
    PyObject *imode = NULL, *bmode = NULL, *format = NULL, *border_val = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &imode, &bmode, &format, &border_val, &scope))
    return -1;

    if (imode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspective)*>(self)->inst().imode =
                    py::cast<decltype(WarpPerspective::imode)>(py::handle(imode));
        } CATCH_ALL(-1)
    }

    if (bmode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspective)*>(self)->inst().bmode =
                    py::cast<decltype(WarpPerspective::bmode)>(py::handle(bmode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspective)*>(self)->inst().format =
                    py::cast<decltype(WarpPerspective::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (border_val) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspective)*>(self)->inst().border_val =
                    py::cast<decltype(WarpPerspective::border_val)>(py::handle(border_val));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(WarpPerspective)::py_getsetters[] = {
    {const_cast<char*>("imode"), py_get_generic(WarpPerspective, imode), py_set_generic(WarpPerspective, imode), const_cast<char*>("imode"), NULL},
    {const_cast<char*>("bmode"), py_get_generic(WarpPerspective, bmode), py_set_generic(WarpPerspective, bmode), const_cast<char*>("bmode"), NULL},
    {const_cast<char*>("format"), py_get_generic(WarpPerspective, format), py_set_generic(WarpPerspective, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("border_val"), py_get_generic(WarpPerspective, border_val), py_set_generic(WarpPerspective, border_val), const_cast<char*>("border_val"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(WarpPerspective)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(WarpPerspective)::getstate, METH_NOARGS, "WarpPerspective getstate"},
    {const_cast<char*>("__setstate__"), PyOp(WarpPerspective)::setstate, METH_VARARGS, "WarpPerspective setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(WarpPerspective)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(WarpPerspective)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(WarpPerspective)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(WarpPerspective)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, imode: Union[str, InterpolationMode] = ..., bmode: Union[str, BorderMode] = ..., format: Union[str, Format] = ..., border_val: float = ...) -> None\n"
};

void _init_py_WarpPerspective(py::module m) {
    using py_op = PyOp(WarpPerspective);
    auto& py_type = PyOpType(WarpPerspective);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.WarpPerspective";
    py_type.tp_basicsize = sizeof(PyOp(WarpPerspective));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "WarpPerspective";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(WarpPerspective), &PyOp(WarpPerspective)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_WarpPerspective_InterpolationMode(py_type);
    _init_py_WarpPerspective_BorderMode(py_type);
    _init_py_WarpPerspective_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("WarpPerspective", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(WarpPerspective::typeinfo(), &py_type).second);
}

void _init_py_WarpPerspectiveBackwardData_InterpolationMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspectiveBackwardData::InterpolationMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "InterpolationMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpPerspectiveBackwardData_BorderMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspectiveBackwardData::BorderMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "BorderMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpPerspectiveBackwardData_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspectiveBackwardData::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(WarpPerspectiveBackwardData) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(WarpPerspectiveBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"imode", serialization<decltype(opdef.imode)>::dump(opdef.imode)},
            {"bmode", serialization<decltype(opdef.bmode)>::dump(opdef.bmode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"border_val", serialization<decltype(opdef.border_val)>::dump(opdef.border_val)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(WarpPerspectiveBackwardData)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("imode");
        if (iter != state.end()) {
            opdef.imode = serialization<decltype(opdef.imode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bmode");
        if (iter != state.end()) {
            opdef.bmode = serialization<decltype(opdef.bmode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("border_val");
        if (iter != state.end()) {
            opdef.border_val = serialization<decltype(opdef.border_val)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(WarpPerspectiveBackwardData)

int PyOp(WarpPerspectiveBackwardData)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"imode", "bmode", "format", "border_val", "scope", NULL};
    PyObject *imode = NULL, *bmode = NULL, *format = NULL, *border_val = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &imode, &bmode, &format, &border_val, &scope))
    return -1;

    if (imode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardData)*>(self)->inst().imode =
                    py::cast<decltype(WarpPerspectiveBackwardData::imode)>(py::handle(imode));
        } CATCH_ALL(-1)
    }

    if (bmode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardData)*>(self)->inst().bmode =
                    py::cast<decltype(WarpPerspectiveBackwardData::bmode)>(py::handle(bmode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardData)*>(self)->inst().format =
                    py::cast<decltype(WarpPerspectiveBackwardData::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (border_val) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardData)*>(self)->inst().border_val =
                    py::cast<decltype(WarpPerspectiveBackwardData::border_val)>(py::handle(border_val));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(WarpPerspectiveBackwardData)::py_getsetters[] = {
    {const_cast<char*>("imode"), py_get_generic(WarpPerspectiveBackwardData, imode), py_set_generic(WarpPerspectiveBackwardData, imode), const_cast<char*>("imode"), NULL},
    {const_cast<char*>("bmode"), py_get_generic(WarpPerspectiveBackwardData, bmode), py_set_generic(WarpPerspectiveBackwardData, bmode), const_cast<char*>("bmode"), NULL},
    {const_cast<char*>("format"), py_get_generic(WarpPerspectiveBackwardData, format), py_set_generic(WarpPerspectiveBackwardData, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("border_val"), py_get_generic(WarpPerspectiveBackwardData, border_val), py_set_generic(WarpPerspectiveBackwardData, border_val), const_cast<char*>("border_val"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(WarpPerspectiveBackwardData)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(WarpPerspectiveBackwardData)::getstate, METH_NOARGS, "WarpPerspectiveBackwardData getstate"},
    {const_cast<char*>("__setstate__"), PyOp(WarpPerspectiveBackwardData)::setstate, METH_VARARGS, "WarpPerspectiveBackwardData setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(WarpPerspectiveBackwardData)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(WarpPerspectiveBackwardData)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(WarpPerspectiveBackwardData)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(WarpPerspectiveBackwardData)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, imode: Union[str, InterpolationMode] = ..., bmode: Union[str, BorderMode] = ..., format: Union[str, Format] = ..., border_val: float = ...) -> None\n"
};

void _init_py_WarpPerspectiveBackwardData(py::module m) {
    using py_op = PyOp(WarpPerspectiveBackwardData);
    auto& py_type = PyOpType(WarpPerspectiveBackwardData);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.WarpPerspectiveBackwardData";
    py_type.tp_basicsize = sizeof(PyOp(WarpPerspectiveBackwardData));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "WarpPerspectiveBackwardData";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(WarpPerspectiveBackwardData), &PyOp(WarpPerspectiveBackwardData)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_WarpPerspectiveBackwardData_InterpolationMode(py_type);
    _init_py_WarpPerspectiveBackwardData_BorderMode(py_type);
    _init_py_WarpPerspectiveBackwardData_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("WarpPerspectiveBackwardData", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(WarpPerspectiveBackwardData::typeinfo(), &py_type).second);
}

void _init_py_WarpPerspectiveBackwardMat_InterpolationMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspectiveBackwardMat::InterpolationMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "InterpolationMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpPerspectiveBackwardMat_BorderMode(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspectiveBackwardMat::BorderMode>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "BorderMode", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

void _init_py_WarpPerspectiveBackwardMat_Format(PyTypeObject& py_type) {
    auto& e_type = EnumWrapper<WarpPerspectiveBackwardMat::Format>::type;

    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "Format", reinterpret_cast<PyObject*>(e_type)) >= 0);
}

PyOpDefBegin(WarpPerspectiveBackwardMat) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    
    static PyObject* getstate(PyObject* self, PyObject*) {
        auto& opdef = reinterpret_cast<PyOp(WarpPerspectiveBackwardMat)*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {
            
            {"imode", serialization<decltype(opdef.imode)>::dump(opdef.imode)},
            {"bmode", serialization<decltype(opdef.bmode)>::dump(opdef.bmode)},
            {"format", serialization<decltype(opdef.format)>::dump(opdef.format)},
            {"border_val", serialization<decltype(opdef.border_val)>::dump(opdef.border_val)}
        };
        return py::cast(state).release().ptr();
    }
    static PyObject* setstate(PyObject* self, PyObject* args) {
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp(WarpPerspectiveBackwardMat)*>(self)->inst();
        static_cast<void>(opdef);
        
        {
        auto&& iter = state.find("imode");
        if (iter != state.end()) {
            opdef.imode = serialization<decltype(opdef.imode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("bmode");
        if (iter != state.end()) {
            opdef.bmode = serialization<decltype(opdef.bmode)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("format");
        if (iter != state.end()) {
            opdef.format = serialization<decltype(opdef.format)>::load(iter->second);
        }
        }

        {
        auto&& iter = state.find("border_val");
        if (iter != state.end()) {
            opdef.border_val = serialization<decltype(opdef.border_val)>::load(iter->second);
        }
        }
        Py_RETURN_NONE;
    }
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
// };
PyOpDefEnd(WarpPerspectiveBackwardMat)

int PyOp(WarpPerspectiveBackwardMat)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static const char* kwlist[] = {"imode", "bmode", "format", "border_val", "scope", NULL};
    PyObject *imode = NULL, *bmode = NULL, *format = NULL, *border_val = NULL, *scope = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", const_cast<char**>(kwlist), &imode, &bmode, &format, &border_val, &scope))
    return -1;

    if (imode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardMat)*>(self)->inst().imode =
                    py::cast<decltype(WarpPerspectiveBackwardMat::imode)>(py::handle(imode));
        } CATCH_ALL(-1)
    }

    if (bmode) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardMat)*>(self)->inst().bmode =
                    py::cast<decltype(WarpPerspectiveBackwardMat::bmode)>(py::handle(bmode));
        } CATCH_ALL(-1)
    }

    if (format) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardMat)*>(self)->inst().format =
                    py::cast<decltype(WarpPerspectiveBackwardMat::format)>(py::handle(format));
        } CATCH_ALL(-1)
    }

    if (border_val) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp(WarpPerspectiveBackwardMat)*>(self)->inst().border_val =
                    py::cast<decltype(WarpPerspectiveBackwardMat::border_val)>(py::handle(border_val));
        } CATCH_ALL(-1)
    }

    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }

    return 0;
}

PyGetSetDef PyOp(WarpPerspectiveBackwardMat)::py_getsetters[] = {
    {const_cast<char*>("imode"), py_get_generic(WarpPerspectiveBackwardMat, imode), py_set_generic(WarpPerspectiveBackwardMat, imode), const_cast<char*>("imode"), NULL},
    {const_cast<char*>("bmode"), py_get_generic(WarpPerspectiveBackwardMat, bmode), py_set_generic(WarpPerspectiveBackwardMat, bmode), const_cast<char*>("bmode"), NULL},
    {const_cast<char*>("format"), py_get_generic(WarpPerspectiveBackwardMat, format), py_set_generic(WarpPerspectiveBackwardMat, format), const_cast<char*>("format"), NULL},
    {const_cast<char*>("border_val"), py_get_generic(WarpPerspectiveBackwardMat, border_val), py_set_generic(WarpPerspectiveBackwardMat, border_val), const_cast<char*>("border_val"), NULL},
    {NULL}  /* Sentinel */
};

    PyMethodDef PyOp(WarpPerspectiveBackwardMat)::tp_methods[] = {
        {const_cast<char*>("__getstate__"), PyOp(WarpPerspectiveBackwardMat)::getstate, METH_NOARGS, "WarpPerspectiveBackwardMat getstate"},
    {const_cast<char*>("__setstate__"), PyOp(WarpPerspectiveBackwardMat)::setstate, METH_VARARGS, "WarpPerspectiveBackwardMat setstate"},
        {NULL}  /* Sentinel */
    };
    
PyObject *PyOp(WarpPerspectiveBackwardMat)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp(WarpPerspectiveBackwardMat)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOp(WarpPerspectiveBackwardMat)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp(WarpPerspectiveBackwardMat)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "__init__(self, imode: Union[str, InterpolationMode] = ..., bmode: Union[str, BorderMode] = ..., format: Union[str, Format] = ..., border_val: float = ...) -> None\n"
};

void _init_py_WarpPerspectiveBackwardMat(py::module m) {
    using py_op = PyOp(WarpPerspectiveBackwardMat);
    auto& py_type = PyOpType(WarpPerspectiveBackwardMat);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.WarpPerspectiveBackwardMat";
    py_type.tp_basicsize = sizeof(PyOp(WarpPerspectiveBackwardMat));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "WarpPerspectiveBackwardMat";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType(WarpPerspectiveBackwardMat), &PyOp(WarpPerspectiveBackwardMat)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
    mgb_assert(PyType_Ready(&py_type) >= 0);
        _init_py_WarpPerspectiveBackwardMat_InterpolationMode(py_type);
    _init_py_WarpPerspectiveBackwardMat_BorderMode(py_type);
    _init_py_WarpPerspectiveBackwardMat_Format(py_type);

    PyType_Modified(&py_type);
    m.add_object("WarpPerspectiveBackwardMat", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(WarpPerspectiveBackwardMat::typeinfo(), &py_type).second);
}
#define INIT_ALL_OP(m) \
    _init_py_AdaptivePooling(m); \
    _init_py_AddAxis(m); \
    _init_py_Argmax(m); \
    _init_py_Argmin(m); \
    _init_py_Argsort(m); \
    _init_py_AssertEqual(m); \
    _init_py_AtlasRuntime(m); \
    _init_py_Barrier(m); \
    _init_py_BatchConvBias(m); \
    _init_py_BatchNorm(m); \
    _init_py_BatchNormBackward(m); \
    _init_py_BatchedIncrMeshIndexing(m); \
    _init_py_BatchedMatrixMul(m); \
    _init_py_BatchedMeshIndexing(m); \
    _init_py_BatchedSetMeshIndexing(m); \
    _init_py_BetaRNG(m); \
    _init_py_Borrow(m); \
    _init_py_Broadcast(m); \
    _init_py_CambriconRuntime(m); \
    _init_py_CheckNonFinite(m); \
    _init_py_CollectiveComm(m); \
    _init_py_Concat(m); \
    _init_py_CondTake(m); \
    _init_py_ConvBias(m); \
    _init_py_Convolution(m); \
    _init_py_Convolution3D(m); \
    _init_py_Convolution3DBackwardData(m); \
    _init_py_ConvolutionBackwardData(m); \
    _init_py_Copy(m); \
    _init_py_Correlation(m); \
    _init_py_Cumsum(m); \
    _init_py_CvtColor(m); \
    _init_py_DeformableConv(m); \
    _init_py_DeformablePSROIPooling(m); \
    _init_py_Diag(m); \
    _init_py_Dimshuffle(m); \
    _init_py_Dot(m); \
    _init_py_Dropout(m); \
    _init_py_Elemwise(m); \
    _init_py_ElemwiseMultiType(m); \
    _init_py_ExternOpr(m); \
    _init_py_Eye(m); \
    _init_py_FakeQuant(m); \
    _init_py_FastpathCopy(m); \
    _init_py_GammaRNG(m); \
    _init_py_GaussianRNG(m); \
    _init_py_GetVarShape(m); \
    _init_py_GroupLocal(m); \
    _init_py_GroupNorm(m); \
    _init_py_Identity(m); \
    _init_py_Images2Neibs(m); \
    _init_py_IncrMeshIndexing(m); \
    _init_py_IncrSubtensor(m); \
    _init_py_IndexingIncrMultiAxisVec(m); \
    _init_py_IndexingMultiAxisVec(m); \
    _init_py_IndexingOneHot(m); \
    _init_py_IndexingSetMultiAxisVec(m); \
    _init_py_IndexingSetOneHot(m); \
    _init_py_InplaceAdd(m); \
    _init_py_LAMBUpdate(m); \
    _init_py_LRN(m); \
    _init_py_LSQ(m); \
    _init_py_LSTM(m); \
    _init_py_LSTMCell(m); \
    _init_py_LayerNorm(m); \
    _init_py_Linspace(m); \
    _init_py_MagicMindRuntime(m); \
    _init_py_MatrixInverse(m); \
    _init_py_MatrixMul(m); \
    _init_py_MeshGrid(m); \
    _init_py_MeshIndexing(m); \
    _init_py_NMSKeep(m); \
    _init_py_NvOf(m); \
    _init_py_Padding(m); \
    _init_py_ParamPackConcat(m); \
    _init_py_ParamPackSplit(m); \
    _init_py_PermutationRNG(m); \
    _init_py_PixelShuffle(m); \
    _init_py_PixelShuffleBackward(m); \
    _init_py_PoissonRNG(m); \
    _init_py_Pooling(m); \
    _init_py_RNN(m); \
    _init_py_RNNCell(m); \
    _init_py_ROIAlign(m); \
    _init_py_ROIPooling(m); \
    _init_py_Reduce(m); \
    _init_py_RegionRestrictedConvolution(m); \
    _init_py_RegionRestrictedConvolutionBackwardData(m); \
    _init_py_Remap(m); \
    _init_py_RemoteRecv(m); \
    _init_py_RemoteSend(m); \
    _init_py_RemoveAxis(m); \
    _init_py_Reshape(m); \
    _init_py_Resize(m); \
    _init_py_SVD(m); \
    _init_py_SetMeshIndexing(m); \
    _init_py_SetSubtensor(m); \
    _init_py_ShuffleRNG(m); \
    _init_py_SlidingWindowTranspose(m); \
    _init_py_Softmax(m); \
    _init_py_Split(m); \
    _init_py_Subtensor(m); \
    _init_py_TQT(m); \
    _init_py_TensorRTRuntime(m); \
    _init_py_TopK(m); \
    _init_py_TypeCvt(m); \
    _init_py_UniformRNG(m); \
    _init_py_WarpAffine(m); \
    _init_py_WarpPerspective(m); \
    _init_py_WarpPerspectiveBackwardData(m); \
    _init_py_WarpPerspectiveBackwardMat(m);
// clang-format on
