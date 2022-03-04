#pragma once

namespace mgb::imperative::python {

PyObject* make_shape_tuple(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* getitem_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* setitem_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* split_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* expand_dims_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* squeeze_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* transpose_cpp(PyObject* self, PyObject* const* args, size_t nargs);

}  // namespace mgb::imperative::python