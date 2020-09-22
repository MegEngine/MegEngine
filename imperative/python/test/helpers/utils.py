import numpy as np

from megengine import tensor


def _default_compare_fn(x, y):
    np.testing.assert_allclose(x.numpy(), y, rtol=1e-6)


def opr_test(cases, func, compare_fn=_default_compare_fn, ref_fn=None, **kwargs):
    """
    :param cases: the list which have dict element, the list length should be 2 for dynamic shape test.
           and the dict should have input,
           and should have output if ref_fn is None.
           should use list for multiple inputs and outputs for each case.
    :param func: the function to run opr.
    :param compare_fn: the function to compare the result and expected, use
        ``np.testing.assert_allclose`` if None.
    :param ref_fn: the function to generate expected data, should assign output if None.

    Examples:

    .. code-block::

        dtype = np.float32
        cases = [{"input": [10, 20]}, {"input": [20, 30]}]
        opr_test(cases,
                 F.eye,
                 ref_fn=lambda n, m: np.eye(n, m).astype(dtype),
                 dtype=dtype)

    """

    def check_results(results, expected):
        if not isinstance(results, (tuple, list)):
            results = (results,)
        for r, e in zip(results, expected):
            compare_fn(r, e)

    def get_param(cases, idx):
        case = cases[idx]
        inp = case.get("input", None)
        outp = case.get("output", None)
        if inp is None:
            raise ValueError("the test case should have input")
        if not isinstance(inp, (tuple, list)):
            inp = (inp,)
        if ref_fn is not None and callable(ref_fn):
            outp = ref_fn(*inp)
        if outp is None:
            raise ValueError("the test case should have output or reference function")
        if not isinstance(outp, (tuple, list)):
            outp = (outp,)

        return inp, outp

    if len(cases) == 0:
        raise ValueError("should give one case at least")

    if not callable(func):
        raise ValueError("the input func should be callable")

    inp, outp = get_param(cases, 0)
    inp_tensor = [tensor(inpi) for inpi in inp]

    results = func(*inp_tensor, **kwargs)
    check_results(results, outp)
