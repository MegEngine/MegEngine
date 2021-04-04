import io

import numpy as np

import megengine.core.tensor.megbrain_graph as G
import megengine.utils.comp_graph_tools as cgtools
from megengine import tensor
from megengine.jit import trace
from megengine.utils.network_node import VarNode


def _default_compare_fn(x, y):
    if isinstance(x, np.ndarray):
        np.testing.assert_allclose(x, y, rtol=1e-6)
    else:
        np.testing.assert_allclose(x.numpy(), y, rtol=1e-6)


def make_tensor(x, network=None, device=None):
    if network is not None:
        if isinstance(x, VarNode):
            return VarNode(x.var)
        return network.make_const(x, device=device)
    else:
        return tensor(x, device=device)


def opr_test(
    cases,
    func,
    compare_fn=_default_compare_fn,
    ref_fn=None,
    test_trace=True,
    network=None,
    **kwargs
):
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
            if not isinstance(r, (tensor, VarNode)):
                r = tensor(r)
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
    inp_tensor = [make_tensor(inpi, network) for inpi in inp]

    if test_trace and not network:
        copied_inp = inp_tensor.copy()
        for symbolic in [False, True]:
            traced_func = trace(symbolic=symbolic)(func)

            for _ in range(3):
                traced_results = traced_func(*copied_inp, **kwargs)
            check_results(traced_results, outp)

        dumped_func = trace(symbolic=True, capture_as_const=True)(func)
        dumped_results = dumped_func(*copied_inp, **kwargs)
        check_results(dumped_results, outp)

        file = io.BytesIO()
        dump_info = dumped_func.dump(file)
        file.seek(0)

        # arg_name has pattern arg_xxx, xxx is int value
        def take_number(arg_name):
            return int(arg_name.split("_")[-1])

        input_names = dump_info[4]
        inps_np = [i.numpy() for i in copied_inp]
        input_names.sort(key=take_number)
        inp_dict = dict(zip(input_names, inps_np))
        infer_cg = cgtools.GraphInference(file)

        # assume #outputs == 1
        loaded_results = list(infer_cg.run(inp_dict=inp_dict).values())[0]
        check_results(loaded_results, outp)

    results = func(*inp_tensor, **kwargs)
    check_results(results, outp)
