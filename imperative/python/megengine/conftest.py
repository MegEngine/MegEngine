import pytest

import megengine


@pytest.fixture(autouse=True)
def import_megengine_path(doctest_namespace):
    doctest_namespace["mge"] = megengine
    doctest_namespace["Tensor"] = megengine.Tensor
    doctest_namespace["F"] = megengine.functional
    doctest_namespace["M"] = megengine.module
    doctest_namespace["Q"] = megengine.quantization
    doctest_namespace["data"] = megengine.data
    doctest_namespace["autodiff"] = megengine.autodiff
    doctest_namespace["optim"] = megengine.optimizer
    doctest_namespace["jit"] = megengine.jit
    doctest_namespace["amp"] = megengine.amp
    doctest_namespace["dist"] = megengine.distributed
    doctest_namespace["tm"] = megengine.traced_module
    doctest_namespace["hub"] = megengine.hub
    doctest_namespace["utils"] = megengine.utils
