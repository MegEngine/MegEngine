from copy import deepcopy
from typing import List, Set

from ...logger import get_logger
from ..traced_module import TracedModule
from .pass_base import get_default_pass_context, get_registered_pass

logger = get_logger(__name__)


def optimize(
    module: TracedModule, enabled_pass: List[str] = ["FuseConvBn"],
) -> TracedModule:
    r"""Performs a set of optimization passes to optimize a `TracedModule` for inference.

    The following passes are currently supported:

        * FuseConvBn: fuse BN layers into to conv2d
        * FuseAddMul: fold adjacent const add or mul binary operations
        * BackwardFoldScale: backward fold const scaling into weights of conv2d

    Args:
        module: the :class:`TracedModule` to be optimized.
        enabled_pass: optimization passes to be enabled during optimization.
            Default: ["FuseConvBn"]

    Returns:
        the optimized :class:`TracedModule`.
    """

    defalut_passes_list = [
        "FuseConvBn",
        "FuseAddMul",
    ]

    if isinstance(enabled_pass, str):
        enabled_pass = [enabled_pass]

    if "BackwardFoldScale" in enabled_pass:
        if "FuseConvBn" not in enabled_pass:
            logger.warning(
                "Since BackwardFoldScale requires FuseConvBn"
                ", FuseConvBn will be enabled."
            )
            enabled_pass.append("FuseConvBn")
        defalut_passes_list.extend(
            ["BackwardFoldScale", "FuseAddMul",]
        )

    pass_ctx = get_default_pass_context()

    def run_pass(mod: TracedModule):
        for pass_name in defalut_passes_list:
            if pass_name in enabled_pass:
                pass_func = get_registered_pass(pass_name)()
                mod = pass_func(mod, pass_ctx)
        return mod

    module = deepcopy(module)
    module = run_pass(module)

    return module
