import jaxlib.mlir.dialects.builtin as builtin
import jaxlib.mlir.dialects.chlo as chlo
import jaxlib.mlir.dialects.func as func
import jaxlib.mlir.dialects.mhlo as mhlo
import jaxlib.mlir.dialects.ml_program as ml_program
import jaxlib.mlir.dialects.sparse_tensor as sparse_tensor
import jaxlib.mlir.dialects.stablehlo as stablehlo
import jaxlib.xla_client as xla_client

# Alias that is set up to abstract away the transition from MHLO to StableHLO.
use_stablehlo = xla_client.mlir_api_version >= 42
if use_stablehlo:
    import jaxlib.mlir.dialects.stablehlo as hlo
else:
    import jaxlib.mlir.dialects.mhlo as hlo  # type: ignore[no-redef]
