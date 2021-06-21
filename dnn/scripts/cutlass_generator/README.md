# Generate device kernel registration code for CUTLASS kernels
## Usage
```bash
python3 generator.py [--operations {gemm, gemv, conv2d, deconv}] [--type {simt, tensorop8816, tensorop8832}]
                     output
```
- operations: operation kind, including gemm|gemv|conv2d|deconv
- type: opcode class, simt|tensorop8816|tensorop8832
- output: the output directory for CUTLASS kernels

## Generate file list for bazel

We generate `list.bzl` because the `genrule` method of bazel requires that the output file list be specified in the analysis phase.

Please call `gen_list.py` when new operations are added.

```bash
python3 gen_list.py
```
