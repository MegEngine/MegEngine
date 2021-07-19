load("list.bzl", "cutlass_gen_list")

genrule(
    name = "cutlass_kimpls",
    outs = cutlass_gen_list,
    cmd = """GEN=$(location //brain/megbrain/dnn/scripts/cutlass_generator:generator.py)
          python3 $$GEN --operations gemm --type simt $(@D)
          python3 $$GEN --operations gemv --type simt $(@D)
          python3 $$GEN --operations deconv --type simt $(@D)
          python3 $$GEN --operations conv2d --type simt $(@D)
          python3 $$GEN --operations conv2d --type tensorop8816 $(@D)
          python3 $$GEN --operations conv2d --type tensorop8832 $(@D)
          """,
    tools = ["//brain/megbrain/dnn/scripts/cutlass_generator:generator.py"],
    visibility = ["//visibility:public"],
)
