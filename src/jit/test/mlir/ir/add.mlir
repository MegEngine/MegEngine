// RUN: mgb-opt --mgb-convert-to-affine --split-input-file -canonicalize -cse %s | mgb-file-check %s
// RUN: mgb-opt --mgb-convert-to-affine --mgb-codegen-convert-affine-to-llvm --split-input-file -canonicalize -cse %s

func @add_dim1(%lhs: memref<2xf32>, %rhs: memref<2xf32>, %res: memref<2xf32>) -> () {
    %0 = "mgb.Elemwise"(%lhs, %rhs) {name = "add.f", mode = 16 : i32} :
       (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    "mgb.assign"(%0, %res) : (memref<2xf32>, memref<2xf32>) -> ()
    mgb.return 
}
// CHECK-LABEL: func @add_dim1(%arg0: memref<2xf32>, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
// CHECK:   %0 = alloc() : memref<2xf32>
// CHECK:   affine.for %arg3 = 0 to 2 {
// CHECK:     %1 = affine.load %arg0[%arg3] : memref<2xf32>
// CHECK:     %2 = affine.load %arg1[%arg3] : memref<2xf32>
// CHECK:     %3 = addf %1, %2 : f32
// CHECK:     affine.store %3, %0[%arg3] : memref<2xf32>
// CHECK:   }
// CHECK:   affine.for %arg3 = 0 to 2 {
// CHECK:     %1 = affine.load %0[%arg3] : memref<2xf32>
// CHECK:     affine.store %1, %arg2[%arg3] : memref<2xf32>
// CHECK:   }
// CHECK:   dealloc %0 : memref<2xf32>
// CHECK:   return
// CHECK: }

func @add_dim4(%lhs: memref<4x3x64x64xf32>, %rhs: memref<4x3x64x64xf32>, %res: memref<4x3x64x64xf32>) -> () {
    %0 = "mgb.Elemwise"(%lhs, %rhs) {name = "add.f", mode = 16 : i32} :
       (memref<4x3x64x64xf32>, memref<4x3x64x64xf32>) -> memref<4x3x64x64xf32>
    "mgb.assign"(%0, %res) : (memref<4x3x64x64xf32>, memref<4x3x64x64xf32>) -> ()
    mgb.return
}
// CHECK-LABEL: func @add_dim4(%arg0: memref<4x3x64x64xf32>, %arg1: memref<4x3x64x64xf32>, %arg2: memref<4x3x64x64xf32>) {
// CHECK:   %0 = alloc() : memref<4x3x64x64xf32>
// CHECK:   affine.for %arg3 = 0 to 4 {
// CHECK:     affine.for %arg4 = 0 to 3 {
// CHECK:       affine.for %arg5 = 0 to 64 {
// CHECK:         affine.for %arg6 = 0 to 64 {
// CHECK:           %1 = affine.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<4x3x64x64xf32>
// CHECK:           %2 = affine.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<4x3x64x64xf32>
// CHECK:           %3 = addf %1, %2 : f32
// CHECK:           affine.store %3, %0[%arg3, %arg4, %arg5, %arg6] : memref<4x3x64x64xf32>
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK:   affine.for %arg3 = 0 to 4 {
// CHECK:     affine.for %arg4 = 0 to 3 {
// CHECK:       affine.for %arg5 = 0 to 64 {
// CHECK:         affine.for %arg6 = 0 to 64 {
// CHECK:           %1 = affine.load %0[%arg3, %arg4, %arg5, %arg6] : memref<4x3x64x64xf32>
// CHECK:           affine.store %1, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<4x3x64x64xf32>
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK:   dealloc %0 : memref<4x3x64x64xf32>
// CHECK:   return
// CHECK: }
