module {
  func @gemm(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 4 {
          %0 = affine.load %arg1[%arg3, %arg5] : memref<4x4xf32>
          %1 = affine.load %arg2[%arg5, %arg4] : memref<4x4xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg0[%arg3, %arg4] : memref<4x4xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg0[%arg3, %arg4] : memref<4x4xf32>
        }
      }
    }
    return
  }
}
