module @gemm {
  %0 = aie.tile(24, 2) {polyaie.leaf}
  %1 = aie.lock(%0, 15) {sym_name = "lock_24_2_15"}
  %2 = aie.buffer(%0) {sym_name = "buf0"} : memref<2x2xf32>
  %3 = aie.buffer(%0) {sym_name = "buf1"} : memref<2x2xf32>
  %4 = aie.buffer(%0) {sym_name = "buf2"} : memref<2x2xf32>
  %5 = aie.buffer(%0) {sym_name = "buf3"} : memref<2x2xf32>
  %6 = aie.core(%0) {
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %7 = affine.load %5[%arg0, %arg1] : memref<2x2xf32>
        affine.store %7, %2[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %8 = affine.load %3[%arg0, %arg2] : memref<2x2xf32>
          %9 = affine.load %4[%arg2, %arg1] : memref<2x2xf32>
          %10 = arith.mulf %8, %9 : f32
          %11 = affine.load %2[%arg0, %arg1] : memref<2x2xf32>
          %12 = arith.addf %11, %10 : f32
          affine.store %12, %2[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%1, Release, 1) {polyaie.runtime}
    aie.end
  }
}

