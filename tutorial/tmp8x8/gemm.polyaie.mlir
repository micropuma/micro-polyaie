module @gemm {
  %0 = memref.alloc() : memref<16x16xf32>
  %1 = memref.alloc() : memref<16x16xf32>
  %2 = memref.alloc() : memref<16x16xf32>
  %3 = aie.tile(24, 2) {polyaie.leaf}
  %4 = aie.lock(%3, 15) {sym_name = "lock_24_2_15"}
  %5 = aie.buffer(%3) {sym_name = "buf0"} : memref<16x16xf32>
  "dataflow.runtime.host_dma"(%0, %5) {kind = 3 : i32, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : (memref<16x16xf32>, memref<16x16xf32>) -> ()
  %6 = aie.buffer(%3) {sym_name = "buf1"} : memref<16x16xf32>
  "dataflow.runtime.host_dma"(%6, %1) {kind = 1 : i32, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : (memref<16x16xf32>, memref<16x16xf32>) -> ()
  %7 = aie.buffer(%3) {sym_name = "buf2"} : memref<16x16xf32>
  "dataflow.runtime.host_dma"(%7, %2) {kind = 1 : i32, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : (memref<16x16xf32>, memref<16x16xf32>) -> ()
  %8 = aie.buffer(%3) {sym_name = "buf3"} : memref<16x16xf32>
  "dataflow.runtime.host_dma"(%8, %0) {kind = 1 : i32, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : (memref<16x16xf32>, memref<16x16xf32>) -> ()
  %9 = aie.core(%3) {
    affine.for %arg0 = 0 to 16 {
      affine.for %arg1 = 0 to 16 {
        %10 = affine.load %8[%arg0, %arg1] : memref<16x16xf32>
        affine.store %10, %5[%arg0, %arg1] : memref<16x16xf32>
        affine.for %arg2 = 0 to 16 {
          %11 = affine.load %6[%arg0, %arg2] : memref<16x16xf32>
          %12 = affine.load %7[%arg2, %arg1] : memref<16x16xf32>
          %13 = arith.mulf %11, %12 : f32
          %14 = affine.load %5[%arg0, %arg1] : memref<16x16xf32>
          %15 = arith.addf %14, %13 : f32
          affine.store %15, %5[%arg0, %arg1] : memref<16x16xf32>
        }
      }
    }
    aie.use_lock(%4, Release, 1) {polyaie.runtime}
    aie.end
  }
}

