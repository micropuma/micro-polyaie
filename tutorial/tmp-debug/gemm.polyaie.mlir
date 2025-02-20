module @gemm {
  func private @extern_kernel(memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>)
  %0 = memref.alloc() : memref<2x2xf32>
  %1 = memref.alloc() : memref<2x2xf32>
  %2 = memref.alloc() : memref<2x2xf32>
  %3 = AIE.tile(25, 2) {polyaie.leaf}
  %4 = AIE.lock(%3, 15)
  %5 = AIE.buffer(%3) {polyaie.iter_num_buf, sym_name = "buf0"} : memref<1xi32>
  %6 = AIE.buffer(%3) {sym_name = "buf1"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %6) {kind = 3 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  %7 = AIE.buffer(%3) {sym_name = "buf2"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%7, %1) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  %8 = AIE.buffer(%3) {sym_name = "buf3"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%8, %2) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  %9 = AIE.buffer(%3) {sym_name = "buf4"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%9, %0) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  %10 = AIE.core(%3) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %11 = memref.load %5[%c0] : memref<1xi32>
    %12 = arith.index_cast %11 : i32 to index
    scf.for %arg0 = %c0 to %12 step %c1 {
      %c2 = arith.constant 2 : index
      %13 = arith.remui %arg0, %c2 : index
      %c0_0 = arith.constant 0 : index
      %14 = arith.cmpi eq, %13, %c0_0 : index
      scf.if %14 {
        call @extern_kernel(%9, %6, %7, %8) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
      } else {
        call @extern_kernel(%9, %6, %7, %8) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
      }
    }
    AIE.useLock(%4, Release, 1) {polyaie.runtime}
    AIE.end
  } {link_with = "kernel.o"}
}

