module @gemm {
  func private @extern_kernel(memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>)
  %0 = AIE.tile(25, 2) {polyaie.leaf}
  %1 = AIE.lock(%0, 15)
  %2 = AIE.buffer(%0) {polyaie.iter_num_buf, sym_name = "buf0"} : memref<1xi32>
  %3 = AIE.buffer(%0) {sym_name = "buf1"} : memref<2x2xf32>
  %4 = AIE.buffer(%0) {sym_name = "buf2"} : memref<2x2xf32>
  %5 = AIE.buffer(%0) {sym_name = "buf3"} : memref<2x2xf32>
  %6 = AIE.buffer(%0) {sym_name = "buf4"} : memref<2x2xf32>
  %7 = AIE.core(%0) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %8 = memref.load %2[%c0] : memref<1xi32>
    %9 = arith.index_cast %8 : i32 to index
    scf.for %arg0 = %c0 to %9 step %c1 {
      %c2 = arith.constant 2 : index
      %10 = arith.remui %arg0, %c2 : index
      %c0_0 = arith.constant 0 : index
      %11 = arith.cmpi eq, %10, %c0_0 : index
      scf.if %11 {
        call @extern_kernel(%6, %3, %4, %5) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
      } else {
        call @extern_kernel(%6, %3, %4, %5) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
      }
    }
    AIE.useLock(%1, Release, 1) {polyaie.runtime}
    AIE.end
  } {link_with = "kernel.o"}
}

