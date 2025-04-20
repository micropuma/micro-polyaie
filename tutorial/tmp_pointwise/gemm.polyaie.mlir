module @pointwise_mult {
  %0 = memref.alloc() : memref<32xf32>
  %1 = memref.alloc() : memref<32xf32>
  %2 = memref.alloc() : memref<32xf32>
  %3 = aie.tile(25, 2) {polyaie.leaf}
  %4 = aie.lock(%3, 15) {sym_name = "lock_25_2_15"}
  %5 = aie.buffer(%3) {sym_name = "buf0"} : memref<32xf32>
  "dataflow.runtime.host_dma"(%2, %5) {kind = 3 : i32, offsets = [0], sizes = [32], strides = [1]} : (memref<32xf32>, memref<32xf32>) -> ()
  %6 = aie.buffer(%3) {sym_name = "buf1"} : memref<32xf32>
  "dataflow.runtime.host_dma"(%6, %0) {kind = 1 : i32, offsets = [0], sizes = [32], strides = [1]} : (memref<32xf32>, memref<32xf32>) -> ()
  %7 = aie.buffer(%3) {sym_name = "buf2"} : memref<32xf32>
  "dataflow.runtime.host_dma"(%7, %1) {kind = 1 : i32, offsets = [0], sizes = [32], strides = [1]} : (memref<32xf32>, memref<32xf32>) -> ()
  %8 = aie.buffer(%3) {sym_name = "buf3"} : memref<32xf32>
  "dataflow.runtime.host_dma"(%8, %2) {kind = 1 : i32, offsets = [0], sizes = [32], strides = [1]} : (memref<32xf32>, memref<32xf32>) -> ()
  %9 = aie.core(%3) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %10 = aievec.upd %8[%arg0] {index = 0 : i8, offset = 0 : si32} : memref<32xf32>, vector<8xf32>
      vector.transfer_write %10, %5[%arg0] {in_bounds = [true]} : vector<8xf32>, memref<32xf32>
    }
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %10 = aievec.upd %6[%arg0] {index = 0 : i8, offset = 0 : si32} : memref<32xf32>, vector<8xf32>
      %11 = aievec.upd %7[%arg0] {index = 0 : i8, offset = 0 : si32} : memref<32xf32>, vector<8xf32>
      %12 = aievec.concat %10, %10 : vector<8xf32>, vector<16xf32>
      %13 = aievec.mul %12, %11 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      vector.transfer_write %13, %5[%arg0] {in_bounds = [true]} : vector<8xf32>, memref<32xf32>
    }
    aie.use_lock(%4, Release, 1) {polyaie.runtime}
    aie.end
  }
}

