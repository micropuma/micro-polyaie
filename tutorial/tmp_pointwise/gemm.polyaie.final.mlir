module @pointwise_mult {
  %0 = aie.tile(25, 2) {polyaie.leaf}
  %1 = aie.lock(%0, 15) {sym_name = "lock_25_2_15"}
  %2 = aie.buffer(%0) {sym_name = "buf0"} : memref<32xf32>
  %3 = aie.buffer(%0) {sym_name = "buf1"} : memref<32xf32>
  %4 = aie.buffer(%0) {sym_name = "buf2"} : memref<32xf32>
  %5 = aie.buffer(%0) {sym_name = "buf3"} : memref<32xf32>
  %6 = aie.core(%0) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %7 = aievec.upd %5[%arg0] {index = 0 : i8, offset = 0 : si32} : memref<32xf32>, vector<8xf32>
      vector.transfer_write %7, %2[%arg0] {in_bounds = [true]} : vector<8xf32>, memref<32xf32>
    }
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %7 = aievec.upd %3[%arg0] {index = 0 : i8, offset = 0 : si32} : memref<32xf32>, vector<8xf32>
      %8 = aievec.upd %4[%arg0] {index = 0 : i8, offset = 0 : si32} : memref<32xf32>, vector<8xf32>
      %9 = aievec.concat %7, %7 : vector<8xf32>, vector<16xf32>
      %10 = aievec.mul %9, %8 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x76543210", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
      vector.transfer_write %10, %2[%arg0] {in_bounds = [true]} : vector<8xf32>, memref<32xf32>
    }
    aie.use_lock(%1, Release, 1) {polyaie.runtime}
    aie.end
  }
}

