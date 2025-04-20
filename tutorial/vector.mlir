// -----// IR Dump After AffineVectorize //----- //
func.func @PE0_AIE1(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>) -> tensor<32xf32> {
  %0 = bufferization.to_memref %arg2 : memref<32xf32>
  %1 = bufferization.to_memref %arg1 : memref<32xf32>
  %2 = bufferization.to_memref %arg0 : memref<32xf32>
  %3 = memref.alloc() : memref<32xf32>
  affine.for %arg3 = 0 to 32 step 8 {
    %cst = arith.constant 0.000000e+00 : f32
    %5 = vector.transfer_read %0[%arg3], %cst : memref<32xf32>, vector<8xf32>
    vector.transfer_write %5, %3[%arg3] : vector<8xf32>, memref<32xf32>
  }
  affine.for %arg3 = 0 to 32 step 8 {
    %cst = arith.constant 0.000000e+00 : f32
    %5 = vector.transfer_read %2[%arg3], %cst : memref<32xf32>, vector<8xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %6 = vector.transfer_read %1[%arg3], %cst_0 : memref<32xf32>, vector<8xf32>
    %7 = arith.mulf %5, %6 : vector<8xf32>
    vector.transfer_write %7, %3[%arg3] : vector<8xf32>, memref<32xf32>
  }
  %4 = bufferization.to_tensor %3 : memref<32xf32>
  return %4 : tensor<32xf32>
}

// -----// IR Dump After AffineVectorize //----- //
func.func @PE0_AIE0(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>) -> tensor<32xf32> {
  %0 = bufferization.to_memref %arg2 : memref<32xf32>
  %1 = bufferization.to_memref %arg1 : memref<32xf32>
  %2 = bufferization.to_memref %arg0 : memref<32xf32>
  %3 = memref.alloc() : memref<32xf32>
  affine.for %arg3 = 0 to 32 step 8 {
    %cst = arith.constant 0.000000e+00 : f32
    %5 = vector.transfer_read %0[%arg3], %cst : memref<32xf32>, vector<8xf32>
    vector.transfer_write %5, %3[%arg3] : vector<8xf32>, memref<32xf32>
  }
  affine.for %arg3 = 0 to 32 step 8 {
    %cst = arith.constant 0.000000e+00 : f32
    %5 = vector.transfer_read %2[%arg3], %cst : memref<32xf32>, vector<8xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %6 = vector.transfer_read %1[%arg3], %cst_0 : memref<32xf32>, vector<8xf32>
    %7 = arith.mulf %5, %6 : vector<8xf32>
    vector.transfer_write %7, %3[%arg3] : vector<8xf32>, memref<32xf32>
  }
  %4 = bufferization.to_tensor %3 : memref<32xf32>
  return %4 : tensor<32xf32>
}


