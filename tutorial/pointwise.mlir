func.func @pointwise_mult (%A: memref<32xf32>, %B: memref<32xf32>, %C: memref<32xf32>) {
    affine.for %arg0 = 0 to 32 {
       %a = affine.load %A[%arg0] : memref<32xf32>
       %b = affine.load %B[%arg0] : memref<32xf32>
       %c = arith.mulf %a, %b : f32
       affine.store %c, %C[%arg0] : memref<32xf32>
    }
    return
}