module @gemm {
  %0 = memref.alloc() : memref<4x4xf32>
  %1 = memref.alloc() : memref<4x4xf32>
  %2 = memref.alloc() : memref<4x4xf32>
  %3 = aie.tile(24, 3)
  %4 = aie.buffer(%3) {sym_name = "buf0"} : memref<2x2xf32>
  %5 = aie.buffer(%3) {sym_name = "buf1"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%5, %1) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %6 = aie.buffer(%3) {sym_name = "buf2"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%6, %2) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %7 = aie.buffer(%3) {sym_name = "buf3"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%7, %0) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %8 = aie.core(%3) {
    aie.use_lock(%11, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %7[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %4[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %5[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %6[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %4[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %4[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%11, Release, 1)
    aie.end
  }
  %9 = aie.tile(24, 2) {polyaie.leaf}
  %10 = aie.lock(%9, 15) {sym_name = "lock_24_2_15"}
  %11 = aie.lock(%9, 0)
  %12 = aie.buffer(%9) {sym_name = "buf4"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %12) {kind = 3 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %13 = aie.buffer(%9) {sym_name = "buf5"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%13, %1) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %14 = aie.buffer(%9) {sym_name = "buf6"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%14, %2) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %15 = aie.core(%9) {
    aie.use_lock(%11, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %4[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %12[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %13[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %14[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %12[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %12[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%11, Release, 0)
    aie.use_lock(%10, Release, 1) {polyaie.runtime}
    aie.end
  }
  %16 = aie.tile(25, 3)
  %17 = aie.buffer(%16) {sym_name = "buf7"} : memref<2x2xf32>
  %18 = aie.buffer(%16) {sym_name = "buf8"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%18, %1) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %19 = aie.buffer(%16) {sym_name = "buf9"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%19, %2) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %20 = aie.buffer(%16) {sym_name = "buf10"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%20, %0) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %21 = aie.core(%16) {
    aie.use_lock(%24, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %20[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %17[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %18[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %19[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %17[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %17[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%24, Release, 1)
    aie.end
  }
  %22 = aie.tile(25, 4) {polyaie.leaf}
  %23 = aie.lock(%22, 15) {sym_name = "lock_25_4_15"}
  %24 = aie.lock(%22, 0)
  %25 = aie.buffer(%22) {sym_name = "buf11"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %25) {kind = 3 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %26 = aie.buffer(%22) {sym_name = "buf12"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%26, %1) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %27 = aie.buffer(%22) {sym_name = "buf13"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%27, %2) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %28 = aie.core(%22) {
    aie.use_lock(%24, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %17[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %25[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %26[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %27[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %25[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %25[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%24, Release, 0)
    aie.use_lock(%23, Release, 1) {polyaie.runtime}
    aie.end
  }
  %29 = aie.tile(23, 4)
  %30 = aie.buffer(%35) {sym_name = "buf14"} : memref<2x2xf32>
  %31 = aie.buffer(%29) {sym_name = "buf15"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%31, %1) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %32 = aie.buffer(%29) {sym_name = "buf16"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%32, %2) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %33 = aie.buffer(%29) {sym_name = "buf17"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%33, %0) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %34 = aie.core(%29) {
    aie.use_lock(%37, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %33[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %30[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %31[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %32[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %30[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %30[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%37, Release, 1)
    aie.end
  }
  %35 = aie.tile(24, 4) {polyaie.leaf}
  %36 = aie.lock(%35, 15) {sym_name = "lock_24_4_15"}
  %37 = aie.lock(%35, 0)
  %38 = aie.buffer(%35) {sym_name = "buf18"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %38) {kind = 3 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %39 = aie.buffer(%35) {sym_name = "buf19"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%39, %1) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %40 = aie.buffer(%35) {sym_name = "buf20"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%40, %2) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %41 = aie.core(%35) {
    aie.use_lock(%37, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %30[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %38[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %39[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %40[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %38[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %38[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%37, Release, 0)
    aie.use_lock(%36, Release, 1) {polyaie.runtime}
    aie.end
  }
  %42 = aie.tile(26, 2)
  %43 = aie.buffer(%42) {sym_name = "buf21"} : memref<2x2xf32>
  %44 = aie.buffer(%42) {sym_name = "buf22"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%44, %1) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %45 = aie.buffer(%42) {sym_name = "buf23"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%45, %2) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %46 = aie.buffer(%42) {sym_name = "buf24"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%46, %0) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %47 = aie.core(%42) {
    aie.use_lock(%50, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %46[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %43[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %44[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %45[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %43[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %43[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%50, Release, 1)
    aie.end
  }
  %48 = aie.tile(26, 3) {polyaie.leaf}
  %49 = aie.lock(%48, 15) {sym_name = "lock_26_3_15"}
  %50 = aie.lock(%48, 0)
  %51 = aie.buffer(%48) {sym_name = "buf25"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %51) {kind = 3 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %52 = aie.buffer(%48) {sym_name = "buf26"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%52, %1) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %53 = aie.buffer(%48) {sym_name = "buf27"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%53, %2) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %54 = aie.core(%48) {
    aie.use_lock(%50, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %55 = affine.load %43[%arg0, %arg1] : memref<2x2xf32>
        affine.store %55, %51[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %56 = affine.load %52[%arg0, %arg2] : memref<2x2xf32>
          %57 = affine.load %53[%arg2, %arg1] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %51[%arg0, %arg1] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %51[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%50, Release, 0)
    aie.use_lock(%49, Release, 1) {polyaie.runtime}
    aie.end
  }
}

