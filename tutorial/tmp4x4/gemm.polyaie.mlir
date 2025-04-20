module @gemm {
  func private @extern_kernel(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>) {
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 2 {
        %55 = affine.load %arg0[%arg4, %arg5] : memref<2x2xf32>
        affine.store %55, %arg1[%arg4, %arg5] : memref<2x2xf32>
        affine.for %arg6 = 0 to 2 {
          %56 = affine.load %arg2[%arg4, %arg6] : memref<2x2xf32>
          %57 = affine.load %arg3[%arg6, %arg5] : memref<2x2xf32>
          %58 = arith.mulf %56, %57 : f32
          %59 = affine.load %arg1[%arg4, %arg5] : memref<2x2xf32>
          %60 = arith.addf %59, %58 : f32
          affine.store %60, %arg1[%arg4, %arg5] : memref<2x2xf32>
        }
      }
    }
    return
  }
  %0 = memref.alloc() : memref<4x4xf32>
  %1 = memref.alloc() : memref<4x4xf32>
  %2 = memref.alloc() : memref<4x4xf32>
  %3 = aie.tile(24, 3)
  %4 = aie.lock(%3, 0)
  %5 = aie.buffer(%3) {sym_name = "buf0"} : memref<2x2xf32>
  %6 = aie.buffer(%3) {sym_name = "buf1"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%6, %1) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %7 = aie.buffer(%3) {sym_name = "buf2"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%7, %2) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %8 = aie.buffer(%3) {sym_name = "buf3"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%8, %0) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %9 = aie.core(%3) {
    aie.use_lock(%4, Acquire, 0)
    call @extern_kernel(%8, %5, %6, %7) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%4, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %10 = aie.tile(25, 3) {polyaie.leaf}
  %11 = aie.lock(%10, 15) {sym_name = "lock_25_3_15"}
  %12 = aie.buffer(%10) {sym_name = "buf4"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %12) {kind = 3 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %13 = aie.buffer(%10) {sym_name = "buf5"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%13, %1) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %14 = aie.buffer(%10) {sym_name = "buf6"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%14, %2) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %15 = aie.core(%10) {
    aie.use_lock(%4, Acquire, 1)
    call @extern_kernel(%5, %12, %13, %14) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%4, Release, 0)
    aie.use_lock(%11, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %16 = aie.tile(26, 3)
  %17 = aie.buffer(%16) {sym_name = "buf7"} : memref<2x2xf32>
  %18 = aie.buffer(%16) {sym_name = "buf8"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%18, %1) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %19 = aie.buffer(%16) {sym_name = "buf9"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%19, %2) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %20 = aie.buffer(%16) {sym_name = "buf10"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%20, %0) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %21 = aie.core(%16) {
    aie.use_lock(%24, Acquire, 0)
    call @extern_kernel(%20, %17, %18, %19) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%24, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %22 = aie.tile(26, 2) {polyaie.leaf}
  %23 = aie.lock(%22, 15) {sym_name = "lock_26_2_15"}
  %24 = aie.lock(%22, 0)
  %25 = aie.buffer(%22) {sym_name = "buf11"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %25) {kind = 3 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %26 = aie.buffer(%22) {sym_name = "buf12"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%26, %1) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %27 = aie.buffer(%22) {sym_name = "buf13"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%27, %2) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %28 = aie.core(%22) {
    aie.use_lock(%24, Acquire, 1)
    call @extern_kernel(%17, %25, %26, %27) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%24, Release, 0)
    aie.use_lock(%23, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %29 = aie.tile(26, 4)
  %30 = aie.lock(%29, 0)
  %31 = aie.buffer(%29) {sym_name = "buf14"} : memref<2x2xf32>
  %32 = aie.buffer(%29) {sym_name = "buf15"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%32, %1) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %33 = aie.buffer(%29) {sym_name = "buf16"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%33, %2) {kind = 1 : i32, offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %34 = aie.buffer(%29) {sym_name = "buf17"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%34, %0) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %35 = aie.core(%29) {
    aie.use_lock(%30, Acquire, 0)
    call @extern_kernel(%34, %31, %32, %33) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%30, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %36 = aie.tile(25, 4) {polyaie.leaf}
  %37 = aie.lock(%36, 15) {sym_name = "lock_25_4_15"}
  %38 = aie.buffer(%36) {sym_name = "buf18"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %38) {kind = 3 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %39 = aie.buffer(%36) {sym_name = "buf19"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%39, %1) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %40 = aie.buffer(%36) {sym_name = "buf20"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%40, %2) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %41 = aie.core(%36) {
    aie.use_lock(%30, Acquire, 1)
    call @extern_kernel(%31, %38, %39, %40) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%30, Release, 0)
    aie.use_lock(%37, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %42 = aie.tile(24, 4)
  %43 = aie.lock(%42, 0)
  %44 = aie.buffer(%42) {sym_name = "buf21"} : memref<2x2xf32>
  %45 = aie.buffer(%42) {sym_name = "buf22"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%45, %1) {kind = 1 : i32, offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %46 = aie.buffer(%42) {sym_name = "buf23"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%46, %2) {kind = 1 : i32, offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %47 = aie.buffer(%42) {sym_name = "buf24"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%47, %0) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %48 = aie.core(%42) {
    aie.use_lock(%43, Acquire, 0)
    call @extern_kernel(%47, %44, %45, %46) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%43, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %49 = aie.tile(23, 4) {polyaie.leaf}
  %50 = aie.lock(%49, 15) {sym_name = "lock_23_4_15"}
  %51 = aie.buffer(%49) {sym_name = "buf25"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%0, %51) {kind = 3 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<4x4xf32>, memref<2x2xf32>) -> ()
  %52 = aie.buffer(%49) {sym_name = "buf26"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%52, %1) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %53 = aie.buffer(%49) {sym_name = "buf27"} : memref<2x2xf32>
  "dataflow.runtime.host_dma"(%53, %2) {kind = 1 : i32, offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, memref<4x4xf32>) -> ()
  %54 = aie.core(%49) {
    aie.use_lock(%43, Acquire, 1)
    call @extern_kernel(%44, %51, %52, %53) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%43, Release, 0)
    aie.use_lock(%50, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
}

