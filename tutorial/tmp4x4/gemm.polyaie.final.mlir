module @gemm {
  func private @extern_kernel(memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>)
  %0 = aie.tile(24, 3)
  %1 = aie.tile(25, 3) {polyaie.leaf}
  %2 = aie.tile(26, 3)
  %3 = aie.tile(26, 2) {polyaie.leaf}
  %4 = aie.tile(26, 4)
  %5 = aie.tile(25, 4) {polyaie.leaf}
  %6 = aie.tile(24, 4)
  %7 = aie.tile(23, 4) {polyaie.leaf}
  %8 = aie.lock(%0, 0)
  %9 = aie.lock(%1, 15) {sym_name = "lock_25_3_15"}
  %10 = aie.lock(%3, 15) {sym_name = "lock_26_2_15"}
  %11 = aie.lock(%3, 0)
  %12 = aie.lock(%4, 0)
  %13 = aie.lock(%5, 15) {sym_name = "lock_25_4_15"}
  %14 = aie.lock(%6, 0)
  %15 = aie.lock(%7, 15) {sym_name = "lock_23_4_15"}
  %16 = aie.buffer(%0) {sym_name = "buf0"} : memref<2x2xf32>
  %17 = aie.buffer(%0) {sym_name = "buf1"} : memref<2x2xf32>
  %18 = aie.buffer(%0) {sym_name = "buf2"} : memref<2x2xf32>
  %19 = aie.buffer(%0) {sym_name = "buf3"} : memref<2x2xf32>
  %20 = aie.buffer(%1) {sym_name = "buf4"} : memref<2x2xf32>
  %21 = aie.buffer(%1) {sym_name = "buf5"} : memref<2x2xf32>
  %22 = aie.buffer(%1) {sym_name = "buf6"} : memref<2x2xf32>
  %23 = aie.buffer(%2) {sym_name = "buf7"} : memref<2x2xf32>
  %24 = aie.buffer(%2) {sym_name = "buf8"} : memref<2x2xf32>
  %25 = aie.buffer(%2) {sym_name = "buf9"} : memref<2x2xf32>
  %26 = aie.buffer(%2) {sym_name = "buf10"} : memref<2x2xf32>
  %27 = aie.buffer(%3) {sym_name = "buf11"} : memref<2x2xf32>
  %28 = aie.buffer(%3) {sym_name = "buf12"} : memref<2x2xf32>
  %29 = aie.buffer(%3) {sym_name = "buf13"} : memref<2x2xf32>
  %30 = aie.buffer(%4) {sym_name = "buf14"} : memref<2x2xf32>
  %31 = aie.buffer(%4) {sym_name = "buf15"} : memref<2x2xf32>
  %32 = aie.buffer(%4) {sym_name = "buf16"} : memref<2x2xf32>
  %33 = aie.buffer(%4) {sym_name = "buf17"} : memref<2x2xf32>
  %34 = aie.buffer(%5) {sym_name = "buf18"} : memref<2x2xf32>
  %35 = aie.buffer(%5) {sym_name = "buf19"} : memref<2x2xf32>
  %36 = aie.buffer(%5) {sym_name = "buf20"} : memref<2x2xf32>
  %37 = aie.buffer(%6) {sym_name = "buf21"} : memref<2x2xf32>
  %38 = aie.buffer(%6) {sym_name = "buf22"} : memref<2x2xf32>
  %39 = aie.buffer(%6) {sym_name = "buf23"} : memref<2x2xf32>
  %40 = aie.buffer(%6) {sym_name = "buf24"} : memref<2x2xf32>
  %41 = aie.buffer(%7) {sym_name = "buf25"} : memref<2x2xf32>
  %42 = aie.buffer(%7) {sym_name = "buf26"} : memref<2x2xf32>
  %43 = aie.buffer(%7) {sym_name = "buf27"} : memref<2x2xf32>
  %44 = aie.core(%0) {
    aie.use_lock(%8, Acquire, 0)
    call @extern_kernel(%19, %16, %17, %18) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%8, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %45 = aie.core(%1) {
    aie.use_lock(%8, Acquire, 1)
    call @extern_kernel(%16, %20, %21, %22) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%8, Release, 0)
    aie.use_lock(%9, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %46 = aie.core(%2) {
    aie.use_lock(%11, Acquire, 0)
    call @extern_kernel(%26, %23, %24, %25) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%11, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %47 = aie.core(%3) {
    aie.use_lock(%11, Acquire, 1)
    call @extern_kernel(%23, %27, %28, %29) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%11, Release, 0)
    aie.use_lock(%10, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %48 = aie.core(%4) {
    aie.use_lock(%12, Acquire, 0)
    call @extern_kernel(%33, %30, %31, %32) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%12, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %49 = aie.core(%5) {
    aie.use_lock(%12, Acquire, 1)
    call @extern_kernel(%30, %34, %35, %36) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%12, Release, 0)
    aie.use_lock(%13, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %50 = aie.core(%6) {
    aie.use_lock(%14, Acquire, 0)
    call @extern_kernel(%40, %37, %38, %39) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%14, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %51 = aie.core(%7) {
    aie.use_lock(%14, Acquire, 1)
    call @extern_kernel(%37, %41, %42, %43) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%14, Release, 0)
    aie.use_lock(%15, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
}

