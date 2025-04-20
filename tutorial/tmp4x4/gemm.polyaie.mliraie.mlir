module @gemm {
  func private @extern_kernel(memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>)
  %0 = aie.tile(24, 3)
  %1 = aie.lock(%0, 0)
  %2 = aie.buffer(%0) {sym_name = "buf0"} : memref<2x2xf32>
  %3 = aie.buffer(%0) {sym_name = "buf1"} : memref<2x2xf32>
  %4 = aie.buffer(%0) {sym_name = "buf2"} : memref<2x2xf32>
  %5 = aie.buffer(%0) {sym_name = "buf3"} : memref<2x2xf32>
  %6 = aie.core(%0) {
    aie.use_lock(%1, Acquire, 0)
    call @extern_kernel(%5, %2, %3, %4) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%1, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %7 = aie.tile(25, 3) {polyaie.leaf}
  %8 = aie.lock(%7, 15) {sym_name = "lock_25_3_15"}
  %9 = aie.buffer(%7) {sym_name = "buf4"} : memref<2x2xf32>
  %10 = aie.buffer(%7) {sym_name = "buf5"} : memref<2x2xf32>
  %11 = aie.buffer(%7) {sym_name = "buf6"} : memref<2x2xf32>
  %12 = aie.core(%7) {
    aie.use_lock(%1, Acquire, 1)
    call @extern_kernel(%2, %9, %10, %11) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%1, Release, 0)
    aie.use_lock(%8, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %13 = aie.tile(26, 3)
  %14 = aie.buffer(%13) {sym_name = "buf7"} : memref<2x2xf32>
  %15 = aie.buffer(%13) {sym_name = "buf8"} : memref<2x2xf32>
  %16 = aie.buffer(%13) {sym_name = "buf9"} : memref<2x2xf32>
  %17 = aie.buffer(%13) {sym_name = "buf10"} : memref<2x2xf32>
  %18 = aie.core(%13) {
    aie.use_lock(%21, Acquire, 0)
    call @extern_kernel(%17, %14, %15, %16) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%21, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %19 = aie.tile(26, 2) {polyaie.leaf}
  %20 = aie.lock(%19, 15) {sym_name = "lock_26_2_15"}
  %21 = aie.lock(%19, 0)
  %22 = aie.buffer(%19) {sym_name = "buf11"} : memref<2x2xf32>
  %23 = aie.buffer(%19) {sym_name = "buf12"} : memref<2x2xf32>
  %24 = aie.buffer(%19) {sym_name = "buf13"} : memref<2x2xf32>
  %25 = aie.core(%19) {
    aie.use_lock(%21, Acquire, 1)
    call @extern_kernel(%14, %22, %23, %24) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%21, Release, 0)
    aie.use_lock(%20, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %26 = aie.tile(26, 4)
  %27 = aie.lock(%26, 0)
  %28 = aie.buffer(%26) {sym_name = "buf14"} : memref<2x2xf32>
  %29 = aie.buffer(%26) {sym_name = "buf15"} : memref<2x2xf32>
  %30 = aie.buffer(%26) {sym_name = "buf16"} : memref<2x2xf32>
  %31 = aie.buffer(%26) {sym_name = "buf17"} : memref<2x2xf32>
  %32 = aie.core(%26) {
    aie.use_lock(%27, Acquire, 0)
    call @extern_kernel(%31, %28, %29, %30) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%27, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %33 = aie.tile(25, 4) {polyaie.leaf}
  %34 = aie.lock(%33, 15) {sym_name = "lock_25_4_15"}
  %35 = aie.buffer(%33) {sym_name = "buf18"} : memref<2x2xf32>
  %36 = aie.buffer(%33) {sym_name = "buf19"} : memref<2x2xf32>
  %37 = aie.buffer(%33) {sym_name = "buf20"} : memref<2x2xf32>
  %38 = aie.core(%33) {
    aie.use_lock(%27, Acquire, 1)
    call @extern_kernel(%28, %35, %36, %37) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%27, Release, 0)
    aie.use_lock(%34, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
  %39 = aie.tile(24, 4)
  %40 = aie.lock(%39, 0)
  %41 = aie.buffer(%39) {sym_name = "buf21"} : memref<2x2xf32>
  %42 = aie.buffer(%39) {sym_name = "buf22"} : memref<2x2xf32>
  %43 = aie.buffer(%39) {sym_name = "buf23"} : memref<2x2xf32>
  %44 = aie.buffer(%39) {sym_name = "buf24"} : memref<2x2xf32>
  %45 = aie.core(%39) {
    aie.use_lock(%40, Acquire, 0)
    call @extern_kernel(%44, %41, %42, %43) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%40, Release, 1)
    aie.end
  } {link_with = "kernel.o"}
  %46 = aie.tile(23, 4) {polyaie.leaf}
  %47 = aie.lock(%46, 15) {sym_name = "lock_23_4_15"}
  %48 = aie.buffer(%46) {sym_name = "buf25"} : memref<2x2xf32>
  %49 = aie.buffer(%46) {sym_name = "buf26"} : memref<2x2xf32>
  %50 = aie.buffer(%46) {sym_name = "buf27"} : memref<2x2xf32>
  %51 = aie.core(%46) {
    aie.use_lock(%40, Acquire, 1)
    call @extern_kernel(%41, %48, %49, %50) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    aie.use_lock(%40, Release, 0)
    aie.use_lock(%47, Release, 1) {polyaie.runtime}
    aie.end
  } {link_with = "kernel.o"}
}

