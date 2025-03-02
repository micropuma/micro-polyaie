module @gemm {
  %0 = aie.tile(25, 4)
  %1 = aie.tile(25, 3) {polyaie.leaf}
  %2 = aie.tile(24, 2)
  %3 = aie.tile(24, 3) {polyaie.leaf}
  %4 = aie.tile(26, 3)
  %5 = aie.tile(26, 4) {polyaie.leaf}
  %6 = aie.tile(23, 3)
  %7 = aie.tile(23, 2) {polyaie.leaf}
  %8 = aie.lock(%1, 15) {sym_name = "lock_25_3_15"}
  %9 = aie.lock(%1, 0)
  %10 = aie.lock(%3, 15) {sym_name = "lock_24_3_15"}
  %11 = aie.lock(%3, 0)
  %12 = aie.lock(%5, 15) {sym_name = "lock_26_4_15"}
  %13 = aie.lock(%5, 0)
  %14 = aie.lock(%7, 15) {sym_name = "lock_23_2_15"}
  %15 = aie.lock(%7, 0)
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
    aie.use_lock(%9, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %19[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %16[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %17[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %18[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %16[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %16[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%9, Release, 1)
    aie.end
  }
  %45 = aie.core(%1) {
    aie.use_lock(%9, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %16[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %20[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %21[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %22[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %20[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %20[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%9, Release, 0)
    aie.use_lock(%8, Release, 1) {polyaie.runtime}
    aie.end
  }
  %46 = aie.core(%2) {
    aie.use_lock(%11, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %26[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %23[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %24[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %25[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %23[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %23[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%11, Release, 1)
    aie.end
  }
  %47 = aie.core(%3) {
    aie.use_lock(%11, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %23[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %27[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %28[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %29[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %27[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %27[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%11, Release, 0)
    aie.use_lock(%10, Release, 1) {polyaie.runtime}
    aie.end
  }
  %48 = aie.core(%4) {
    aie.use_lock(%13, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %33[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %30[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %31[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %32[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %30[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %30[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%13, Release, 1)
    aie.end
  }
  %49 = aie.core(%5) {
    aie.use_lock(%13, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %30[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %34[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %35[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %36[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %34[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %34[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%13, Release, 0)
    aie.use_lock(%12, Release, 1) {polyaie.runtime}
    aie.end
  }
  %50 = aie.core(%6) {
    aie.use_lock(%15, Acquire, 0)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %40[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %37[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %38[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %39[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %37[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %37[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%15, Release, 1)
    aie.end
  }
  %51 = aie.core(%7) {
    aie.use_lock(%15, Acquire, 1)
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %52 = affine.load %37[%arg0, %arg1] : memref<2x2xf32>
        affine.store %52, %41[%arg0, %arg1] : memref<2x2xf32>
        affine.for %arg2 = 0 to 2 {
          %53 = affine.load %42[%arg0, %arg2] : memref<2x2xf32>
          %54 = affine.load %43[%arg2, %arg1] : memref<2x2xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %41[%arg0, %arg1] : memref<2x2xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %41[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    aie.use_lock(%15, Release, 0)
    aie.use_lock(%14, Release, 1) {polyaie.runtime}
    aie.end
  }
}

