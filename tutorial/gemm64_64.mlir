module @gemm {
  %0 = aie.tile(23, 2)
  %6 = aie.tile(24, 2) {polyaie.leaf}
  %13 = aie.tile(24, 3)
  %19 = aie.tile(24, 4) {polyaie.leaf}
  %26 = aie.tile(25, 4)
  %32 = aie.tile(26, 4) {polyaie.leaf}
  %39 = aie.tile(25, 3)
  %46 = aie.tile(26, 3) {polyaie.leaf}

  %1 = aie.buffer(%6) {sym_name = "buf0"} : memref<32x32xf32>
  %2 = aie.buffer(%0) {sym_name = "buf1"} : memref<32x32xf32>
  %3 = aie.buffer(%0) {sym_name = "buf2"} : memref<32x32xf32>
  %4 = aie.buffer(%0) {sym_name = "buf3"} : memref<32x32xf32>

  %7 = aie.lock(%6, 15) {sym_name = "lock_24_2_15"}
  %8 = aie.lock(%6, 0)
  %9 = aie.buffer(%6) {sym_name = "buf4"} : memref<32x32xf32>
  %10 = aie.buffer(%6) {sym_name = "buf5"} : memref<32x32xf32>
  %11 = aie.buffer(%6) {sym_name = "buf6"} : memref<32x32xf32>

  %14 = aie.buffer(%13) {sym_name = "buf7"} : memref<32x32xf32>
  %15 = aie.buffer(%13) {sym_name = "buf8"} : memref<32x32xf32>
  %16 = aie.buffer(%13) {sym_name = "buf9"} : memref<32x32xf32>
  %17 = aie.buffer(%13) {sym_name = "buf10"} : memref<32x32xf32>

  %20 = aie.lock(%19, 15) {sym_name = "lock_24_4_15"}
  %21 = aie.lock(%19, 0)
  %22 = aie.buffer(%19) {sym_name = "buf11"} : memref<32x32xf32>
  %23 = aie.buffer(%19) {sym_name = "buf12"} : memref<32x32xf32>
  %24 = aie.buffer(%19) {sym_name = "buf13"} : memref<32x32xf32>

  %27 = aie.buffer(%32) {sym_name = "buf14"} : memref<32x32xf32>
  %28 = aie.buffer(%26) {sym_name = "buf15"} : memref<32x32xf32>
  %29 = aie.buffer(%26) {sym_name = "buf16"} : memref<32x32xf32>
  %30 = aie.buffer(%26) {sym_name = "buf17"} : memref<32x32xf32>

  %33 = aie.lock(%32, 15) {sym_name = "lock_26_4_15"}
  %34 = aie.lock(%32, 0)
  %35 = aie.buffer(%32) {sym_name = "buf18"} : memref<32x32xf32>
  %36 = aie.buffer(%32) {sym_name = "buf19"} : memref<32x32xf32>
  %37 = aie.buffer(%32) {sym_name = "buf20"} : memref<32x32xf32>

  %40 = aie.lock(%39, 0)
  %41 = aie.buffer(%39) {sym_name = "buf21"} : memref<32x32xf32>
  %42 = aie.buffer(%39) {sym_name = "buf22"} : memref<32x32xf32>
  %43 = aie.buffer(%39) {sym_name = "buf23"} : memref<32x32xf32>
  %44 = aie.buffer(%39) {sym_name = "buf24"} : memref<32x32xf32>

  %47 = aie.lock(%46, 15) {sym_name = "lock_26_3_15"}
  %48 = aie.buffer(%46) {sym_name = "buf25"} : memref<32x32xf32>
  %49 = aie.buffer(%46) {sym_name = "buf26"} : memref<32x32xf32>
  %50 = aie.buffer(%46) {sym_name = "buf27"} : memref<32x32xf32>

  %5 = aie.core(%0) {
    aie.use_lock(%8, Acquire, 0)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %4[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %1[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %2[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %3[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %1[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %1[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%8, Release, 1)
    aie.end
  }
  
  %12 = aie.core(%6) {
    aie.use_lock(%8, Acquire, 1)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %1[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %9[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %10[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %11[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %9[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %9[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%8, Release, 0)
    aie.use_lock(%7, Release, 1) {polyaie.runtime}
    aie.end
  }

  %18 = aie.core(%13) {
    aie.use_lock(%21, Acquire, 0)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %17[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %14[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %15[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %16[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %14[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %14[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%21, Release, 1)
    aie.end
  }

  %25 = aie.core(%19) {
    aie.use_lock(%21, Acquire, 1)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %14[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %22[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %23[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %24[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %22[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %22[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%21, Release, 0)
    aie.use_lock(%20, Release, 1) {polyaie.runtime}
    aie.end
  }


  %31 = aie.core(%26) {
    aie.use_lock(%34, Acquire, 0)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %30[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %27[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %28[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %29[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %27[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %27[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%34, Release, 1)
    aie.end
  }

  %38 = aie.core(%32) {
    aie.use_lock(%34, Acquire, 1)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %27[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %35[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %36[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %37[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %35[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %35[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%34, Release, 0)
    aie.use_lock(%33, Release, 1) {polyaie.runtime}
    aie.end
  }

  %45 = aie.core(%39) {
    aie.use_lock(%40, Acquire, 0)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %44[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %41[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %42[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %43[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %41[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %41[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%40, Release, 1)
    aie.end
  }

  %51 = aie.core(%46) {
    aie.use_lock(%40, Acquire, 1)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %52 = affine.load %41[%arg0, %arg1] : memref<32x32xf32>
        affine.store %52, %48[%arg0, %arg1] : memref<32x32xf32>
        affine.for %arg2 = 0 to 32 {
          %53 = affine.load %49[%arg0, %arg2] : memref<32x32xf32>
          %54 = affine.load %50[%arg2, %arg1] : memref<32x32xf32>
          %55 = arith.mulf %53, %54 : f32
          %56 = affine.load %48[%arg0, %arg1] : memref<32x32xf32>
          %57 = arith.addf %56, %55 : f32
          affine.store %57, %48[%arg0, %arg1] : memref<32x32xf32>
        }
      }
    }
    aie.use_lock(%40, Release, 0)
    aie.use_lock(%47, Release, 1) {polyaie.runtime}
    aie.end
  }
}
