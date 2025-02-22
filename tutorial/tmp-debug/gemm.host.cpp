
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for MLIR-AIE host kernel.
//
//===----------------------------------------------------------------------===//

#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096

#include "aie_inc.cpp"

void gemm(
  float arg0[2][2],
  float arg1[2][2],
  float arg2[2][2],
  unsigned iter_num = 1) {

  printf("Configure AIE array...\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);


  printf("Initialize buffers...\n");

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(fd != -1 && "memory is not available");

  mlir_aie_clear_tile_memory(_xaie, 24, 2);

  unsigned bufIdx;

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf1(_xaie, bufIdx++, arg1[idx0][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf2(_xaie, bufIdx++, arg2[idx0][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf3(_xaie, bufIdx++, arg0[idx0][idx1]);


  bool results[1];

  for (auto &result : results)
    result = false;

  auto kernel_complete = [&]() {
    bool flag = true;
    for (auto result : results) {
      flag &= result;
      // printf("%d ", result);
    }
    // printf("\n");
    return flag;
  };


  printf("Start cores...\n");
  mlir_aie_start_cores(_xaie);


  while(!kernel_complete()) {
    if (mlir_aie_acquire_lock(_xaie, 24, 2, 15, 1, 0))
      results[0] = true;
  }

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      arg0[idx0][idx1] = mlir_aie_read_buffer_buf0(_xaie, bufIdx++);



  mlir_aie_deinit_libxaie(_xaie);

  printf("Complete compute.\n");
}
