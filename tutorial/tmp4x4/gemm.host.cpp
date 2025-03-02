
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

int main() {
  float arg0[4][4];
  for (int64_t idx0 = 0; idx0 < 4; ++idx0) {
    for (int64_t idx1 = 0; idx1 < 4; ++idx1) {
      arg0[idx0][idx1] = 0.0;
    }
  }
  float arg1[4][4];
  for (int64_t idx0 = 0; idx0 < 4; ++idx0) {
    for (int64_t idx1 = 0; idx1 < 4; ++idx1) {
      arg1[idx0][idx1] = 0.0;
    }
  }
  float arg2[4][4];
  for (int64_t idx0 = 0; idx0 < 4; ++idx0) {
    for (int64_t idx1 = 0; idx1 < 4; ++idx1) {
      arg2[idx0][idx1] = 0.0;
    }
  }

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

  mlir_aie_clear_tile_memory(_xaie, 25, 4);
  mlir_aie_clear_tile_memory(_xaie, 25, 3);
  mlir_aie_clear_tile_memory(_xaie, 24, 2);
  mlir_aie_clear_tile_memory(_xaie, 24, 3);
  mlir_aie_clear_tile_memory(_xaie, 26, 3);
  mlir_aie_clear_tile_memory(_xaie, 26, 4);
  mlir_aie_clear_tile_memory(_xaie, 23, 3);
  mlir_aie_clear_tile_memory(_xaie, 23, 2);

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

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf5(_xaie, bufIdx++, arg1[idx0][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf6(_xaie, bufIdx++, arg2[idx0 + 2][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf8(_xaie, bufIdx++, arg1[idx0][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf9(_xaie, bufIdx++, arg2[idx0][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf10(_xaie, bufIdx++, arg0[idx0][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf12(_xaie, bufIdx++, arg1[idx0][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf13(_xaie, bufIdx++, arg2[idx0 + 2][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf15(_xaie, bufIdx++, arg1[idx0 + 2][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf16(_xaie, bufIdx++, arg2[idx0][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf17(_xaie, bufIdx++, arg0[idx0 + 2][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf19(_xaie, bufIdx++, arg1[idx0 + 2][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf20(_xaie, bufIdx++, arg2[idx0 + 2][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf22(_xaie, bufIdx++, arg1[idx0 + 2][idx1]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf23(_xaie, bufIdx++, arg2[idx0][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf24(_xaie, bufIdx++, arg0[idx0 + 2][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf26(_xaie, bufIdx++, arg1[idx0 + 2][idx1 + 2]);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      mlir_aie_write_buffer_buf27(_xaie, bufIdx++, arg2[idx0 + 2][idx1 + 2]);



  printf("Start cores...\n");
  mlir_aie_start_cores(_xaie);


  if (mlir_aie_acquire_lock_24_2_15(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired lock0 (1) in tile (1,4). Done.");
  else
    printf("Timed out (1000) while trying to acquire lock14_0 (1).");
  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      arg0[idx0][idx1] = mlir_aie_read_buffer_buf4(_xaie, bufIdx++);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      arg0[idx0][idx1 + 2] = mlir_aie_read_buffer_buf11(_xaie, bufIdx++);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      arg0[idx0 + 2][idx1] = mlir_aie_read_buffer_buf18(_xaie, bufIdx++);

  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 2; ++idx0)
    for (int64_t idx1 = 0; idx1 < 2; ++idx1)
      arg0[idx0 + 2][idx1 + 2] = mlir_aie_read_buffer_buf25(_xaie, bufIdx++);



  mlir_aie_deinit_libxaie(_xaie);

  printf("Complete compute.\n");
  return 1;
}
