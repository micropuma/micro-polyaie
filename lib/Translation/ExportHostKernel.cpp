//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/AIE/AIEDialect.h"

#include "polyaie/Exporters.h"
#include "polyaie/InitAllDialects.h"
#include "polyaie/Utils.h"

using namespace mlir;
using namespace polyaie;
using namespace dataflow;

// 
using namespace xilinx::AIE;

static llvm::cl::opt<bool>
    dryRunHostKernel("dry-run-host-kernel",
                     llvm::cl::desc("run the host kernel without real data"),
                     llvm::cl::init(false));

static llvm::cl::opt<bool>
    debugTile("debug-tile", llvm::cl::desc("print out the tile status"),
              llvm::cl::init(false));

static StringRef emitType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.getWidth() != 32 || !intType.isSignless())
      return "UNKNOWN_TYPE";
    return "int32_t";
  } else if (auto floatType = type.dyn_cast<Float32Type>())
    return "float";
  return "UNKNOWN_TYPE";
}

namespace {
class HostKernelExporter : public ExporterBase {
public:
  explicit HostKernelExporter(ExporterState &state) : ExporterBase(state) {}
  void exportHostKernel(ModuleOp mod);
  void emitHostDMA(dataflow::HostDMAOp hostDMA, unsigned argIdx);
};
} // namespace

void HostKernelExporter::emitHostDMA(dataflow::HostDMAOp hostDMA,
                                     unsigned argIdx) {
  auto isDDR = hostDMA.kind() == HostDMAKind::HostToDDR ||
               hostDMA.kind() == HostDMAKind::DDRToHost;
  auto isWrite = hostDMA.kind() == HostDMAKind::HostToAIE ||
                 hostDMA.kind() == HostDMAKind::HostToDDR;

  auto buf = isWrite ? hostDMA.target() : hostDMA.source();
  auto bufType = buf.getType().dyn_cast<MemRefType>();
  auto bufOp = buf.getDefiningOp();

  auto bufName = bufOp->getAttrOfType<StringAttr>("sym_name").getValue();
  auto bufSize = bufType.getNumElements();

  if (isDDR) {
    // Navigate to the external memory address.
    indent() << emitType(bufType.getElementType()) << " *" << bufName
             << "_ptr = (" << emitType(bufType.getElementType())
             << " *)mmap(NULL, "
             << bufType.getNumElements() * bufType.getElementTypeBitWidth() / 8
             << ", PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x"
             << llvm::utohexstr(cast<ExternalBufferOp>(bufOp).address())
             << ");\n";
  }

  // Generate the loop head if buffer size is not 1.
  // TODO: Make this more robust. Now we assume we won't read and write buffer
  // simultaneously in the host kernel.
  if (bufSize != 1) {
    indent() << "bufIdx = 0;\n";
    unsigned ivIdx = 0;
    for (auto dimSize : bufType.getShape()) {
      auto ivName = "idx" + std::to_string(ivIdx++);
      indent() << "for (int64_t " << ivName << " = 0; " << ivName << " < "
               << dimSize << "; ++" << ivName << ")\n";
      addIndent();
    }
  }

  // A helper to emit the argument fetching verbatim.
  auto emitArgValue = [&]() {
    os << "arg" << argIdx;
    if (bufSize != 1)
      for (unsigned ivIdx = 0; ivIdx < bufType.getRank(); ++ivIdx) {
        reduceIndent();
        os << "[idx" << ivIdx;
        auto offset = hostDMA.offsets()[ivIdx].cast<IntegerAttr>().getInt();
        if (offset)
          os << " + " << offset << "]";
        else
          os << "]";
      }
  };

  auto bufIdx = bufSize != 1 ? "bufIdx++" : "0";
  if (isDDR) {
    // Generate the DMA for external memories.
    if (isWrite) {
      indent() << bufName << "_ptr[" << bufIdx << "] = ";
      emitArgValue();
      os << ";\n\n";
    } else {
      indent();
      emitArgValue();
      os << " = " << bufName << "_ptr[" << bufIdx << "];\n\n";
    }
  } else {
    // Generate the DMA for on-chip buffers.
    if (isWrite) {
      indent() << "mlir_aie_write_buffer_" << bufName << "(_xaie, " << bufIdx
               << ", ";
      emitArgValue();
      os << ");\n\n";
    } else {
      indent();
      emitArgValue();
      os << " = mlir_aie_read_buffer_" << bufName << "(_xaie, " << bufIdx
         << ");\n\n";
    }
  }
}

void HostKernelExporter::exportHostKernel(ModuleOp mod) {
  os << R"XXX(
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

)XXX";

  // Generate the host kernel signature.
  if (!mod.getName().hasValue())
    emitError(mod.getLoc(), "top module is not named");
  os << "int main() {\n";
  addIndent();

  // Each allocated global memory is an argument of the kernel.
  // Each allocated global memory is an argument of the kernel.
  SmallVector<memref::AllocOp, 8> allocs(mod.getOps<memref::AllocOp>());
  unsigned argIdx = 0;
  DenseMap<Value, unsigned> argIdxMap;
  for (auto alloc : allocs) {
    auto type = alloc.getType();
    argIdxMap[alloc.getResult()] = argIdx;

    // Emit argument declaration
    indent() << emitType(type.getElementType()) << " arg" << argIdx++;
    if (type.getNumElements() != 1)
      for (auto dimSize : type.getShape())
        os << "[" << dimSize << "]";
    os << ";\n";

    // Emit initialization code
    indent() << "for (int64_t idx0 = 0; idx0 < " << type.getShape()[0] << "; ++idx0) {\n";
    addIndent();
    for (size_t i = 1; i < type.getShape().size(); ++i) {
      indent() << "for (int64_t idx" << i << " = 0; idx" << i << " < " << type.getShape()[i] << "; ++idx" << i << ") {\n";
      addIndent();
    }

    indent() << "arg" << argIdx - 1 << "[";
    for (size_t i = 0; i < type.getShape().size(); ++i) {
      os << "idx" << i;
      if (i < type.getShape().size() - 1) os << "][";
      else os << "]";
    }
    os << " = ";
    
    // Set initial values based on the argument index
    os << "0.0";  // You can customize this logic for different initialization values.

    os << ";\n";
    for (size_t i = 1; i < type.getShape().size(); ++i) {
      reduceIndent();
      indent() << "}\n";
    }
    reduceIndent();
    indent() << "}\n";
  }

  // Generate the kernel body.
  os << R"XXX(
  printf("Configure AIE array...\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);


  printf("Initialize buffers...\n");

)XXX";

  // Collect tile operations.
  SmallVector<TileOp, 32> tiles;
  SmallVector<UseLockOp, 16> acquires;
  SmallVector<UseLockOp, 16> releases;
  for (auto tile : mod.getOps<TileOp>())
    if (auto core = tile.getCoreOp()) {
      tiles.push_back(tile);

      if (!tile.isShimNOCTile())
        continue;
      for (auto useLock : core.getOps<UseLockOp>()) {
        if (useLock.release())
          releases.push_back(useLock);
        else if (useLock.value() == 1)
          acquires.push_back(useLock);
      }
    }

  // TODO: The lock user collection is highly experimental! We need a more
  // systematic runtime IR.
  if (acquires.empty()) {
    mod.walk([&](UseLockOp useLock) {
      if (useLock->getAttr("polyaie.runtime"))
        acquires.push_back(useLock);
    });
  }

  // Collect memory copy operations.
  SmallVector<dataflow::HostDMAOp, 8> loads;
  SmallVector<dataflow::HostDMAOp, 8> stores;

  for (auto hostDMA : mod.getOps<dataflow::HostDMAOp>())
    if (hostDMA.kind() == HostDMAKind::HostToAIE ||
        hostDMA.kind() == HostDMAKind::HostToDDR)
      loads.push_back(hostDMA);
    else if (hostDMA.kind() == HostDMAKind::AIEToHost ||
             hostDMA.kind() == HostDMAKind::DDRToHost)
      stores.push_back(hostDMA);
    else
      llvm_unreachable("invalid host DMA kind");

  if (!dryRunHostKernel) {
    // Open the ddr device.
    indent() << "int fd = open(\"/dev/mem\", O_RDWR | O_SYNC);\n";
    indent() << "assert(fd != -1 && \"memory is not available\");\n\n";

    // Clear the local memory of all tiles.
    for (auto tile : tiles)
      if (!tile.isShimNOCorPLTile())
        indent() << "mlir_aie_clear_tile_memory(_xaie, " << tile.col() << ", "
                 << tile.row() << ");\n";
    os << "\n";

    // Initialize local memories.
    indent() << "unsigned bufIdx;\n\n";
    for (auto load : loads)
      emitHostDMA(load, argIdxMap[load.source()]);
  }

  // Configure the iteration number buffer of each AIE.
  for (auto buf : mod.getOps<BufferOp>())
    if (buf->getAttr("polyaie.iter_num_buf"))
      indent() << "mlir_aie_write_buffer_" << buf.name().getValue()
               << "(_xaie, 0, iter_num);\n";
  os << "\n";

//   // Create a "results" array to indicate the completion of the kernel and then
//   // start the execution.
//   indent() << "bool results[" << acquires.size() << "];\n";

  os << R"XXX(
  printf("Start cores...\n");
  mlir_aie_start_cores(_xaie);

)XXX";

  if (debugTile) {
    // Debug the tiles.
    indent() << "sleep(1);\n";
    for (auto tile : tiles)
      if (!tile.isShimNOCorPLTile()) {
        indent() << "mlir_aie_print_tile_status(_xaie, " << tile.col() << ", "
                 << tile.row() << ");\n";
        indent() << "mlir_aie_print_dma_status(_xaie, " << tile.col() << ", "
                 << tile.row() << ");\n";
        indent() << "printf(\"\\n\");\n";
      }
    os << "\n";
  }

  // Release locks to 1.
  for (auto release : releases)
    if (release.value() == 1) {
      auto lock = release.lock().getDefiningOp<LockOp>();
      auto tile = lock.tile().getDefiningOp<TileOp>();

      indent() << "mlir_aie_release_lock(_xaie, " << tile.col() << ", "
               << tile.row() << ", " << lock.lockID() << ", 1, 0);\n";
    }
  os << "\n";

  if (debugTile) {
    // Debug the tiles.
    indent() << "sleep(1);\n";
    for (auto tile : tiles)
      if (!tile.isShimNOCorPLTile()) {
        indent() << "mlir_aie_print_tile_status(_xaie, " << tile.col() << ", "
                 << tile.row() << ");\n";
        indent() << "mlir_aie_print_dma_status(_xaie, " << tile.col() << ", "
                 << tile.row() << ");\n";
        indent() << "printf(\"\\n\");\n";
      }
    os << "\n";
  }

  // Iterate until all locks in "results" are acquired.
  indent() << "if (mlir_aie_acquire_lock_24_2_15(_xaie, 1, 1000) == XAIE_OK)\n";
  addIndent();
  indent() << "printf(\"Acquired lock0 (1) in tile (1,4). Done.\");\n";
  reduceIndent();
  indent() << "else\n";
  addIndent();
  indent() << "printf(\"Timed out (1000) while trying to acquire lock14_0 (1).\");\n";

  // for (auto load : loads) {
  //   if (!load->getAttr("polyaie.load_each_iter"))
  //     continue;

  //   auto tile = load.target().getDefiningOp<BufferOp>().getTileOp();
  //   indent() << "if (mlir_aie_acquire_lock(_xaie, " << tile.col() << ", "
  //            << tile.row() << ", 15, 1, 0)) {\n";
  //   addIndent();

  //   if (!dryRunHostKernel)
  //     emitHostDMA(load, argIdxMap[load.source()]);

  //   indent() << "mlir_aie_release_lock(_xaie, " << tile.col() << ", "
  //            << tile.row() << ", 15, 0, 0);\n";
  //   reduceIndent();
  //   indent() << "}\n";
  // }

  reduceIndent();

  if (!dryRunHostKernel) {
    // Write back results from local to global memory.
    for (auto hostDMA : stores)
      emitHostDMA(hostDMA, argIdxMap[hostDMA.target()]);
  }

  // Release locks back to 0.
  for (auto release : releases)
    if (release.value() == 0) {
      auto lock = release.lock().getDefiningOp<LockOp>();
      auto tile = lock.tile().getDefiningOp<TileOp>();

      indent() << "mlir_aie_release_lock(_xaie, " << tile.col() << ", "
               << tile.row() << ", " << lock.lockID() << ", 0, 0);\n";
    }
  os << "\n";

  os << R"XXX(
  mlir_aie_deinit_libxaie(_xaie);

  printf("Complete compute.\n");
  return 1;
)XXX";

  os << "}\n";
}

LogicalResult polyaie::exportHostKernel(ModuleOp module, raw_ostream &os) {
  ExporterState state(os);
  HostKernelExporter(state).exportHostKernel(module);
  return failure(state.encounteredError);
}
