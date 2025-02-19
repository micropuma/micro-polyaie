//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_INITALLPASSES_H
#define POLYAIE_INITALLPASSES_H

// 先注释AIE部分，后续再添加
#include "polyaie/AIE/Dialect/AIEVec/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"
#include "polyaie/Transforms/Passes.h"

namespace mlir {
namespace polyaie {

// Add all the related passes.
inline void registerAllPasses() {
  // MLIR passes.
  mlir::registerAllPasses();

  // PolyAIE passes.
  polyaie::registerPolyAIEPasses();

  // AIE passes.
  // 先注释AIE部分，后续再添加
  xilinx::AIE::registerAIEPasses();
  xilinx::aievec::registerAIEVecPasses();
}

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_INITALLPASSES_H
