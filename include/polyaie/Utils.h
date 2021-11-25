//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_UTILS_H
#define POLYAIE_UTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace polyaie {

/// Inference the buffer offsets from the input `type`.
SmallVector<int64_t, 4> getBufferOffsets(MemRefType type);

unsigned getCol(Operation *call);
unsigned getRow(Operation *call);

bool haveShareableBuffer(unsigned srcCol, unsigned srcRow, unsigned tgtCol,
                         unsigned tgtRow);

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_UTILS_H
