//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_EXPORTERS_H
#define POLYAIE_EXPORTERS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace polyaie {

LogicalResult exportHostKernel(ModuleOp module, llvm::raw_ostream &os);

void registerExportHostKernel();

//===----------------------------------------------------------------------===//
// Base classes of exporters
//===----------------------------------------------------------------------===//

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various exporters.
class ExporterState {
public:
  explicit ExporterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

private:
  ExporterState(const ExporterState &) = delete;
  void operator=(const ExporterState &) = delete;
};

/// This is the base class for all of the exporters.
class ExporterBase {
public:
  explicit ExporterBase(ExporterState &state) : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  ExporterState &state;

  // The stream to emit to.
  raw_ostream &os;

private:
  ExporterBase(const ExporterBase &) = delete;
  void operator=(const ExporterBase &) = delete;
};

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_EXPORTERS_H
