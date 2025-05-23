//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

// 
#include "polyaie/Exporters.h"

int main(int argc, char **argv) {
  // 
  mlir::polyaie::registerPolyAIEExporters();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "PolyAIE Translation Tool"));
}
