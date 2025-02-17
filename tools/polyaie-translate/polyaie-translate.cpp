//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

// # 先注释AIE部分，后续再添加
// #include "polyaie/Exporters.h"

int main(int argc, char **argv) {
  // # 先注释AIE部分，后续再添加
  // mlir::polyaie::registerPolyAIEExporters();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "PolyAIE Translation Tool"));
}
